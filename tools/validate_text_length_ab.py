#!/usr/bin/env python3
"""Single-file OACIR latent text-length A/B validation, without training.

Install under OACIR/tools/. Default: Product, SAME checkpoint at 32/64 tokens.
Image-only raw cache/gallery features are shared. Official query inference,
candidate generation and reranking are run afresh for EACH text length.
No existing source/checkpoint/cache is modified. No Excel dependency.

Examples:
  python tools/validate_text_length_ab.py --checkpoint /path/to/model.pt
  python tools/validate_text_length_ab.py --checkpoint /path/to/model.pt --max-queries 16
  python tools/validate_text_length_ab.py --self-test

Default checkpoint is the earlier split-mask e34 run. Missing defaults fail
explicitly; the script NEVER substitutes a margin/FiLM checkpoint silently.
"""

import argparse
import copy
import csv
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import random
import subprocess
import sys
import tempfile
from datetime import datetime


DEFAULT_CHECKPOINT = (
    "./checkpoints/latent_split_mask_from_pretrain/"
    "OACIR_Union_finetune_oacir_latent_split_mask_2026-10-23_02:40:31/"
    "saved_models/adafocal_finetune_best_identity.pt"
)
SUPPORTED = (
    "oacir_latent", "oacir_latent_split_mask", "oacir_latent_margin_split",
    "oacir_latent_identity_split", "oacir_latent_slot_film",
)
PARAMETERS = {
    "core_matcher_temp": .07, "core_matcher_topk": 50,
    "core_matcher_lambda_id": 1.25, "core_matcher_fusion_weight": .4,
    "core_matcher_cycle_weight": .25, "core_matcher_text_weight": .25,
    "core_matcher_bbox_floor": .05, "core_matcher_query_chunk_size": 2,
    "region_topk": 16, "region_topk_max": 96, "region_area_scale": 1.5,
    "region_spatial_kernel": 3, "region_spatial_weight": .15,
    "region_temperature": .07,
}
METRICS = ("R_ID@1", "R_ID@3", "R_ID@5", "R@1", "R@5", "R@10", "R@50")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model", choices=SUPPORTED, help="Normally inferred from checkpoint sidecar")
    parser.add_argument("--training-config", help="Override training_hyperparameters.json location")
    parser.add_argument("--repo-root")
    parser.add_argument("--dataset", choices=["Product", "Fashion", "Car", "Landmark"], default="Product")
    parser.add_argument("--lengths", nargs=2, type=int, default=[32, 64], metavar=("A", "B"))
    parser.add_argument("--gpu", type=int, help="One physical GPU; otherwise respect CUDA_VISIBLE_DEVICES or select a relatively idle card")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-root")
    parser.add_argument("--val-feature-cache", help="Raw IMAGE cache; accepts {variant_lower}")
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--gallery-batch-size", type=int, default=128)
    parser.add_argument("--gallery-chunk-size", type=int, default=2048)
    parser.add_argument("--query-chunk-size", type=int, help="Optional compute-only override; normally inherited")
    parser.add_argument("--max-queries", type=int, default=0, help="First N queries for smoke tests; 0=full domain")
    parser.add_argument("--output-dir", help="New directory only; refuses to overwrite existing results")
    parser.add_argument("--self-test", action="store_true", help="Synthetic CPU checks; no checkpoint, GPU, or training")
    args = parser.parse_args()
    if not (2 <= args.lengths[0] < args.lengths[1]):
        parser.error("--lengths needs two increasing limits >=2, e.g. 32 64")
    if args.max_queries < 0 or min(args.query_batch_size, args.gallery_batch_size, args.gallery_chunk_size) < 1:
        parser.error("Batch/chunk sizes must be positive; max-queries must be nonnegative")
    return args


def find_repo(explicit=None):
    options = [Path(explicit)] if explicit else [Path.cwd(), Path(__file__).resolve().parent,
                                               Path(__file__).resolve().parent.parent]
    for root in options:
        if (root / "data_utils.py").is_file() and (root / "lavis").is_dir():
            return root.resolve()
    raise FileNotFoundError("Run from the OACIR repository or pass --repo-root")


def set_runtime(args):
    # Avoid AF_UNIX path-too-long failures in DataLoader workers.
    os.environ["TMPDIR"] = "/tmp"
    tempfile.tempdir = "/tmp"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.device == "cpu" or args.self_test:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return
    visible = str(args.gpu) if args.gpu is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
        cards = []
        for line in result.stdout.splitlines():
            index, used, free, util = map(int, line.split(","))
            if used <= 12000 and free >= 10000 and util <= 70:
                cards.append((-free, util, index, used))
        if not cards:
            raise RuntimeError("No relatively idle GPU with >=10000 MiB free; retry later or specify --gpu")
        neg_free, util, index, used = min(cards)
        visible = str(index)
        print(f"Selected ONE GPU {index}: used={used} MiB, free={-neg_free} MiB, utilization={util}%", flush=True)
    if not visible.strip() or "," in visible:
        raise ValueError("Exactly one GPU is permitted; set --gpu INDEX or one CUDA_VISIBLE_DEVICES value")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible


def reset_seed():
    import numpy as np
    import torch
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)


def read_checkpoint(path):
    import torch
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def load_model(args, checkpoint):
    import torch
    from omegaconf import OmegaConf
    from lavis.common.registry import registry
    from lavis.models import load_preprocess
    config_path = Path(args.training_config) if args.training_config else checkpoint.parent.parent / "training_hyperparameters.json"
    training = json.loads(config_path.read_text()) if config_path.is_file() else {}
    name = args.model or training.get("blip_model_name")
    if name not in SUPPORTED:
        raise ValueError("Cannot infer a supported latent model; pass --model and, if needed, --training-config")
    if args.model and training.get("blip_model_name") not in (None, args.model):
        raise ValueError("--model conflicts with training metadata; do not silently change model variants")
    importlib.import_module("lavis.models.blip2_models.blip2_qformer_" + name)
    cls = registry.get_model_class(name)
    config = OmegaConf.load(cls.default_config_path(training.get("vit_backbone", "pretrain")))
    mc = config.model.copy()
    mc.load_pretrained = False
    mc.load_finetuned = False
    model = cls.from_config(mc)
    payload = read_checkpoint(checkpoint)
    states = []
    is_state = lambda x: isinstance(x, dict) and bool(x) and all(torch.is_tensor(v) for v in x.values())
    if is_state(payload):
        state, epoch = payload, None
    elif isinstance(payload, dict):
        states = [v for v in payload.values() if is_state(v)]
        if len(states) != 1:
            raise ValueError("Expected exactly one full model state dictionary")
        state, epoch = states[0], payload.get("epoch")
    else:
        raise ValueError("Unsupported checkpoint format")
    model.load_state_dict(state, strict=True)
    del payload, state, states
    params = {key: training.get(key, value) for key, value in PARAMETERS.items()}
    if args.query_chunk_size is not None:
        if args.query_chunk_size < 1:
            raise ValueError("query-chunk-size must be positive")
        params["core_matcher_query_chunk_size"] = args.query_chunk_size
    model.set_latent_config(**params)
    maximum = getattr(model.Qformer.config, "max_position_embeddings", None)
    embeddings = getattr(getattr(model.Qformer, "bert", None), "embeddings", None)
    positions = getattr(embeddings, "position_embeddings", None)
    if positions is not None:
        maximum = positions.num_embeddings
    if maximum is None or max(args.lengths) > maximum:
        raise ValueError(f"Requested text length exceeds/unverifiably fits position capacity {maximum}")
    if getattr(model.tokenizer, "truncation_side", "right") != "right":
        raise ValueError("This diagnostic expects right-side truncation")
    original_length = model.max_txt_len
    model = model.to("cuda:0" if args.device == "cuda" else "cpu").eval()
    _, processors = load_preprocess(config.preprocess)
    return model, processors, training, {
        "model": name, "checkpoint_epoch": epoch,
        "training_config": str(config_path.resolve()) if config_path.is_file() else None,
        "checkpoint_default_max_txt_len": original_length,
        "position_embedding_capacity": maximum,
        "inference_parameters": params,
        "inference_parameters_source": "training sidecar with documented script defaults for absent fields",
    }


def prepare(args, model, processors, training):
    from data_utils import OACIRRDataset, targetpad_transform, squarepad_transform
    from evaluate import extract_index_blip_features_from_raw_cache
    transform = training.get("transform", "targetpad")
    ratio = training.get("target_ratio", 1.25)
    if transform == "targetpad":
        preprocess = targetpad_transform(ratio, 224)
    elif transform == "squarepad":
        preprocess = squarepad_transform(224)
    else:
        raise ValueError("Unsupported checkpoint image transform: " + str(transform))
    data_root = args.data_root or training.get("data_root", "./Datasets/OACIRR")
    relative = OACIRRDataset(data_root, args.dataset, "val", "relative", preprocess,
                             highlight_inference=True, text_entity=training.get("text_entity", False))
    relative = copy.copy(relative)
    full_query_count = len(relative)
    if args.max_queries:
        relative.quadruples = relative.quadruples[:args.max_queries]
    if not len(relative):
        raise ValueError("Empty validation domain")
    classic = OACIRRDataset(data_root, args.dataset, "val", "classic", preprocess)
    cache = args.val_feature_cache or training.get("val_feature_cache")
    if not cache:
        if transform != "targetpad" or ratio != 1.25 or training.get("vit_backbone", "pretrain") != "pretrain":
            raise ValueError("Pass a matching --val-feature-cache for this transform/backbone")
        cache = "./cache/oacirr_{variant_lower}_val_vitg_targetpad125_raw_fp16.pt"
    cache = cache.format(variant=args.dataset, variant_lower=args.dataset.lower())
    if not Path(cache).is_file():
        raise FileNotFoundError("Missing raw IMAGE cache: " + str(Path(cache).resolve()))
    texts = []
    for i, q in enumerate(relative.quadruples):
        text = q["modification_text_mllm"]
        if relative.text_entity:
            text = relative.prompts[i % relative.num_prompts].format(entity=q["object_category"]) + ", " + text
        texts.append(processors["eval"](text))
    token_counts = []
    for start in range(0, len(texts), 512):
        encoded = model.tokenizer(texts[start:start + 512], truncation=False, padding=False,
                                  add_special_tokens=True, return_attention_mask=False,
                                  return_token_type_ids=False)
        token_counts.extend(len(tokens) for tokens in encoded["input_ids"])
    print(f"{args.dataset}: {len(texts)} queries; {sum(n > args.lengths[0] for n in token_counts)} "
          f"exceed {args.lengths[0]} tokens; {sum(n > args.lengths[1] for n in token_counts)} "
          f"exceed {args.lengths[1]} tokens (including special tokens)", flush=True)
    # This official loader only reads raw IMAGE cache and rebuilds image-only
    # Q-Former features with the current checkpoint; no query feature cache.
    gallery, raw, names = extract_index_blip_features_from_raw_cache(
        classic, model, cache, save_memory=True, batch_size=args.gallery_batch_size)
    if len(names) != len(set(names)) or len(names) < 2:
        raise ValueError("Gallery names must be unique and contain at least two images")
    return relative, gallery, raw, list(names), texts, token_counts, {
        "data_root": str(Path(data_root).resolve()), "raw_image_cache": str(Path(cache).resolve()),
        "full_domain_queries": full_query_count, "evaluated_queries": len(texts),
        "is_full_validation": len(texts) == full_query_count, "gallery_images": len(names),
    }


def query_hits(top_indices, target_indices, gallery_identities):
    """Pure evaluation: labels never influence features, candidates or scores."""
    rows = []
    for top, target in zip(top_indices, target_indices):
        exact = [i == target for i in top]
        identity = [gallery_identities[i] == gallery_identities[target] for i in top]
        row = {f"R@{k}": any(exact[:k]) for k in [1, 5, 10, 50]}
        row.update({f"R_ID@{k}": any(identity[:k]) for k in [1, 3, 5]})
        row.update(top1=top[0], top5=top[:5], target_rank_top50=(top.index(target) + 1 if target in top else None))
        rows.append(row)
    return rows


def group_masks(counts, lengths):
    a, b = lengths
    return {"all": [True] * len(counts), f"truncated_at_{a}": [n > a for n in counts],
            f"not_truncated_at_{a}": [n <= a for n in counts],
            f"still_truncated_at_{b}": [n > b for n in counts]}


def metrics(rows, mask):
    selected = [r for r, keep in zip(rows, mask) if keep]
    count = len(selected)
    result = {"queries": count, **{k: 100. * sum(r[k] for r in selected) / count if count else None for k in METRICS}}
    result["arithmetic_mean"] = sum(result[f"R@{k}"] for k in [1, 5, 10, 50]) / 4. if count else None
    return result


def comparisons(before, after, masks):
    if len(before) != len(after):
        raise ValueError("A/B query counts differ")
    output = {}
    for name, mask in masks.items():
        a, b = metrics(before, mask), metrics(after, mask)
        counts = {"top1_changed": sum(x["top1"] != y["top1"] for x, y, keep in zip(before, after, mask) if keep)}
        for label, key in [("exact", "R@1"), ("identity", "R_ID@1")]:
            counts[label + "_improved"] = sum(not x[key] and y[key] for x, y, keep in zip(before, after, mask) if keep)
            counts[label + "_damaged"] = sum(x[key] and not y[key] for x, y, keep in zip(before, after, mask) if keep)
        output[name] = {"A": a, "B": b,
                        "delta_B_minus_A_pp": {k: b[k] - a[k] if a[k] is not None else None for k in [*METRICS, "arithmetic_mean"]},
                        "paired_top1_changes": counts}
    return output


def evaluate_length(args, model, relative, gallery, raw, names, processors, texts, length):
    import torch
    from evaluate import generate_oacirr_val_predictions_latent
    model.max_txt_len = int(length)
    reset_seed()
    print(f"\n=== SAME checkpoint, max_txt_len={length}: recomputing query features, candidates and reranking ===", flush=True)
    result = generate_oacirr_val_predictions_latent(
        blip_model=model, relative_val_dataset=relative, index_names=names,
        index_features=gallery, index_raw_embeds=raw, txt_processors=processors,
        save_memory=True, latent_gallery_chunk_size=args.gallery_chunk_size,
        eval_batch_size=args.query_batch_size)
    scores, references, targets, captions = result[:4]
    expected_refs = [q["reference"] for q in relative.quadruples]
    expected_targets = [q["target"] for q in relative.quadruples]
    if list(references) != expected_refs or list(targets) != expected_targets or list(captions) != texts:
        raise ValueError("Official loader changed/skipped/reordered queries or text preprocessing; refusing unpaired A/B")
    if tuple(scores.shape) != (len(texts), len(names)):
        raise ValueError("Unexpected official prediction shape")
    lookup = {name: i for i, name in enumerate(names)}
    target_indices = [lookup[name] for name in targets]
    identities = [PurePosixPath(str(relative.name_to_relpath[name]).replace("\\", "/")).parent.as_posix() for name in names]
    top_rows = []
    for start in range(0, len(references), 256):
        end = min(start + 256, len(references))
        batch = scores[start:end]
        if torch.isnan(batch).any():
            raise ValueError("NaN prediction scores")
        indices = torch.tensor([lookup[n] for n in references[start:end]], device=batch.device)
        batch[torch.arange(end - start, device=batch.device), indices] = -torch.inf
        top_rows.extend(batch.topk(min(50, len(names) - 1), dim=1).indices.cpu().tolist())
    rows = query_hits(top_rows, target_indices, identities)
    del result, scores
    return rows


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def write_queries(path, relative, names, texts, counts, lengths, before, after):
    fields = ["query_index", "reference", "target", "raw_text", "processed_text", "full_token_count",
              "truncated_A", "truncated_B", "top1_changed", "A_top1", "B_top1", "A_top5", "B_top5",
              "A_exact_correct", "B_exact_correct", "A_identity_correct", "B_identity_correct",
              "A_target_rank_top50", "B_target_rank_top50"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, (q, a, b) in enumerate(zip(relative.quadruples, before, after)):
            writer.writerow({"query_index": i, "reference": q["reference"], "target": q["target"],
                             "raw_text": q["modification_text_mllm"], "processed_text": texts[i],
                             "full_token_count": counts[i], "truncated_A": counts[i] > lengths[0],
                             "truncated_B": counts[i] > lengths[1], "top1_changed": a["top1"] != b["top1"],
                             "A_top1": names[a["top1"]], "B_top1": names[b["top1"]],
                             "A_top5": json.dumps([names[n] for n in a["top5"]]),
                             "B_top5": json.dumps([names[n] for n in b["top5"]]),
                             "A_exact_correct": a["R@1"], "B_exact_correct": b["R@1"],
                             "A_identity_correct": a["R_ID@1"], "B_identity_correct": b["R_ID@1"],
                             "A_target_rank_top50": a["target_rank_top50"],
                             "B_target_rank_top50": b["target_rank_top50"]})


def self_test():
    identities = ["a", "a", "b", "b", "c", "d"]
    target = [1, 2, 4, 3]
    a = query_hits([[0, 1, 2], [2, 3, 1], [5, 4, 1], [3, 2, 1]], target, identities)
    b = query_hits([[1, 0, 2], [3, 2, 1], [5, 4, 1], [0, 3, 2]], target, identities)
    masks = group_masks([40, 10, 70, 32], [32, 64])
    result = comparisons(a, b, masks)
    assert result["all"]["A"]["R@1"] == 50.
    assert result["all"]["B"]["R@1"] == 25.
    assert result["truncated_at_32"]["delta_B_minus_A_pp"]["R@1"] == 50.
    assert result["not_truncated_at_32"]["delta_B_minus_A_pp"]["R@1"] == -100.
    assert result["all"]["paired_top1_changes"] == {"top1_changed": 3, "exact_improved": 1, "exact_damaged": 2, "identity_improved": 0, "identity_damaged": 1}
    assert result["still_truncated_at_64"]["A"]["queries"] == 1
    empty = metrics(a, [False] * 4)
    assert empty["queries"] == 0 and empty["R@1"] is None
    assert metrics(query_hits([[1]], [1], identities), [True])["R@50"] == 100.
    print("PASS exact-vs-identity metrics, short galleries, paired changes, fixed groups and empty groups")
    print("SELF_TEST_PASS (synthetic helpers only; no checkpoint/GPU inference)")


def main():
    args = arguments()
    if args.self_test:
        self_test()
        return
    repo = find_repo(args.repo_root)
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = repo / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}\nPass --checkpoint YOUR_V6_CHECKPOINT; no fallback to another method is permitted.")
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    set_runtime(args)
    import torch
    torch.set_num_threads(4)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this Python environment")
    reset_seed()
    output = Path(args.output_dir).resolve() if args.output_dir else repo / "checkpoints/text_length_ab" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "metadata": {
        "checkpoint": str(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
        "repo_root": str(repo), "dataset": args.dataset, "lengths": args.lengths,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "only_changed_model_setting": "max_txt_len; no training or checkpoint modification",
        "inference": "official generate_oacirr_val_predictions_latent; native model scoring, no parameter search",
        "caching": "raw images/gallery features shared; query features/candidates/reranking recomputed for each length",
        "grouping": "same fixed query groups defined by processed token count INCLUDING special tokens",
        "num_workers": "official evaluator DataLoader default (currently 6)",
    }}
    model = None
    original_length = None
    try:
        with torch.inference_mode():
            model, processors, training, meta = load_model(args, checkpoint)
            original_length = model.max_txt_len
            report["metadata"].update(meta)
            relative, gallery, raw, names, texts, counts, meta = prepare(args, model, processors, training)
            report["metadata"].update(meta)
            masks = group_masks(counts, args.lengths)
            report["token_statistics"] = {"max": max(counts), "mean": sum(counts) / len(counts),
                                          "group_counts": {k: sum(v) for k, v in masks.items()}}
            before = evaluate_length(args, model, relative, gallery, raw, names, processors, texts, args.lengths[0])
            report["A_completed"] = {group: metrics(before, mask) for group, mask in masks.items()}
            write_json(output / "report.json", report)
            after = evaluate_length(args, model, relative, gallery, raw, names, processors, texts, args.lengths[1])
            report["groups"] = comparisons(before, after, masks)
            write_queries(output / "query_comparison.csv", relative, names, texts, counts, args.lengths, before, after)
            report["status"] = "complete"
            write_json(output / "report.json", report)
            for group, result in report["groups"].items():
                print(f"\n{group}: n={result['A']['queries']}", flush=True)
                for key in ["R_ID@1", "R@1", "R@5"]:
                    print(key, "A=", result["A"][key], "B=", result["B"][key],
                          "delta_pp=", result["delta_B_minus_A_pp"][key], flush=True)
                print("paired_top1_changes:", result["paired_top1_changes"], flush=True)
            print("\nResults:", output, flush=True)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(output / "report.json", report)
        raise
    finally:
        if model is not None and original_length is not None:
            model.max_txt_len = original_length


if __name__ == "__main__":
    main()
