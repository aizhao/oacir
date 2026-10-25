#!/usr/bin/env python3
"""Single-file, inference-only R_comp ablation for the existing dual-mask models.

A: original model, visual correspondence + cycle + text_weight * cosine.
B: R_comp = cosine(composed_feature, target_patch), coefficient exactly 1.0.
Keep spatial smoothing, patch count, temperature, identity scores, candidate
set fixed. Optionally sweep final score weights without extra model forwards.
No source patching or training.
Run from an OACIR repo, placing this file in tools/ or the repo root.
"""

import argparse
import csv
import importlib
import inspect
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SUPPORTED = (
    "oacir_latent_split_mask",
    "oacir_latent_identity_split",
    "oacir_latent_margin_split",
)
RECALL_KEYS = ("R_ID@1", "R_ID@3", "R_ID@5", "R@1", "R@5", "R@10", "R@50")
DEFAULTS = {
    "core_matcher_temp": 0.07,
    "core_matcher_topk": 50,
    "core_matcher_lambda_id": 1.25,
    "core_matcher_fusion_weight": 0.4,
    "core_matcher_cycle_weight": 0.25,
    "core_matcher_text_weight": 0.25,
    "core_matcher_bbox_floor": 0.05,
    "core_matcher_query_chunk_size": 2,
    "region_topk": 16,
    "region_topk_max": 96,
    "region_area_scale": 1.5,
    "region_spatial_kernel": 3,
    "region_spatial_weight": 0.15,
    "region_temperature": 0.07,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", help="Existing trained checkpoint, required except --self-test")
    p.add_argument("--model", "--blip-model-name", dest="model", choices=SUPPORTED)
    p.add_argument("--dataset", "--datasets", nargs="+", default=["Car"],
                   choices=["Fashion", "Car", "Landmark", "Product"])
    p.add_argument("--repo-root", help="OACIR repo directory; normally discovered automatically")
    p.add_argument("--data-root")
    p.add_argument("--val-feature-cache", help="Raw val cache path/template with {variant_lower}")
    p.add_argument("--output-dir")
    p.add_argument("--fusion-weights", nargs="+", type=float,
                   help="Final reranker weights, e.g. 0.4 0.5 0.6 0.8; default: checkpoint value")
    p.add_argument("--lambda-ids", nargs="+", type=float,
                   help="Final identity weights, e.g. 0.75 1 1.25 1.5 2; default: checkpoint value")
    p.add_argument("--matcher-temps", nargs="+", type=float,
                   help="Final denominators; default: checkpoint value. Redundant with fusion weight")
    p.add_argument("--comp-score-weights", nargs="+", type=float, default=[1.0],
                   help="Multiplier on final S_comp only; does NOT change R_comp or M_comp")
    p.add_argument("--gpu", help="One physical GPU index; default respects CUDA_VISIBLE_DEVICES or selects idle GPU")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-queries", type=int, default=0,
                   help="Debug only: first N queries per domain, 0 = all")
    p.add_argument("--self-test", action="store_true", help="CPU semantic tests; no checkpoint/GPU/model download")
    return p.parse_args()


def find_repo(explicit=None):
    candidates = [Path(explicit)] if explicit else [Path.cwd(), Path(__file__).resolve().parent,
                                                   Path(__file__).resolve().parent.parent]
    for path in candidates:
        if (path / "data_utils.py").is_file() and (path / "lavis").is_dir():
            return path.resolve()
    raise FileNotFoundError("Cannot locate OACIR repo; pass --repo-root /path/to/oacir")


def configure_gpu(args):
    if args.self_test or args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return
    visible = args.gpu if args.gpu is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
        )
        cards = []
        for line in result.stdout.splitlines():
            index, memory, utilization = [int(x.strip()) for x in line.split(",")]
            if memory <= 12000 and utilization <= 70:
                cards.append((memory, utilization, index))
        if not cards:
            raise RuntimeError("No GPU meets <=12000 MiB / <=70% utilization; retry later or explicitly pass --gpu INDEX")
        memory, utilization, index = min(cards)
        visible = str(index)
        print(f"Selected physical GPU {index}: used={memory} MiB, utilization={utilization}%", flush=True)
    if not visible.strip() or "," in visible:
        raise ValueError("Use exactly one GPU via --gpu INDEX or CUDA_VISIBLE_DEVICES=INDEX")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    print(f"CUDA_VISIBLE_DEVICES={visible}", flush=True)


def init_runtime(repo):
    global torch, F
    sys.path.insert(0, str(repo))
    import torch
    from torch.nn import functional as F


def composition_only(candidate_tokens, composed_features, region_sizes,
                     spatial_kernel=3, spatial_weight=0.15, temperature=0.07):
    """Return S_comp and M_comp using response = cosine, no 0.25 multiplier."""
    if temperature <= 0 or spatial_kernel < 1 or spatial_kernel % 2 != 1:
        raise ValueError("Require positive temperature and positive odd spatial kernel")
    patches = F.normalize(candidate_tokens, dim=-1)
    composed = F.normalize(composed_features, dim=-1)
    response = torch.einsum("bd,bcnd->bcn", composed, patches)
    batch, candidates, count = response.shape
    grid = int(math.sqrt(count))
    has_grid = grid * grid == count
    if has_grid and spatial_kernel > 1:
        response = F.avg_pool2d(
            F.pad(response.reshape(-1, 1, grid, grid), (spatial_kernel // 2,) * 4,
                  mode="replicate"), spatial_kernel, stride=1,
        ).reshape(batch, candidates, count)
    if has_grid:
        yy, xx = torch.meshgrid(torch.arange(grid, device=response.device),
                                torch.arange(grid, device=response.device), indexing="ij")
        coords = torch.stack([yy.flatten(), xx.flatten()], dim=-1).to(response.dtype)
    selected = torch.zeros_like(response, dtype=torch.bool)
    for i in range(batch):
        size = min(max(int(region_sizes[i].item()), 1), count)
        rank = response[i]
        if has_grid and size < count:
            peaks = coords[rank.argmax(dim=-1)]
            distances = (coords[None] - peaks[:, None]).pow(2).sum(-1)
            distances = distances / max(float((grid - 1) ** 2), 1.0)
            rank = rank - float(spatial_weight) * distances
        selected[i].scatter_(1, rank.topk(size, dim=-1).indices, True)
    logits = (response / float(temperature)).masked_fill(~selected, torch.finfo(response.dtype).min)
    weights = F.softmax(logits, dim=-1)
    region = F.normalize(torch.einsum("bcn,bcnd->bcd", weights, patches), dim=-1)
    score = torch.einsum("bd,bcd->bc", composed, region)
    return score, weights


def first_stage_scores(model, reference_raw, gallery_features, captions, boxes, chunk_size):
    """Same computation as inference_composition_from_raw, query encoded once."""
    device = model._get_device()
    raw = reference_raw.to(device, non_blocking=True)
    normalized = model.ln_vision(raw.to(dtype=model.ln_vision.weight.dtype))
    query = model._compute_fusion_features(normalized, captions, boxes)
    chunks = []
    for start in range(0, len(gallery_features), chunk_size):
        gallery = gallery_features[start:start + chunk_size].to(device, non_blocking=True)
        chunks.append((model._compute_composition_scores(query, gallery) / model.temp).cpu())
    return torch.cat(chunks, dim=1)


def compare_chunk(model, reference_raw, target_raw, boxes, text_features):
    """Reuse exact A identity scores; replace only composition for B."""
    original = model._compute_core_matcher_candidate_scores(
        reference_raw, target_raw, boxes, text_features, return_heatmap=True,
    )
    reference = model._compute_core_reference_features(reference_raw, boxes, text_features)
    torch.testing.assert_close(reference["fusion_scalar"], original["fusion_scalar"], rtol=0, atol=0)
    shape = target_raw.shape
    candidates = model._project_raw_patch_tokens(target_raw.reshape(-1, shape[-2], shape[-1]))
    candidates = F.normalize(candidates, dim=-1).reshape(shape[0], shape[1], shape[-2] - 1, -1)
    simple_score, simple_weights = composition_only(
        candidates, reference["composed_feature"], reference["region_sizes"],
        spatial_kernel=model.region_spatial_kernel, spatial_weight=model.region_spatial_weight,
        temperature=model.region_temperature,
    )
    mask_a = original["target_attention"] > 0
    mask_b = simple_weights > 0
    intersection = (mask_a & mask_b).sum(-1).float()
    union = (mask_a | mask_b).sum(-1).clamp_min(1).float()
    diagnostics = {
        "candidate_pairs": mask_a.size(0) * mask_a.size(1),
        "changed_comp_masks": int(mask_a.ne(mask_b).any(-1).sum().item()),
        "comp_mask_iou_sum": float((intersection / union).sum().item()),
    }
    return original["score_id"].cpu(), original["score_comp"].cpu(), simple_score.cpu(), diagnostics


def prepare_ranking(global_scores, indices, reference_indices):
    """Sort the shared backbone tail once for every weight setting."""
    batch, gallery_size = global_scores.shape
    selected = torch.zeros_like(global_scores, dtype=torch.bool).scatter_(1, indices, True)
    backbone_order = global_scores.argsort(1, descending=True)
    tail = backbone_order[~selected.gather(1, backbone_order)].reshape(batch, gallery_size - indices.size(1))
    # At most 50 tail images can enter the final top 50; retain one extra
    # because the reference may occur here. Preserve its original ordering.
    return indices, tail[:, :51], reference_indices, min(50, gallery_size - 1)


def rank_prepared(context, reranked):
    indices, tail, reference_indices, limit = context
    ordered_candidates = indices.gather(1, reranked.argsort(1, descending=True))
    ordered = torch.cat([ordered_candidates, tail], dim=1)
    # The reference may be outside the truncated tail, so avoid reshaping
    # filtered rows to a common length before selecting their first 50.
    positions = torch.arange(ordered.size(1)).expand_as(ordered)
    positions = positions.masked_fill(ordered.eq(reference_indices[:, None]), ordered.size(1))
    keep = positions.topk(limit, dim=1, largest=False, sorted=True).indices
    return ordered.gather(1, keep)


def rank_top50(global_scores, indices, reranked, reference_indices):
    """Match merge_topk_ranking then official exclusion of the reference."""
    return rank_prepared(prepare_ranking(global_scores, indices, reference_indices), reranked)


def fuse_scores(backbone, sid, scomp, fusion_weight, lambda_id, temperature, comp_weight=1.0):
    """Only final score weights vary; region response/masks stay fixed."""
    return backbone + float(fusion_weight) * (
        (float(comp_weight) * scomp + float(lambda_id) * sid) / float(temperature)
    )


def weight_settings(args, model):
    base = float(model.core_matcher_fusion_weight)
    fusions = list(dict.fromkeys([base] + list(args.fusion_weights or [])))
    comps = list(dict.fromkeys([1.0] + list(args.comp_score_weights)))
    identities = list(dict.fromkeys([float(model.core_matcher_lambda_id)] + list(args.lambda_ids or [])))
    temps = list(dict.fromkeys([float(model.core_matcher_temp)] + list(args.matcher_temps or [])))
    settings = []
    for mode in ["A_original", "B_comp_only"]:
        for fw in fusions:
            for cw in comps:
                for lid in identities:
                    for temp in temps:
                        settings.append({
                            "mode": mode, "fusion_weight": fw, "comp_score_weight": cw,
                            "lambda_id": lid, "matcher_temp": temp,
                            "effective_comp_coefficient": fw * cw / temp,
                            "effective_id_coefficient": fw * lid / temp,
                            "key": f"{mode}|fusion={fw:g}|comp={cw:g}|lambda_id={lid:g}|temp={temp:g}",
                        })
    return settings


def recall_metrics(top_indices, target_indices, target_folders, gallery_folders):
    exact = top_indices.eq(target_indices[:, None])
    identity = gallery_folders[top_indices].eq(target_folders[:, None])
    metrics = {}
    for prefix, hits, ks in [("R", exact, [1, 5, 10, 50]), ("R_ID", identity, [1, 3, 5])]:
        for k in ks:
            metrics[f"{prefix}@{k}"] = 100.0 * hits[:, :k].any(1).sum().item() / len(hits)
    metrics["arithmetic_mean"] = sum(metrics[f"R@{k}"] for k in [1, 5, 10, 50]) / 4
    return metrics, exact[:, 0], identity[:, 0]


def load_experiment(args, repo):
    from omegaconf import OmegaConf
    from lavis.common.registry import registry
    from lavis.models import load_preprocess

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    sidecar = checkpoint.parent.parent / "training_hyperparameters.json"
    training = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    name = args.model or training.get("blip_model_name")
    if name not in SUPPORTED:
        raise ValueError(f"Specify --model from {SUPPORTED}; inferred model={name!r}")
    importlib.import_module("lavis.models.blip2_models.blip2_qformer_" + name)
    model_class = registry.get_model_class(name)
    model_type = training.get("vit_backbone", "pretrain")
    config = OmegaConf.load(model_class.default_config_path(model_type))
    model_config = config.model.copy()
    model_config.load_pretrained = False
    model_config.load_finetuned = False
    model = model_class.from_config(model_config)
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    payload = torch.load(checkpoint, **load_kwargs)
    epoch = payload.get("epoch") if isinstance(payload, dict) else None
    def tensor_dict(value):
        return isinstance(value, dict) and bool(value) and all(torch.is_tensor(v) for v in value.values())
    if tensor_dict(payload):
        state = payload
    else:
        states = [v for v in payload.values() if tensor_dict(v)] if isinstance(payload, dict) else []
        if len(states) != 1:
            raise ValueError("Cannot uniquely identify full model state in checkpoint")
        state = states[0]
    model.load_state_dict(state, strict=True)
    del payload, state
    latent_config = {key: training.get(key, value) for key, value in DEFAULTS.items()}
    model.set_latent_config(**latent_config)
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    model = model.to(device).eval()
    _, text_processors = load_preprocess(config.preprocess)
    print(f"Loaded {name}, checkpoint epoch={epoch}, device={device}", flush=True)
    return model, text_processors, training, {
        "checkpoint": str(checkpoint), "checkpoint_epoch": epoch, "model": name,
        "training_config_path": str(sidecar) if sidecar.is_file() else None,
        "latent_config": latent_config, "device": str(device),
        "R_comp_A": "visual_correspondence + cycle_weight * cycle + text_weight * cosine",
        "R_comp_B": "cosine(composed_feature, target_patch); coefficient=1.0",
        "identity_scores": "computed once by original model and reused exactly for A and B",
        "max_queries": args.max_queries,
        "final_score_formula": "S_backbone + fusion_weight * (comp_score_weight * S_comp + lambda_id * S_id) / core_matcher_temp",
        "weight_sweep": weight_settings(args, model),
        "selection_constraint": "R@1 >= A_original_at_checkpoint_weights_R@1 - 0.2 percentage points",
    }


def evaluate_domain(args, model, processors, training, domain):
    from data_utils import OACIRRDataset, targetpad_transform
    from evaluate import extract_index_blip_features_from_raw_cache
    from utils import custom_collate_fn
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    data_root = args.data_root or training.get("data_root", "./Datasets/OACIRR")
    cache_template = args.val_feature_cache or training.get("val_feature_cache") or (
        "./cache/oacirr_{variant_lower}_val_vitg_targetpad125_raw_fp16.pt"
    )
    if training.get("transform", "targetpad") != "targetpad":
        raise ValueError("This script currently supports targetpad checkpoints only")
    preprocess = targetpad_transform(training.get("target_ratio", 1.25), 224)
    relative = OACIRRDataset(data_root, domain, "val", "relative", preprocess,
                             highlight_inference=True, text_entity=training.get("text_entity", False))
    classic = OACIRRDataset(data_root, domain, "val", "classic", preprocess)
    if len(relative) == 0:
        raise ValueError(f"Empty validation dataset: {domain}")
    gallery, raw_bank, names = extract_index_blip_features_from_raw_cache(
        classic, model, cache_template.format(variant=domain, variant_lower=domain.lower()),
        save_memory=True, batch_size=training.get("val_feature_batch_size", 128),
    )
    name_to_index = {name: i for i, name in enumerate(names)}
    folder_keys = {name: path.split("/")[-2] for name, path in relative.name_to_relpath.items()}
    folder_to_index = {key: i for i, key in enumerate(dict.fromkeys(folder_keys.values()))}
    gallery_folders = torch.tensor([folder_to_index[folder_keys[n]] for n in names], dtype=torch.long)
    loader = DataLoader(relative, batch_size=args.batch_size, num_workers=args.num_workers,
                        shuffle=False, collate_fn=custom_collate_fn)
    all_top_a, all_top_b, all_targets, all_folders, details = [], [], [], [], []
    settings = weight_settings(args, model)
    sweep_tops = {setting["key"]: [] for setting in settings}
    diagnostics = {"candidate_pairs": 0, "changed_comp_masks": 0, "comp_mask_iou_sum": 0.0}
    processed = 0
    device = model._get_device()
    for batch in tqdm(loader, desc=f"{domain}: shared candidates + A/B", ncols=120):
        if args.max_queries and processed >= args.max_queries:
            break
        limit = min(len(batch[0]), args.max_queries - processed) if args.max_queries else len(batch[0])
        refs, targets, captions = [list(x[:limit]) for x in batch[:3]]
        boxes = (batch[4] if len(batch) == 5 else batch[3])[:limit]
        texts = [processors["eval"](x) for x in captions]
        ref_indices = torch.tensor([name_to_index[n] for n in refs])
        target_indices = torch.tensor([name_to_index[n] for n in targets])
        ref_raw = raw_bank[ref_indices]
        global_scores = first_stage_scores(
            model, ref_raw, gallery, texts, boxes, training.get("latent_gallery_chunk_size", 2048)
        )
        count = min(int(model.core_matcher_topk), len(names))
        candidate_indices = global_scores.topk(count, dim=1).indices
        candidate_backbone = global_scores.gather(1, candidate_indices)
        id_chunks, a_chunks, b_chunks = [], [], []
        chunk = max(int(model.core_matcher_query_chunk_size), 1)
        text_features = model._compute_text_features(texts, device)
        for start in range(0, limit, chunk):
            end = min(start + chunk, limit)
            ref_chunk = ref_raw[start:end].to(device, non_blocking=True)
            target_chunk = raw_bank[candidate_indices[start:end]].to(device, non_blocking=True)
            sid, scomp_a, scomp_b, diag = compare_chunk(
                model, ref_chunk, target_chunk, boxes[start:end], text_features[start:end]
            )
            id_chunks.append(sid); a_chunks.append(scomp_a); b_chunks.append(scomp_b)
            for key in diagnostics:
                diagnostics[key] += diag[key]
        sid, comp_a, comp_b = [torch.cat(x) for x in [id_chunks, a_chunks, b_chunks]]
        context = prepare_ranking(global_scores, candidate_indices, ref_indices)
        baseline_tops = {}
        for setting in settings:
            score = fuse_scores(
                candidate_backbone, sid, comp_a if setting["mode"] == "A_original" else comp_b,
                setting["fusion_weight"], setting["lambda_id"], setting["matcher_temp"],
                setting["comp_score_weight"],
            )
            top = rank_prepared(context, score)
            sweep_tops[setting["key"]].append(top)
            if (setting["fusion_weight"] == float(model.core_matcher_fusion_weight)
                    and setting["comp_score_weight"] == 1.0
                    and setting["lambda_id"] == float(model.core_matcher_lambda_id)
                    and setting["matcher_temp"] == float(model.core_matcher_temp)):
                baseline_tops[setting["mode"]] = top
        top_a, top_b = baseline_tops["A_original"], baseline_tops["B_comp_only"]
        target_folders = gallery_folders[target_indices]
        all_top_a.append(top_a); all_top_b.append(top_b)
        all_targets.append(target_indices); all_folders.append(target_folders)
        for i in range(limit):
            exact_a, exact_b = top_a[i, 0].item() == target_indices[i].item(), top_b[i, 0].item() == target_indices[i].item()
            id_a = gallery_folders[top_a[i, 0]].item() == target_folders[i].item()
            id_b = gallery_folders[top_b[i, 0]].item() == target_folders[i].item()
            details.append({
                "dataset": domain, "query_index": processed + i,
                "reference": refs[i], "target": targets[i], "text": captions[i],
                "A_top1": names[top_a[i, 0]], "B_top1": names[top_b[i, 0]],
                "A_exact_correct": int(exact_a), "B_exact_correct": int(exact_b),
                "A_identity_correct": int(id_a), "B_identity_correct": int(id_b),
                "exact_change": "improved" if exact_b and not exact_a else "damaged" if exact_a and not exact_b else "unchanged",
                "identity_change": "improved" if id_b and not id_a else "damaged" if id_a and not id_b else "unchanged",
            })
        processed += limit
    targets, folders = torch.cat(all_targets), torch.cat(all_folders)
    metrics_a, _, _ = recall_metrics(torch.cat(all_top_a), targets, folders, gallery_folders)
    metrics_b, _, _ = recall_metrics(torch.cat(all_top_b), targets, folders, gallery_folders)
    delta = {key: metrics_b[key] - metrics_a[key] for key in metrics_a}
    sweeps = []
    for setting in settings:
        metrics, _, _ = recall_metrics(torch.cat(sweep_tops[setting["key"]]), targets, folders, gallery_folders)
        sweeps.append({**setting, "metrics": metrics,
                       "delta_vs_A_original_pp": {k: metrics[k] - metrics_a[k] for k in metrics},
                       "delta_vs_B_current_pp": {k: metrics[k] - metrics_b[k] for k in metrics},
                       "R1_constraint_pass": metrics["R@1"] >= metrics_a["R@1"] - 0.2 - 1e-10})
    feasible = [row for row in sweeps if row["mode"] == "B_comp_only" and row["R1_constraint_pass"]]
    best = max(feasible, key=lambda row: (row["metrics"]["R_ID@1"], row["metrics"]["R@1"],
                                         -row["fusion_weight"], -row["comp_score_weight"],
                                         -abs(row["lambda_id"] - float(model.core_matcher_lambda_id)))) if feasible else None
    counts = {f"{kind}_{change}": sum(row[f"{kind}_change"] == change for row in details)
              for kind in ["exact", "identity"] for change in ["improved", "damaged"]}
    pairs = max(diagnostics["candidate_pairs"], 1)
    diagnostics["changed_comp_mask_fraction"] = diagnostics["changed_comp_masks"] / pairs
    diagnostics["mean_comp_mask_iou"] = diagnostics["comp_mask_iou_sum"] / pairs
    result = {"dataset": domain, "queries": processed, "A_original": metrics_a,
              "B_comp_only": metrics_b, "delta_B_minus_A_pp": delta,
              "top1_changes": counts, "diagnostics": diagnostics,
              "weight_sweep": sweeps, "best_feasible_B_on_validation": best}
    print(f"\n{domain}: R_comp B = cosine, coefficient 1.0")
    print(f"{'metric':<18} {'A original':>12} {'B comp-only':>12} {'B-A (pp)':>12}")
    for key in (*RECALL_KEYS, "arithmetic_mean"):
        print(f"{key:<18} {metrics_a[key]:>12.4f} {metrics_b[key]:>12.4f} {delta[key]:>+12.4f}")
    print(json.dumps({"top1_changes": counts, "diagnostics": diagnostics}, ensure_ascii=False, indent=2))
    if len(sweeps) > 2:
        print("\nFinal-weight sweep (same model scores, regions and candidates):")
        print(f"{'mode':<14} {'fusion':>7} {'comp':>7} {'lambda':>7} {'temp':>7} {'R_ID@1':>10} {'R@1':>10} {'dID vs A':>10} {'dR1 vs A':>10} {'R1>=A-0.2':>11}")
        for row in sweeps:
            m, d = row["metrics"], row["delta_vs_A_original_pp"]
            print(f"{row['mode']:<14} {row['fusion_weight']:>7g} {row['comp_score_weight']:>7g} "
                  f"{row['lambda_id']:>7g} {row['matcher_temp']:>7g} "
                  f"{m['R_ID@1']:>10.4f} {m['R@1']:>10.4f} {d['R_ID@1']:>+10.4f} {d['R@1']:>+10.4f} "
                  f"{str(row['R1_constraint_pass']):>11}")
        print("Best feasible B on this validation set:", best["key"] if best else "none")
    return result, details


def write_outputs(output, meta, results, details):
    report = {"metadata": meta, "domains": results}
    if len(results) > 1:
        report["macro_average"] = {
            group: {key: sum(row[group][key] for row in results) / len(results)
                    for key in results[0][group]}
            for group in ["A_original", "B_comp_only", "delta_B_minus_A_pp"]
        }
        print("Macro average:", json.dumps(report["macro_average"], ensure_ascii=False, indent=2))
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "mode", *RECALL_KEYS, "arithmetic_mean"])
        writer.writeheader()
        for row in results:
            for key in ["A_original", "B_comp_only", "delta_B_minus_A_pp"]:
                writer.writerow({"dataset": row["dataset"], "mode": key, **row[key]})
    with (output / "query_changes.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0]))
        writer.writeheader(); writer.writerows(details)
    with (output / "weight_sweep.csv").open("w", encoding="utf-8-sig", newline="") as f:
        delta_fields = ["delta_vs_A_R_ID@1_pp", "delta_vs_A_R@1_pp", "delta_vs_B_R_ID@1_pp", "delta_vs_B_R@1_pp"]
        config_fields = ["mode", "fusion_weight", "comp_score_weight", "lambda_id", "matcher_temp",
                         "effective_comp_coefficient", "effective_id_coefficient", "R1_constraint_pass"]
        writer = csv.DictWriter(f, fieldnames=["dataset", *config_fields,
                                             *RECALL_KEYS, "arithmetic_mean", *delta_fields])
        writer.writeheader()
        for result in results:
            for row in result["weight_sweep"]:
                writer.writerow({"dataset": result["dataset"], **{k: row[k] for k in config_fields},
                                 **row["metrics"],
                                 **{f"delta_vs_{label}_{metric}_pp": row[group][metric]
                                    for label, group in [("A", "delta_vs_A_original_pp"), ("B", "delta_vs_B_current_pp")]
                                    for metric in ["R_ID@1", "R@1"]}})
    print(f"\nResults saved to: {output}")


def self_test():
    """Test actual formula, independent identity path, and official rank merging."""
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    from torch import nn
    from lavis.models.blip2_models.oacir_latent_matching import compute_bidirectional_region_scores
    from lavis.models.blip2_models.blip2_qformer_oacir_latent_split_mask import Blip2QformerOacirLatentSplitMask

    patches = torch.randn(2, 3, 16, 8)
    composed = torch.randn(2, 8)
    sizes = torch.tensor([4, 7])
    for kernel in [1, 3]:
        score, weights = composition_only(patches, composed, sizes, kernel, .15, .07)
        oracle = compute_bidirectional_region_scores(
            torch.zeros(2, 1, 8), patches, composed, sizes,
            spatial_kernel_size=kernel, spatial_weight=.15, region_temperature=.07,
            cycle_weight=0., text_weight=1., return_heatmap=True,
        )
        torch.testing.assert_close(score, oracle["score_comp"], rtol=0, atol=0)
        torch.testing.assert_close(weights, oracle["target_attention"], rtol=0, atol=0)
    quarter = compute_bidirectional_region_scores(
        torch.zeros(2, 1, 8), patches, composed, sizes,
        cycle_weight=0., text_weight=.25, return_heatmap=True,
    )
    assert not torch.allclose(weights, quarter["target_attention"])
    print("PASS B response is cosine * 1.0, not cosine * 0.25; smoothing/selection match original")

    class Tiny(Blip2QformerOacirLatentSplitMask):
        def __init__(self):
            nn.Module.__init__(self)
            self.ln_vision = nn.LayerNorm(8)
            self.temp = nn.Parameter(torch.tensor(.07))
            self.region_spatial_kernel = 3
            self.region_spatial_weight = .15
            self.region_temperature = .07
            self.core_matcher_cycle_weight = .25
            self.core_matcher_text_weight = .25
            self.core_matcher_lambda_id = 1.25
            self.core_matcher_fusion_weight = .4
            self.core_matcher_query_chunk_size = 2
            self.core_matcher_temp = .07
        def _project_raw_patch_tokens(self, raw):
            return raw[:, 1:]
        def _compute_fusion_features(self, ref, text, boxes):
            return F.normalize(ref[:, 1:].mean(1) + text, dim=-1)
        def _compute_core_reference_features(self, raw, boxes, text):
            return {"region_tokens": F.normalize(raw[:, 1:5], dim=-1),
                    "composed_feature": self._compute_fusion_features(raw, text, boxes),
                    "region_sizes": torch.tensor([4] * len(raw)),
                    "reference_attention": torch.ones(len(raw), 4, 16) / 16,
                    "fusion_scalar": torch.ones(len(raw), 1) * .5}
    model = Tiny().eval()
    ref, target, text = torch.randn(3, 17, 8), torch.randn(3, 4, 17, 8), torch.randn(3, 8)
    boxes = [[0, 0, 224, 224]] * 3
    gallery = F.normalize(torch.randn(9, 4, 8), dim=-1)
    with torch.no_grad():
        shared_stage = first_stage_scores(model, ref, gallery, text, boxes, 4)
        expected_stage = model.inference_composition_from_raw(ref, gallery, text, boxes)
        torch.testing.assert_close(shared_stage, expected_stage)
        sid, a, b, diag = compare_chunk(model, ref, target, boxes, text)
        baseline = model._compute_core_matcher_candidate_scores(ref, target, boxes, text)
        torch.testing.assert_close(sid, baseline["score_id"], rtol=0, atol=0)
        torch.testing.assert_close(a, baseline["score_comp"], rtol=0, atol=0)
        assert not torch.allclose(a, b)
        backbone = torch.randn_like(a)
        fused = fuse_scores(backbone, sid, a, .4, 1.25, .07)
        torch.testing.assert_close(fused, backbone + .4 * (baseline["score_final"] / .07), rtol=0, atol=0)
        torch.testing.assert_close(fuse_scores(backbone, sid, a, .8, 1.25, .14), fused, rtol=0, atol=0)
        torch.testing.assert_close(fuse_scores(backbone, sid, a, 0., 2., .07), backbone, rtol=0, atol=0)
        assert not torch.allclose(fuse_scores(backbone, sid, a, .6, 1.25, .07), fused)
        assert not torch.allclose(fuse_scores(backbone, sid, a, .4, 2., .07), fused)
    print("PASS first stage equals original; A equals original; identity reused without modification")
    from types import SimpleNamespace
    settings = weight_settings(SimpleNamespace(fusion_weights=[.2, .3, .4, .5, .6, .8],
                                              comp_score_weights=[1.], lambda_ids=[.75, 1., 1.25, 1.5, 2.],
                                              matcher_temps=None), model)
    assert len(settings) == 60
    assert len({row["key"] for row in settings}) == 60
    assert all(row["matcher_temp"] == .07 for row in settings)
    print("PASS fusion_weight x lambda_id grid: 30 settings per mode; baseline and temperature equivalence verified")

    global_scores = torch.randn(4, 90)
    indices = global_scores.topk(50, dim=1).indices
    # Cover reference inside and outside candidate set.
    references = torch.tensor([indices[0, 0], indices[1, -1], global_scores[2].argmin(), 3])
    scores = torch.randn(4, 50)
    ours = rank_top50(global_scores, indices, scores, references)
    merged = model.merge_topk_ranking(global_scores, indices, scores)
    merged[torch.arange(4), references] = -torch.inf
    official = merged.topk(50, dim=1).indices
    assert torch.equal(ours, official)
    for row in range(4):
        assert references[row] not in ours[row]
    print("PASS final ranking exactly matches original merge and reference exclusion")
    for gallery_size in [9, 90, 300]:
        for count in sorted({1, min(7, gallery_size), min(50, gallery_size), gallery_size}):
            backbone = torch.randn(4, gallery_size)
            candidates = backbone.topk(count, dim=1).indices
            refs = torch.tensor([candidates[0, 0], candidates[1, -1], backbone[2].argmin(), 3])
            context = prepare_ranking(backbone, candidates, refs)
            for _ in range(3):
                reranked = torch.randn(4, count)
                ours = rank_prepared(context, reranked)
                official = model.merge_topk_ranking(backbone, candidates, reranked)
                official[torch.arange(4), refs] = -torch.inf
                official = official.topk(min(50, gallery_size - 1), dim=1).indices
                assert torch.equal(ours, official)
    print("PASS shared ranking context matches original for short/full candidate sets and truncated tails")
    print("ALL_RCOMP_AB_CHECKS_PASS")


def main():
    args = parse_args()
    repo = find_repo(args.repo_root)
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    if not args.self_test and (checkpoint is None or not checkpoint.is_file()):
        raise FileNotFoundError("Pass --checkpoint /actual/path/to/checkpoint.pt")
    if args.batch_size < 1 or args.num_workers < 0 or args.max_queries < 0:
        raise ValueError("Invalid batch size, worker count, or query limit")
    for value in list(args.fusion_weights or []) + list(args.comp_score_weights) + list(args.lambda_ids or []):
        if not math.isfinite(value) or value < 0:
            raise ValueError("Score weights must be finite and nonnegative")
    if any(not math.isfinite(value) or value <= 0 for value in (args.matcher_temps or [])):
        raise ValueError("Final denominators must be finite and strictly positive")
    configure_gpu(args)
    os.environ.setdefault("TMPDIR", "/tmp")
    os.chdir(repo)
    init_runtime(repo)
    if args.self_test:
        self_test()
        return
    args.checkpoint = str(checkpoint)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; activate the OACIR environment or explicitly use --device cpu")
    model, processors, training, meta = load_experiment(args, repo)
    output = Path(args.output_dir or (
        "./checkpoints/rcomp_ab/" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )).resolve()
    output.mkdir(parents=True, exist_ok=False)
    meta["repo_root"] = str(repo)
    meta["script_sha256"] = __import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest()
    results, details = [], []
    with torch.no_grad():
        for domain in dict.fromkeys(args.dataset):
            result, domain_details = evaluate_domain(args, model, processors, training, domain)
            results.append(result); details.extend(domain_details)
            write_outputs(output, meta, results, details)


if __name__ == "__main__":
    main()
