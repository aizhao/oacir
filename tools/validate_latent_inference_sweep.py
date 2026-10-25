#!/usr/bin/env python3
"""OACIR: single-file, single-GPU inference-only staged parameter search.

Place this file in tools/ of the existing OACIR repository and run:
  python tools/validate_latent_inference_sweep.py
No dependency on validate_rcomp_ab.py, no source patches, no training, no Excel.
Search is staged, NOT an exhaustive joint grid over all region parameters.
Ground-truth image/identity labels are used only to evaluate/select trials,
never to construct candidates, score features, or make gate decisions.
"""

import argparse
import csv
import hashlib
import importlib
import inspect
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path, PurePosixPath


DEFAULT_CHECKPOINT = (
    "./checkpoints/latent_split_mask_from_pretrain/"
    "OACIR_Union_finetune_oacir_latent_split_mask_2026-10-23_02:40:31/"
    "saved_models/adafocal_finetune_best_identity.pt"
)
SUPPORTED = ("oacir_latent_split_mask", "oacir_latent_identity_split", "oacir_latent_margin_split")
METRICS = ("R_ID@1", "R_ID@3", "R_ID@5", "R@1", "R@5", "R@10", "R@50", "arithmetic_mean")
DEFAULTS = {
    "core_matcher_temp": .07, "core_matcher_topk": 50,
    "core_matcher_lambda_id": 1.25, "core_matcher_fusion_weight": .4,
    "core_matcher_cycle_weight": .25, "core_matcher_text_weight": .25,
    "core_matcher_bbox_floor": .05, "core_matcher_query_chunk_size": 2,
    "region_topk": 16, "region_topk_max": 96, "region_area_scale": 1.5,
    "region_spatial_kernel": 3, "region_spatial_weight": .15, "region_temperature": .07,
}
# One-parameter-at-a-time region trials around B, plus one combined trial.
REGION_GRID = {
    "region_temperature": [.03, .05, .07, .10],
    "region_area_scale": [.75, 1., 1.5, 2.],
    "region_spatial_weight": [0., .05, .15, .30],
    "region_spatial_kernel": [1, 3, 5],
    "core_matcher_cycle_weight": [0., .1, .25, .5],
    "core_matcher_topk": [20, 50, 100],
    "core_matcher_bbox_floor": [.01, .05, .10],
    "region_topk": [8, 16, 32],
    "region_topk_max": [64, 96, 128],
}
FUSIONS = [.2, .3, .4, .5, .6, .8]
IDENTITIES = [.75, 1., 1.25, 1.5, 2.]
GATE_THRESHOLDS = [-.05, -.025, 0., .025, .05, .10]


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--repo-root")
    p.add_argument("--model", choices=SUPPORTED)
    p.add_argument("--dataset", "--datasets", nargs="+", default=["Car"],
                   choices=["Fashion", "Car", "Landmark", "Product"])
    p.add_argument("--data-root")
    p.add_argument("--val-feature-cache")
    p.add_argument("--output-dir")
    p.add_argument("--gpu", type=int, help="One physical GPU; default selects a relatively idle GPU")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--query-chunk-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--max-queries", type=int, default=0, help="Smoke test only: first N queries; 0=all")
    p.add_argument("--max-r1-drop", type=float, default=.2, help="Percentage points relative to A baseline")
    p.add_argument("--self-test", action="store_true", help="CPU synthetic/original-code tests; no model weights")
    p.add_argument("--show-plan", action="store_true", help="Print search plan without loading torch/model")
    return p.parse_args()


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def find_repo(explicit):
    choices = [Path(explicit)] if explicit else [Path.cwd(), Path(__file__).resolve().parent,
                                                Path(__file__).resolve().parent.parent]
    for path in choices:
        if (path / "data_utils.py").is_file() and (path / "lavis").is_dir():
            return path.resolve()
    raise FileNotFoundError("Put this file in OACIR/tools/ and run from the repository, or pass --repo-root")


def runtime(args, repo):
    # Set this before importing torch/multiprocessing to avoid long UNIX socket paths.
    os.environ["TMPDIR"] = "/tmp"
    tempfile.tempdir = "/tmp"
    if args.self_test or args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        visible = str(args.gpu) if args.gpu is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is None:
            result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
            eligible = []
            for line in result.stdout.splitlines():
                index, used, free, utilization = [int(x.strip()) for x in line.split(",")]
                if used <= 12000 and free >= 10000 and utilization <= 70:
                    eligible.append((-free, utilization, index, used))
            if not eligible:
                raise RuntimeError("No relatively idle GPU with >=10000 MiB free; retry later or pass --gpu INDEX")
            neg_free, utilization, index, used = min(eligible)
            visible = str(index)
            print(f"Selected GPU {index}: used={used} MiB, free={-neg_free} MiB, utilization={utilization}%", flush=True)
        if not visible.strip() or "," in visible:
            raise ValueError("Exactly one GPU is allowed; use --gpu INDEX")
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    global torch, F
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(args.cpu_threads)
    random.seed(2026)
    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)
    if not args.self_test and args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this Python environment")


def load_torch(path):
    kw = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kw["weights_only"] = False
    return torch.load(path, **kw)


def load_model(args):
    from omegaconf import OmegaConf
    from lavis.common.registry import registry
    from lavis.models import load_preprocess
    checkpoint = Path(args.checkpoint)
    sidecar = checkpoint.parent.parent / "training_hyperparameters.json"
    training = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    name = args.model or training.get("blip_model_name")
    if name not in SUPPORTED:
        raise ValueError(f"Cannot infer a supported dual-mask model; specify --model from {SUPPORTED}")
    importlib.import_module("lavis.models.blip2_models.blip2_qformer_" + name)
    cls = registry.get_model_class(name)
    config = OmegaConf.load(cls.default_config_path(training.get("vit_backbone", "pretrain")))
    mc = config.model.copy()
    mc.load_pretrained = False
    mc.load_finetuned = False
    model = cls.from_config(mc)
    payload = load_torch(checkpoint)
    is_state = lambda x: isinstance(x, dict) and bool(x) and all(torch.is_tensor(v) for v in x.values())
    states = [payload] if is_state(payload) else [v for v in payload.values() if is_state(v)]
    if len(states) != 1:
        raise ValueError("Checkpoint must contain exactly one full model state dictionary")
    model.load_state_dict(states[0], strict=True)
    epoch = payload.get("epoch")
    del payload, states
    params = {key: training.get(key, value) for key, value in DEFAULTS.items()}
    model.set_latent_config(**params)
    model = model.to("cuda:0" if args.device == "cuda" else "cpu").eval()
    _, processors = load_preprocess(config.preprocess)
    meta = {"checkpoint": str(checkpoint), "checkpoint_epoch": epoch, "model": name,
            "training_config_path": str(sidecar) if sidecar.is_file() else None,
            "checkpoint_inference_parameters": params}
    print(f"Loaded {name}, epoch={epoch}; model.eval(), gradients disabled", flush=True)
    return model, processors, training, params, meta


def config_key(config):
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def region_plan(base):
    configs = [{"mode": "A_original", "params": dict(base), "axis": "baseline_A"},
               {"mode": "B_comp_only", "params": dict(base), "axis": "baseline_B"}]
    for name, values in REGION_GRID.items():
        for value in values:
            if value == base[name]:
                continue
            params = {**base, name: value}
            if params["region_topk"] > params["region_topk_max"]:
                continue
            configs.append({"mode": "B_comp_only", "params": params, "axis": name})
    for index, config in enumerate(configs):
        config["id"] = f"region_{index:02d}"
    return configs


def parameter_plan(base):
    return {"strategy": "A/B controls + B one-factor-at-a-time region trials + tested combined trial + final weights + identity-gain gates",
            "not_exhaustive_joint_search": True, "region_trial_count_including_controls": len(region_plan(base)),
            "region_grid": REGION_GRID, "fusion_weights": FUSIONS, "lambda_ids": IDENTITIES,
            "final_temperature": base["core_matcher_temp"], "comp_score_coefficient": 1.,
            "gate_thresholds_in_raw_identity_cosine_score_units": GATE_THRESHOLDS,
            "B_region_response": "cosine(composed_feature, patch), coefficient exactly 1.0",
            "learned_weights_changed": False}


def first_stage(model, raw, gallery, captions, boxes, chunk):
    device = model._get_device()
    normalized = model.ln_vision(raw.to(device=device, dtype=model.ln_vision.weight.dtype))
    query = model._compute_fusion_features(normalized, captions, boxes)
    parts = []
    for start in range(0, len(gallery), chunk):
        features = gallery[start:start + chunk].to(device)
        parts.append((model._compute_composition_scores(query, features) / model.temp).cpu())
    return torch.cat(parts, 1)


def identity_key(path):
    path = PurePosixPath(str(path).replace("\\", "/"))
    if len(path.parts) < 2 or ".." in path.parts:
        raise ValueError(f"Invalid annotated identity path: {path}")
    return path.parent.as_posix()


def union_candidates(indices_per_k):
    """Preserve exact torch.topk(K) sets/order for every K, including ties."""
    largest = max(indices_per_k)
    rows, maps = [], {k: [] for k in indices_per_k}
    for i in range(len(indices_per_k[largest])):
        values = indices_per_k[largest][i].tolist()
        lookup = {value: pos for pos, value in enumerate(values)}
        for k, indices in indices_per_k.items():
            for value in indices[i].tolist():
                if value not in lookup:
                    lookup[value] = len(values)
                    values.append(value)
            maps[k].append(torch.tensor([lookup[v] for v in indices[i].tolist()]))
        rows.append(torch.tensor(values, dtype=torch.long))
    return rows, {k: torch.stack(v) for k, v in maps.items()}


def prepare_data(args, model, processors, training, domain, base):
    from data_utils import OACIRRDataset, targetpad_transform
    from evaluate import extract_index_blip_features_from_raw_cache
    from utils import custom_collate_fn
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    if training.get("transform", "targetpad") != "targetpad":
        raise ValueError("Only targetpad checkpoints are supported; do not silently mix transforms")
    preprocess = targetpad_transform(training.get("target_ratio", 1.25), 224)
    data_root = args.data_root or training.get("data_root", "./Datasets/OACIRR")
    relative = OACIRRDataset(data_root, domain, "val", "relative", preprocess,
                             highlight_inference=True, text_entity=training.get("text_entity", False))
    classic = OACIRRDataset(data_root, domain, "val", "classic", preprocess)
    cache = args.val_feature_cache or training.get("val_feature_cache") or (
        "./cache/oacirr_{variant_lower}_val_vitg_targetpad125_raw_fp16.pt")
    cache = cache.format(variant=domain, variant_lower=domain.lower())
    if len(relative) == 0:
        raise ValueError(f"Empty validation domain: {domain}")
    gallery, raw_bank, names = extract_index_blip_features_from_raw_cache(
        classic, model, cache, save_memory=True, batch_size=training.get("val_feature_batch_size", 128))
    lookup = {name: index for index, name in enumerate(names)}
    if len(lookup) != len(names) or len(names) < 2:
        raise ValueError("Gallery names must be unique and contain at least two images")
    keys = [identity_key(relative.name_to_relpath[name]) for name in names]
    key_ids = {key: i for i, key in enumerate(dict.fromkeys(keys))}
    folder_ids = torch.tensor([key_ids[key] for key in keys], dtype=torch.long)
    ks = sorted({min(int(k), len(names)) for k in [base["core_matcher_topk"], *REGION_GRID["core_matcher_topk"]]})
    contexts = {k: {"indices": [], "positions": [], "tail": []} for k in ks}
    data = {"names": names, "gallery_folders": folder_ids, "gallery_size": len(names), "contexts": contexts,
            "references": [], "targets": [], "captions": [], "boxes": [], "reference_indices": [],
            "target_indices": [], "bank_rows": [], "backbone_rows": [], "text_features": [],
            "stage1_top50": [], "domain": domain, "raw_cache_path": str(Path(cache).resolve())}
    loader = DataLoader(relative, batch_size=args.batch_size, num_workers=args.num_workers,
                        shuffle=False, collate_fn=custom_collate_fn)
    seen = 0
    for batch in tqdm(loader, desc=f"{domain}: stage1 ONCE", ncols=110):
        if args.max_queries and seen >= args.max_queries:
            break
        size = min(len(batch[0]), args.max_queries - seen) if args.max_queries else len(batch[0])
        refs, targets, captions = [list(v[:size]) for v in batch[:3]]
        boxes = (batch[4] if len(batch) == 5 else batch[3])[:size]
        boxes = [v.detach().cpu().tolist() if torch.is_tensor(v) else v for v in boxes]
        texts = [processors["eval"](caption) for caption in captions]
        ref_indices = torch.tensor([lookup[name] for name in refs])
        target_indices = torch.tensor([lookup[name] for name in targets])
        scores = first_stage(model, raw_bank[ref_indices], gallery, texts, boxes,
                             training.get("latent_gallery_chunk_size", 2048))
        order = scores.argsort(dim=1, descending=True)
        valid_order = order[~order.eq(ref_indices[:, None])].reshape(size, -1)
        data["stage1_top50"].append(valid_order[:, :50])
        indices = {k: scores.topk(k, dim=1).indices for k in ks}
        banks, maps = union_candidates(indices)
        for i, bank in enumerate(banks):
            data["bank_rows"].append(bank)
            data["backbone_rows"].append(scores[i, bank])
        for k, selected in indices.items():
            is_selected = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, selected, True)
            tail = order[~is_selected.gather(1, order)].reshape(size, len(names) - k)[:, :51]
            contexts[k]["indices"].append(selected)
            contexts[k]["positions"].append(maps[k])
            contexts[k]["tail"].append(tail)
        data["references"].extend(refs); data["targets"].extend(targets)
        data["captions"].extend(captions); data["boxes"].extend(boxes)
        data["reference_indices"].append(ref_indices); data["target_indices"].append(target_indices)
        data["text_features"].append(model._compute_text_features(texts, model._get_device()).cpu())
        seen += size
    for key in ["reference_indices", "target_indices", "text_features", "stage1_top50"]:
        data[key] = torch.cat(data[key])
    for context in contexts.values():
        for key in context:
            context[key] = torch.cat(context[key])
    width = max(row.numel() for row in data["bank_rows"])
    data["bank_indices"] = torch.stack([F.pad(row, (0, width - len(row)), value=int(row[0])) for row in data.pop("bank_rows")])
    data["bank_backbone"] = torch.stack([F.pad(row, (0, width - len(row)), value=float(row[0])) for row in data.pop("backbone_rows")])
    data["queries"] = seen
    del gallery, loader, classic, relative
    print(f"{domain}: queries={seen}, exact Top-K sets={ks}, union bank width={width}", flush=True)
    return data, raw_bank


def primitives(regions, candidates, composed):
    """Cache expensive patch correspondences; region parameters reuse them."""
    regions = F.normalize(regions, dim=-1)
    candidates = F.normalize(candidates, dim=-1)
    composed = F.normalize(composed, dim=-1)
    similarities = torch.einsum("brd,bcnd->bcrn", regions, candidates)
    forward_indices = similarities.max(dim=-1).indices
    reverse, reverse_indices = similarities.max(dim=-2)
    roundtrip = forward_indices.gather(2, reverse_indices)
    cycle = roundtrip.eq(torch.arange(candidates.size(2), device=candidates.device)[None, None])
    composition = torch.einsum("bd,bcnd->bcn", composed, candidates)
    count = cycle.sum(-1).clamp_min(1).to(reverse.dtype)
    cycle_similarity = (reverse * cycle.to(reverse.dtype)).sum(-1) / count
    return {"similarities": similarities, "reverse": reverse, "cycle": cycle,
            "cycle_similarity": cycle_similarity, "composition": composition,
            "patches": candidates, "composed": composed}


def region_weights(response, sizes, params):
    batch, candidates, count = response.shape
    grid = int(math.sqrt(count))
    has_grid = grid * grid == count
    kernel = int(params["region_spatial_kernel"])
    if kernel < 1 or kernel % 2 == 0 or params["region_temperature"] <= 0:
        raise ValueError("Require a positive odd spatial kernel and positive region temperature")
    if has_grid and kernel > 1:
        response = F.avg_pool2d(F.pad(response.reshape(-1, 1, grid, grid), (kernel // 2,) * 4,
                                    mode="replicate"), kernel, stride=1).reshape(batch, candidates, count)
    selected = torch.zeros_like(response, dtype=torch.bool)
    if has_grid:
        yy, xx = torch.meshgrid(torch.arange(grid, device=response.device),
                                torch.arange(grid, device=response.device), indexing="ij")
        coords = torch.stack([yy.flatten(), xx.flatten()], -1).to(response.dtype)
    for i in range(batch):
        size = min(max(int(sizes[i]), 1), count)
        rank = response[i]
        if has_grid and size < count:
            peak = coords[rank.argmax(-1)]
            distance = (coords[None] - peak[:, None]).pow(2).sum(-1) / max(float((grid - 1) ** 2), 1.)
            rank = rank - float(params["region_spatial_weight"]) * distance
        selected[i].scatter_(1, rank.topk(size, dim=-1).indices, True)
    weights = F.softmax((response / float(params["region_temperature"])).masked_fill(
        ~selected, torch.finfo(response.dtype).min), dim=-1)
    return selected, weights


def region_sizes(area, patch_count, params):
    minimum = min(max(int(params["region_topk"]), 1), patch_count)
    maximum = min(max(int(params["region_topk_max"]), minimum), patch_count)
    return (area.float() * float(params["region_area_scale"])).round().long().clamp(min=minimum, max=maximum)


def score_regions(primitive, area, config):
    p = config["params"]
    sizes = region_sizes(area, primitive["patches"].size(2), p)
    cw = float(p["core_matcher_cycle_weight"])
    # M_id is always text-free. Changes to shared region parameters affect
    # BOTH routes; only B's response formula is held at pure cosine.
    identity_response = primitive["reverse"] + cw * primitive["cycle"].to(primitive["reverse"].dtype)
    selected, weights = region_weights(identity_response, sizes, p)
    minimum = torch.finfo(primitive["similarities"].dtype).min
    coverage = primitive["similarities"].masked_fill(~selected.unsqueeze(2), minimum).max(-1).values.mean(-1)
    purity = (weights * primitive["reverse"]).sum(-1)
    sid = (coverage + purity + cw * primitive["cycle_similarity"]) / (2. + cw)
    if config["mode"] == "A_original":
        # Preserve the original expression's operation order.
        response = (primitive["reverse"] + float(p["core_matcher_text_weight"]) * primitive["composition"]
                    + cw * primitive["cycle"].to(primitive["reverse"].dtype))
    else:
        response = primitive["composition"]  # coefficient exactly 1.0, NOT 0.25
    _, comp_weights = region_weights(response, sizes, p)
    pooled = F.normalize(torch.einsum("bcn,bcnd->bcd", comp_weights, primitive["patches"]), dim=-1)
    scomp = torch.einsum("bd,bcd->bc", primitive["composed"], pooled)
    return sid, scomp, sizes


def compute_components(args, model, data, raw_bank, configs, audit=False):
    from tqdm import tqdm
    count, width = data["bank_indices"].shape
    components = {cfg["id"]: {"sid": [], "scomp": [], "sizes": []} for cfg in configs}
    device = model._get_device()
    original_floor = model.core_matcher_bbox_floor
    try:
        for start in tqdm(range(0, count, args.query_chunk_size), desc=f"{data['domain']}: {len(configs)} region configs", ncols=110):
            end = min(start + args.query_chunk_size, count)
            boxes = data["boxes"][start:end]
            reference_raw = raw_bank[data["reference_indices"][start:end]].to(device)
            text = data["text_features"][start:end].to(device)
            raw = raw_bank[data["bank_indices"][start:end]].to(device)
            shape = raw.shape
            patches = F.normalize(model._project_raw_patch_tokens(raw.reshape(-1, shape[-2], shape[-1])), dim=-1)
            patches = patches.reshape(end - start, width, shape[-2] - 1, -1)
            del raw
            area = model._build_anchor_mask(boxes, reference_raw.size(1), end - start, device).sum(-1)
            by_floor = {}
            for cfg in configs:
                by_floor.setdefault(float(cfg["params"]["core_matcher_bbox_floor"]), []).append(cfg)
            for floor, current in by_floor.items():
                model.core_matcher_bbox_floor = floor
                ref = model._compute_core_reference_features(reference_raw, boxes, text)
                primitive = primitives(ref["region_tokens"], patches, ref["composed_feature"])
                for cfg in current:
                    sid, scomp, sizes = score_regions(primitive, area, cfg)
                    components[cfg["id"]]["sid"].append(sid.cpu())
                    components[cfg["id"]]["scomp"].append(scomp.cpu())
                    components[cfg["id"]]["sizes"].append(sizes.cpu())
                    if audit and start == 0 and cfg["axis"] == "baseline_A":
                        k = min(int(cfg["params"]["core_matcher_topk"]), data["gallery_size"])
                        context = data["contexts"][k]
                        target = raw_bank[context["indices"][start:end]].to(device)
                        official = model._compute_core_matcher_candidate_scores(reference_raw, target, boxes, text)
                        pos = context["positions"][start:end]
                        torch.testing.assert_close(sid.cpu().gather(1, pos), official["score_id"].cpu(), rtol=1e-5, atol=1e-6)
                        torch.testing.assert_close(scomp.cpu().gather(1, pos), official["score_comp"].cpu(), rtol=1e-5, atol=1e-6)
                        print("PASS actual-checkpoint first-chunk A scores match original model", flush=True)
                del primitive, ref
            del patches, reference_raw, text
    finally:
        model.core_matcher_bbox_floor = original_floor
    for entry in components.values():
        for key in entry:
            entry[key] = torch.cat(entry[key])
    return components


def rank_context(context, scores, references, gallery_size):
    ordered_candidates = context["indices"].gather(1, scores.argsort(dim=1, descending=True))
    ordered = torch.cat([ordered_candidates, context["tail"]], 1)
    positions = torch.arange(ordered.size(1)).expand_as(ordered)
    positions = positions.masked_fill(ordered.eq(references[:, None]), ordered.size(1))
    keep = positions.topk(min(50, gallery_size - 1), dim=1, largest=False, sorted=True).indices
    return ordered.gather(1, keep)


def final_scores(backbone, sid, scomp, params):
    return backbone + float(params["core_matcher_fusion_weight"]) * (
        (scomp + float(params["core_matcher_lambda_id"]) * sid) / float(params["core_matcher_temp"]))


def trial_top(data, component, config, gate=None):
    p = config["params"]
    k = min(int(p["core_matcher_topk"]), data["gallery_size"])
    context = data["contexts"][k]
    pos = context["positions"]
    backbone = data["bank_backbone"].gather(1, pos)
    sid = component["sid"].gather(1, pos)
    scomp = component["scomp"].gather(1, pos)
    scores = final_scores(backbone, sid, scomp, p)
    blocked, changed, fallback = 0, 0, 0
    if gate is not None:
        valid = ~context["indices"].eq(data["reference_indices"][:, None])
        # Use the actual full-gallery stage1 Top-1, including its tie order.
        baseline_pos = context["indices"].eq(data["stage1_top50"][:, :1]).long().argmax(1)
        proposed_pos = scores.masked_fill(~valid, -torch.inf).argmax(1)
        gain = sid.gather(1, proposed_pos[:, None]).squeeze(1) - sid.gather(1, baseline_pos[:, None]).squeeze(1)
        moved = proposed_pos.ne(baseline_pos) & valid.any(1)
        if gate["kind"] == "stage1_only":
            accepted = torch.zeros_like(moved)
        else:
            accepted = ~moved | gain.ge(float(gate["threshold"]))
        blocked = int((moved & ~accepted).sum())
        changed = int(moved.sum())
        fallback = int((~accepted).sum())
        scores = torch.where(accepted[:, None], scores, backbone)
    top = rank_context(context, scores, data["reference_indices"], data["gallery_size"])
    if gate is not None:
        top = torch.where(accepted[:, None], top, data["stage1_top50"])
    return top, {"changed_top1_proposals": changed, "blocked_top1_proposals": blocked, "fallback_queries": fallback}


def metrics_for(data, top):
    target = data["target_indices"]
    exact = top.eq(target[:, None])
    folders = data["gallery_folders"]
    identity = folders[top].eq(folders[target][:, None])
    result = {}
    for label, hits, ks in [("R", exact, [1, 5, 10, 50]), ("R_ID", identity, [1, 3, 5])]:
        for k in ks:
            result[f"{label}@{k}"] = 100. * hits[:, :k].any(1).sum().item() / len(top)
    result["arithmetic_mean"] = sum(result[f"R@{k}"] for k in [1, 5, 10, 50]) / 4.
    return result


def summarize_sizes(sizes, patch_count, p):
    minimum = min(max(int(p["region_topk"]), 1), patch_count)
    maximum = min(max(int(p["region_topk_max"]), minimum), patch_count)
    return {"min": int(sizes.min()), "max": int(sizes.max()), "mean": float(sizes.float().mean()),
            "fraction_at_lower_bound": float(sizes.eq(minimum).float().mean()),
            "fraction_at_upper_bound": float(sizes.eq(maximum).float().mean())}


def evaluate_trial(data, component, config, baseline_a, baseline_b, args, stage, gate=None):
    top, gate_info = trial_top(data, component, config, gate)
    metrics = metrics_for(data, top)
    return {"trial_id": config["id"], "stage": stage, "axis": config["axis"], "mode": config["mode"],
            "params": config["params"], "gate": gate, "metrics": metrics,
            "delta_vs_A_pp": {key: metrics[key] - baseline_a[key] for key in metrics},
            "delta_vs_B_pp": {key: metrics[key] - baseline_b[key] for key in metrics},
            "R1_constraint_pass": metrics["R@1"] >= baseline_a["R@1"] - args.max_r1_drop - 1e-10,
            "gate_diagnostics": gate_info,
            "region_size_statistics": summarize_sizes(component["sizes"], data["patch_count"], config["params"])}, top


def best_trial(rows, feasible=True):
    valid = [row for row in rows if row["mode"] == "B_comp_only" and (row["R1_constraint_pass"] or not feasible)]
    return max(valid, key=lambda row: (row["metrics"]["R_ID@1"], row["metrics"]["R@1"])) if valid else None


def print_trial(row):
    print(f"{row['trial_id']:<28} {row['mode']:<12} RID@1={row['metrics']['R_ID@1']:.4f} "
          f"R@1={row['metrics']['R@1']:.4f} dID={row['delta_vs_A_pp']['R_ID@1']:+.4f} "
          f"dR1={row['delta_vs_A_pp']['R@1']:+.4f} pass={row['R1_constraint_pass']}", flush=True)


def write_report(output, meta, domain_reports):
    report = {"metadata": meta, "domains": domain_reports}
    dump_json(output / "report.json", report)
    fields = ["dataset", "trial_id", "stage", "axis", "mode", "R1_constraint_pass", *METRICS,
              "delta_R_ID@1_vs_A_pp", "delta_R@1_vs_A_pp", *DEFAULTS, "gate", "region_size_statistics"]
    with (output / "trials.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for domain in domain_reports:
            for row in domain["trials"]:
                writer.writerow({"dataset": domain["dataset"],
                                 **{k: row[k] for k in ["trial_id", "stage", "axis", "mode", "R1_constraint_pass"]},
                                 **row["metrics"], "delta_R_ID@1_vs_A_pp": row["delta_vs_A_pp"]["R_ID@1"],
                                 "delta_R@1_vs_A_pp": row["delta_vs_A_pp"]["R@1"], **row["params"],
                                 "gate": json.dumps(row["gate"]),
                                 "region_size_statistics": json.dumps(row["region_size_statistics"])})
    dump_json(output / "best.json", {domain["dataset"]: domain.get("best_feasible_B") for domain in domain_reports})


def save_query_changes(output, data, top_a, top_b, top_best):
    fields = ["query_index", "reference", "target", "text", "A_top1", "B_top1", "best_top1",
              "A_exact_correct", "B_exact_correct", "best_exact_correct",
              "A_identity_correct", "B_identity_correct", "best_identity_correct"]
    with (output / f"{data['domain'].lower()}_query_changes.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(data["queries"]):
            target = data["target_indices"][i]
            folder = data["gallery_folders"][target]
            row = {"query_index": i, "reference": data["references"][i], "target": data["targets"][i], "text": data["captions"][i]}
            for name, top in [("A", top_a), ("B", top_b), ("best", top_best)]:
                index = top[i, 0]
                row[f"{name}_top1"] = data["names"][index]
                row[f"{name}_exact_correct"] = int(index == target)
                row[f"{name}_identity_correct"] = int(data["gallery_folders"][index] == folder)
            writer.writerow(row)


def run_domain(args, model, processors, training, base, domain, output, meta, previous):
    configs = region_plan(base)
    data, raw_bank = prepare_data(args, model, processors, training, domain, base)
    data["patch_count"] = raw_bank.size(1) - 1
    components = compute_components(args, model, data, raw_bank, configs, audit=True)
    config_by_id = {cfg["id"]: cfg for cfg in configs}
    source_by_id = {cfg["id"]: cfg["id"] for cfg in configs}
    top_a, _ = trial_top(data, components["region_00"], configs[0])
    top_b, _ = trial_top(data, components["region_01"], configs[1])
    baseline_a, baseline_b = metrics_for(data, top_a), metrics_for(data, top_b)
    report = {"dataset": domain, "queries": data["queries"], "baseline_A": baseline_a,
              "baseline_B": baseline_b, "stage1_only": metrics_for(data, data["stage1_top50"]),
              "trials": [], "best_feasible_B": None,
              "selection_warning": "Selected on this validation domain; not an unbiased test result or a cross-domain shared optimum"}
    domain_reports = previous + [report]

    def record(config, source, stage, gate=None):
        row, top = evaluate_trial(data, components[source], config, baseline_a, baseline_b, args, stage, gate)
        report["trials"].append(row)
        config_by_id[config["id"]] = config
        source_by_id[config["id"]] = source
        report["best_feasible_B"] = best_trial(report["trials"])
        print_trial(row)
        write_report(output, meta, domain_reports)
        return row, top

    for cfg in configs:
        record(cfg, cfg["id"], "region_one_factor")
    # Combine the individually best feasible setting for each region axis.
    # The resulting interaction is actually recomputed, never inferred from OAT results.
    combined = dict(base)
    for axis in REGION_GRID:
        rows = [row for row in report["trials"] if row["axis"] in ["baseline_B", axis]]
        best = best_trial(rows)
        if best:
            combined[axis] = best["params"][axis]
    if combined["region_topk"] > combined["region_topk_max"]:
        combined["region_topk_max"] = combined["region_topk"]
    combined_cfg = {"id": "region_combined", "mode": "B_comp_only", "params": combined, "axis": "combined_OAT_winners"}
    existing = next((cfg for cfg in configs if cfg["mode"] == "B_comp_only" and cfg["params"] == combined), None)
    if existing is None:
        components.update(compute_components(args, model, data, raw_bank, [combined_cfg]))
        combined_source = combined_cfg["id"]
    else:
        combined_source = existing["id"]
    combined_row, _ = record(combined_cfg, combined_source, "combined_region_test")
    # Scan final weights on B baseline, best feasible OAT, highest-ID OAT,
    # and the combined configuration. This is cheap score-only arithmetic.
    region_rows = [row for row in report["trials"] if row["stage"] == "region_one_factor"]
    seed_rows = [region_rows[1], best_trial(region_rows), best_trial(region_rows, feasible=False), combined_row]
    seeds, seen = [], set()
    for row in seed_rows:
        if row is None:
            continue
        source = source_by_id[row["trial_id"]]
        if source not in seen:
            seen.add(source); seeds.append((row, source))
    for seed_index, (seed, source) in enumerate(seeds):
        fusions = list(dict.fromkeys([base["core_matcher_fusion_weight"], *FUSIONS]))
        identities = list(dict.fromkeys([base["core_matcher_lambda_id"], *IDENTITIES]))
        for fw in fusions:
            for lid in identities:
                cfg = {"id": f"weight_s{seed_index}_f{fw:g}_id{lid:g}", "mode": "B_comp_only", "axis": "fusion_x_lambda_id",
                       "params": {**seed["params"], "core_matcher_fusion_weight": fw, "core_matcher_lambda_id": lid}}
                record(cfg, source, "final_weight_grid")
    # Gate only the current best feasible policy. The gate sees scores, not labels.
    selected = best_trial(report["trials"])
    if selected:
        source = source_by_id[selected["trial_id"]]
        for i, threshold in enumerate(GATE_THRESHOLDS):
            cfg = {"id": f"gate_{i:02d}", "mode": "B_comp_only", "params": dict(selected["params"]), "axis": "identity_gain_threshold"}
            record(cfg, source, "confidence_gate", {"kind": "identity_gain", "threshold": threshold})
        cfg = {"id": "gate_stage1_only", "mode": "B_comp_only", "params": dict(selected["params"]), "axis": "gate_control"}
        record(cfg, source, "confidence_gate", {"kind": "stage1_only"})
    winner = best_trial(report["trials"])
    report["best_feasible_B"] = winner
    if winner:
        cfg = config_by_id[winner["trial_id"]]
        top_best, _ = trial_top(data, components[source_by_id[winner["trial_id"]]], cfg, winner["gate"])
    else:
        top_best = top_b
    save_query_changes(output, data, top_a, top_b, top_best)
    # Small CPU score cache: no raw image cache or trained model is duplicated.
    score_path = output / f"{domain.lower()}_score_cache.pt"
    torch.save({"metadata": meta, "query_data": data, "configs": config_by_id,
                "component_source_by_trial": source_by_id, "components": components}, score_path)
    report["score_cache"] = str(score_path)
    write_report(output, meta, domain_reports)
    print(f"\n{domain} best feasible B:", json.dumps(winner, ensure_ascii=False, indent=2), flush=True)
    del raw_bank, components, data
    return report


def self_test():
    # Original matching function serves as the oracle; no pretrained weights.
    from lavis.models.blip2_models.oacir_latent_matching import compute_bidirectional_region_scores
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    base = dict(DEFAULTS)
    base["region_topk"] = 2
    base["region_topk_max"] = 12
    area = torch.tensor([2, 6])
    patches = torch.randn(2, 4, 16, 8)
    regions = torch.randn(2, 3, 8)
    composed = torch.randn(2, 8)
    primitive = primitives(regions, patches, composed)
    tests = [{**base}, *[{**base, axis: value} for axis, values in REGION_GRID.items()
                        if axis not in ["core_matcher_topk", "core_matcher_bbox_floor"] for value in values]]
    for params in tests:
        sizes = region_sizes(area, 16, params)
        kwargs = {"spatial_kernel_size": params["region_spatial_kernel"], "spatial_weight": params["region_spatial_weight"],
                  "region_temperature": params["region_temperature"], "cycle_weight": params["core_matcher_cycle_weight"]}
        oracle_id = compute_bidirectional_region_scores(regions, patches, composed, sizes, text_weight=0., **kwargs)
        oracle_a = compute_bidirectional_region_scores(regions, patches, composed, sizes,
                                                       text_weight=params["core_matcher_text_weight"], **kwargs)
        for mode in ["A_original", "B_comp_only"]:
            sid, comp, _ = score_regions(primitive, area, {"mode": mode, "params": params})
            torch.testing.assert_close(sid, oracle_id["score_id"], rtol=0, atol=0)
            if mode == "A_original":
                torch.testing.assert_close(comp, oracle_a["score_comp"], rtol=0, atol=0)
            else:
                zero_ref = torch.zeros_like(regions[:, :1])
                oracle_b = compute_bidirectional_region_scores(zero_ref, patches, composed, sizes,
                    spatial_kernel_size=params["region_spatial_kernel"], spatial_weight=params["region_spatial_weight"],
                    region_temperature=params["region_temperature"], cycle_weight=0., text_weight=1.)
                torch.testing.assert_close(comp, oracle_b["score_comp"], rtol=0, atol=0)
    print("PASS all region axes: A/identity match original; B=response cosine x1.0")
    from lavis.models.blip2_models.blip2_qformer_oacir_latent import Blip2QformerOacirLatent
    for n in [9, 90, 300]:
        backbone = torch.randn(4, n)
        if n == 90:
            backbone = (backbone * 3).round()  # Exercise tied first-stage scores.
        chosen = {k: backbone.topk(k, dim=1).indices for k in sorted({1, min(20, n), min(50, n), min(100, n)})}
        banks, positions = union_candidates(chosen)
        for k, indices in chosen.items():
            for i in range(4):
                assert torch.equal(banks[i][positions[k][i]], indices[i])
            order = backbone.argsort(1, descending=True)
            selected = torch.zeros_like(backbone, dtype=torch.bool).scatter_(1, indices, True)
            tail = order[~selected.gather(1, order)].reshape(4, n - k)[:, :51]
            context = {"indices": indices, "tail": tail}
            references = torch.tensor([indices[0, 0], indices[1, -1], backbone[2].argmin(), 3])
            score = torch.randn(4, k)
            ours = rank_context(context, score, references, n)
            merged = Blip2QformerOacirLatent.merge_topk_ranking(backbone, indices, score)
            merged[torch.arange(4), references] = -torch.inf
            assert torch.equal(ours, merged.topk(min(50, n - 1), dim=1).indices)
    print("PASS exact candidate sets, tied scores, original merge and reference exclusion")
    assert len(region_plan(DEFAULTS)) == 24
    backbone = torch.randn(3, 5); sid = torch.randn(3, 5); comp = torch.randn(3, 5)
    expected = backbone + .4 * ((comp + 1.25 * sid) / .07)
    torch.testing.assert_close(final_scores(backbone, sid, comp, DEFAULTS), expected, rtol=0, atol=0)
    print("PASS 24 initial trials and original final-score formula")
    synthetic_pipeline_test()
    print("ALL_INFERENCE_SWEEP_CHECKS_PASS")


def synthetic_pipeline_test():
    """Real region encoder/original scorer, synthetic tensors; no real data or GPU."""
    import contextlib
    import io
    from types import SimpleNamespace
    from torch import nn
    from lavis.models.blip2_models.blip2_qformer_oacir_latent import CoreReferenceRegionEncoder, CoreAdaptiveFusion
    from lavis.models.blip2_models.blip2_qformer_oacir_latent_split_mask import Blip2QformerOacirLatentSplitMask

    class Tiny(Blip2QformerOacirLatentSplitMask):
        def __init__(self):
            nn.Module.__init__(self)
            self.ln_vision = nn.LayerNorm(8)
            self.instance_proj = nn.Linear(8, 8)
            self.core_reference_encoder = CoreReferenceRegionEncoder(8, num_regions=3)
            self.core_adaptive_fusion = CoreAdaptiveFusion(8)
            self.latent_image_size = 224
            self.latent_bbox_format = "xyxy"
            self.temp = nn.Parameter(torch.tensor(.07))
            self.set_latent_config(**DEFAULTS)

        def _compute_fusion_features(self, normalized, text, boxes):
            return F.normalize(normalized[:, 1:].mean(1) + text, dim=-1)

    model = Tiny().eval()
    queries, gallery_size = 4, 120
    raw = torch.randn(gallery_size, 17, 8)
    refs = torch.tensor([0, 2, 4, 6])
    targets = torch.tensor([1, 3, 5, 7])
    text = torch.randn(queries, 8)
    boxes = [[0, 0, 112, 112], [56, 56, 224, 224], [0, 112, 224, 224], [0, 0, 224, 224]]
    gallery = F.normalize(torch.randn(gallery_size, 4, 8), dim=-1)
    with torch.inference_mode():
        actual = first_stage(model, raw[refs], gallery, text, boxes, 31)
        expected = model.inference_composition_from_raw(raw[refs], gallery, text, boxes)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    scores = torch.randn(queries, gallery_size)
    order = scores.argsort(1, descending=True)
    chosen = {k: scores.topk(k, dim=1).indices for k in [20, 50, 100]}
    banks, maps = union_candidates(chosen)
    width = max(len(row) for row in banks)
    indices = torch.stack([F.pad(row, (0, width - len(row)), value=int(row[0])) for row in banks])
    contexts = {}
    for k, selected in chosen.items():
        mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, selected, True)
        tail = order[~mask.gather(1, order)].reshape(queries, gallery_size - k)[:, :51]
        contexts[k] = {"indices": selected, "tail": tail, "positions": maps[k]}
    stage1 = order[~order.eq(refs[:, None])].reshape(queries, -1)[:, :50]
    data = {"names": [f"synthetic_{i}" for i in range(gallery_size)], "gallery_folders": torch.arange(gallery_size) // 2,
            "gallery_size": gallery_size, "contexts": contexts, "domain": "SYNTHETIC_TEST_ONLY",
            "queries": queries, "boxes": boxes, "reference_indices": refs, "target_indices": targets,
            "text_features": text, "bank_indices": indices, "bank_backbone": scores.gather(1, indices),
            "references": [f"synthetic_{i}" for i in refs.tolist()],
            "targets": [f"synthetic_{i}" for i in targets.tolist()], "captions": ["synthetic test only"] * queries,
            "stage1_top50": stage1}
    args = SimpleNamespace(query_chunk_size=2, max_r1_drop=100.)
    global prepare_data
    original_prepare = prepare_data
    try:
        prepare_data = lambda *unused: (data, raw)
        with tempfile.TemporaryDirectory(prefix="oacir_inference_selftest_") as directory:
            output = Path(directory)
            with torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                report = run_domain(args, model, None, {}, dict(DEFAULTS), data["domain"], output,
                                    {"synthetic_test_only": True}, [])
            assert len([r for r in report["trials"] if r["stage"] == "region_one_factor"]) == 24
            assert any(r["stage"] == "combined_region_test" for r in report["trials"])
            assert any(r["stage"] == "final_weight_grid" for r in report["trials"])
            gate_rows = [r for r in report["trials"] if r["stage"] == "confidence_gate"]
            assert len(gate_rows) == len(GATE_THRESHOLDS) + 1
            control = next(r for r in gate_rows if r["gate"]["kind"] == "stage1_only")
            assert control["metrics"] == report["stage1_only"]
            with (output / "trials.csv").open(encoding="utf-8-sig", newline="") as handle:
                assert len(list(csv.DictReader(handle))) == len(report["trials"])
            saved = load_torch(Path(report["score_cache"]))
            assert saved["components"]["region_00"]["sid"].shape == (queries, width)
            assert json.loads((output / "report.json").read_text())["metadata"]["synthetic_test_only"]
    finally:
        prepare_data = original_prepare
    print("PASS synthetic end-to-end: original CORE encoder/scorer audit, cached stage1, all stages, gate, JSON/CSV/PT")


def main():
    args = arguments()
    if args.show_plan:
        print(json.dumps(parameter_plan(DEFAULTS), ensure_ascii=False, indent=2))
        return
    if args.batch_size < 1 or args.query_chunk_size < 1 or args.num_workers < 0 or args.cpu_threads < 1 or args.max_queries < 0:
        raise ValueError("Invalid batch/chunk/worker/thread/query-limit setting")
    if not math.isfinite(args.max_r1_drop) or args.max_r1_drop < 0:
        raise ValueError("max-r1-drop must be finite and nonnegative")
    repo = find_repo(args.repo_root)
    # Repository-relative defaults work even when invoked from another cwd.
    ckpt = Path(args.checkpoint).expanduser()
    ckpt = (repo / ckpt).resolve() if not ckpt.is_absolute() else ckpt.resolve()
    if not args.self_test and not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}; pass --checkpoint /actual/file.pt")
    args.checkpoint = str(ckpt)
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        repo / "checkpoints" / "inference_parameter_sweep" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    runtime(args, repo)
    if args.self_test:
        self_test()
        return
    model, processors, training, base, meta = load_model(args)
    output.mkdir(parents=True, exist_ok=False)
    meta.update({"search_plan": parameter_plan(base), "repo_root": str(repo),
                 "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "max_queries": args.max_queries, "max_R1_drop_pp": args.max_r1_drop,
                 "constraint_baseline": "A original at the checkpoint's inference parameters",
                 "device": str(model._get_device()), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                 "selection_split": "validation", "parameters_optimized_independently_per_domain": True,
                 "final_formula": "S_stage1 + fusion_weight * (S_comp + lambda_id * S_id) / core_matcher_temp"})
    print(json.dumps(meta["search_plan"], ensure_ascii=False, indent=2), flush=True)
    results = []
    with torch.inference_mode():
        for domain in dict.fromkeys(args.dataset):
            results.append(run_domain(args, model, processors, training, base, domain, output, meta, results))
    print(f"\nFinished. Results: {output}\nSee report.json, trials.csv, best.json and *_query_changes.csv", flush=True)


if __name__ == "__main__":
    main()
