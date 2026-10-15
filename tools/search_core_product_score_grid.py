import argparse
import csv
import json
import sys
from operator import itemgetter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_utils import OACIRRDataset, targetpad_transform
from evaluate import extract_index_blip_features_from_raw_cache, load_checkpoint_safely
from lavis.models import load_model_and_preprocess
from utils import custom_collate_fn, device


RECALL_KEYS = ("R_ID@1", "R_ID@3", "R_ID@5", "R@1", "R@5", "R@10", "R@50")
DEFAULT_FUSION_VALUES = (
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
    0.65,
    0.80,
    1.00,
    1.25,
    1.50,
)
DEFAULT_LAMBDA_VALUES = (0.00, 0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 3.00)


def parse_args():
    parser = argparse.ArgumentParser(
        "Broad score-only fusion/lambda search on OACIRR Product"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="./Datasets/OACIRR")
    parser.add_argument(
        "--val-feature-cache",
        default="./cache/oacirr_product_val_vitg_targetpad125_raw_fp16.pt",
    )
    parser.add_argument("--output-dir", default="./core_matcher_product_score_search")
    parser.add_argument("--score-cache", default=None)
    parser.add_argument(
        "--reuse-score-cache",
        action="store_true",
        help="Skip model inference and reuse --score-cache from an earlier run.",
    )
    parser.add_argument("--blip-model-name", default="oacir_latent")
    parser.add_argument("--vit-backbone", default="pretrain")
    parser.add_argument("--target-ratio", default=1.25, type=float)
    parser.add_argument("--val-feature-batch-size", default=128, type=int)
    parser.add_argument("--eval-batch-size", default=64, type=int)
    parser.add_argument("--gallery-chunk-size", default=2048, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--max-queries", default=0, type=int)
    parser.add_argument("--core-matcher-temp", default=0.07, type=float)
    parser.add_argument("--core-matcher-topk", default=50, type=int)
    parser.add_argument("--core-matcher-cycle-weight", default=0.25, type=float)
    parser.add_argument("--core-matcher-text-weight", default=0.75, type=float)
    parser.add_argument("--core-matcher-bbox-floor", default=0.05, type=float)
    parser.add_argument("--core-matcher-query-chunk-size", default=8, type=int)
    parser.add_argument("--region-topk", default=16, type=int)
    parser.add_argument("--region-topk-max", default=96, type=int)
    parser.add_argument("--region-area-scale", default=1.5, type=float)
    parser.add_argument("--region-spatial-kernel", default=3, type=int)
    parser.add_argument("--region-spatial-weight", default=0.15, type=float)
    parser.add_argument("--region-temperature", default=0.07, type=float)
    parser.add_argument(
        "--fusion-values",
        nargs="+",
        type=float,
        default=list(DEFAULT_FUSION_VALUES),
    )
    parser.add_argument(
        "--lambda-values",
        nargs="+",
        type=float,
        default=list(DEFAULT_LAMBDA_VALUES),
    )
    parser.add_argument("--fine-seeds", default=3, type=int)
    parser.add_argument("--no-fine-search", action="store_true")
    parser.add_argument("--top-results", default=30, type=int)
    return parser.parse_args()


def make_config(fusion, lambda_id, stage):
    fusion = round(max(float(fusion), 0.0), 4)
    lambda_id = round(max(float(lambda_id), 0.0), 4)
    if fusion == 0.0:
        lambda_id = 0.0
    return {"fusion": fusion, "lambda_id": lambda_id, "stage": stage}


def deduplicate_configs(configs):
    unique = {}
    for config in configs:
        key = (config["fusion"], config["lambda_id"])
        if key not in unique or config["stage"] == "coarse":
            unique[key] = config
    return list(unique.values())


def make_coarse_configs(args):
    configs = [
        make_config(fusion, lambda_id, "coarse")
        for fusion in args.fusion_values
        for lambda_id in args.lambda_values
    ]
    return deduplicate_configs(configs)


def make_fine_configs(seed_rows):
    configs = []
    fusion_offsets = (-0.05, -0.025, 0.0, 0.025, 0.05)
    lambda_offsets = (-0.25, -0.125, 0.0, 0.125, 0.25)
    for seed in seed_rows:
        for fusion_offset in fusion_offsets:
            for lambda_offset in lambda_offsets:
                configs.append(
                    make_config(
                        seed["fusion"] + fusion_offset,
                        seed["lambda_id"] + lambda_offset,
                        "fine",
                    )
                )
    return deduplicate_configs(configs)


def rank_key(row):
    return (row["R@1"], row["R@5"], row["Avg7"], row["R_ID@1"])


def build_top_indices(cache, candidate_scores):
    reranked_order = candidate_scores.argsort(dim=1, descending=True)
    ordered_candidates = torch.gather(cache["candidate_indices"], 1, reranked_order)
    fallback = cache["backbone_order"][:, cache["candidate_indices"].size(1):]
    merged = torch.cat([ordered_candidates, fallback], dim=1)

    positions = torch.arange(merged.size(1)).unsqueeze(0).expand_as(merged)
    positions = positions.masked_fill(
        merged.eq(cache["reference_indices"].unsqueeze(1)),
        merged.size(1),
    )
    keep_positions = positions.topk(
        min(50, merged.size(1) - 1),
        dim=1,
        largest=False,
        sorted=True,
    ).indices
    return torch.gather(merged, 1, keep_positions)


def compute_metrics(cache, config):
    core_logits = (
        cache["comp_logits"]
        + float(config["lambda_id"]) * cache["id_logits"]
    )
    candidate_scores = (
        cache["candidate_ada"]
        + float(config["fusion"]) * core_logits
    )
    top_indices = build_top_indices(cache, candidate_scores)
    image_hits = top_indices.eq(cache["target_indices"].unsqueeze(1))
    class_hits = cache["gallery_folders"][top_indices].eq(
        cache["target_folder_indices"].unsqueeze(1)
    )

    queries = max(top_indices.size(0), 1)
    metrics = {}
    for k in (1, 5, 10, 50):
        metrics[f"R@{k}"] = (
            100.0
            * image_hits[:, : min(k, image_hits.size(1))].any(dim=1).sum().item()
            / queries
        )
    for k in (1, 3, 5):
        metrics[f"R_ID@{k}"] = (
            100.0
            * class_hits[:, : min(k, class_hits.size(1))].any(dim=1).sum().item()
            / queries
        )

    metrics["Avg7"] = sum(metrics[key] for key in RECALL_KEYS) / len(RECALL_KEYS)
    metrics["R_mean_1_5"] = (metrics["R@1"] + metrics["R@5"]) / 2.0
    metrics["primary3"] = (
        metrics["R_ID@1"] + metrics["R@1"] + metrics["R@5"]
    ) / 3.0
    return {**config, **metrics}


def evaluate_grid(cache, configs, description):
    results = []
    for config in tqdm(configs, desc=description, ncols=120, ascii=True):
        results.append(compute_metrics(cache, config))
    return results


def compact_cache(cache):
    return {
        key: value.contiguous() if torch.is_tensor(value) else value
        for key, value in cache.items()
    }


@torch.no_grad()
def extract_score_components(args):
    model, _, text_processors = load_model_and_preprocess(
        name=args.blip_model_name,
        model_type=args.vit_backbone,
        is_eval=False,
        device=device,
    )
    load_checkpoint_safely(model, args.checkpoint)
    model.set_latent_config(
        core_matcher_temp=args.core_matcher_temp,
        core_matcher_lambda_id=1.0,
        core_matcher_fusion_weight=1.0,
        core_matcher_topk=args.core_matcher_topk,
        core_matcher_cycle_weight=args.core_matcher_cycle_weight,
        core_matcher_text_weight=args.core_matcher_text_weight,
        core_matcher_bbox_floor=args.core_matcher_bbox_floor,
        core_matcher_query_chunk_size=args.core_matcher_query_chunk_size,
        region_topk=args.region_topk,
        region_topk_max=args.region_topk_max,
        region_area_scale=args.region_area_scale,
        region_spatial_kernel=args.region_spatial_kernel,
        region_spatial_weight=args.region_spatial_weight,
        region_temperature=args.region_temperature,
    )
    model.eval()
    latent_model = model.module if hasattr(model, "module") else model

    preprocess = targetpad_transform(args.target_ratio, 224)
    relative_dataset = OACIRRDataset(
        data_root=args.data_root,
        variant="Product",
        split="val",
        mode="relative",
        preprocess=preprocess,
        highlight_inference=True,
    )
    classic_dataset = OACIRRDataset(
        data_root=args.data_root,
        variant="Product",
        split="val",
        mode="classic",
        preprocess=preprocess,
    )
    index_features, index_raw, index_names = extract_index_blip_features_from_raw_cache(
        classic_dataset,
        model,
        args.val_feature_cache,
        save_memory=True,
        batch_size=args.val_feature_batch_size,
    )

    loader = DataLoader(
        relative_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        shuffle=False,
    )
    name_to_gallery = {name: idx for idx, name in enumerate(index_names)}
    name_to_raw = dict(zip(index_names, index_raw))
    name_to_folder = {
        name: path.split("/")[-2]
        for name, path in relative_dataset.name_to_relpath.items()
    }
    folders = list(dict.fromkeys(name_to_folder.values()))
    folder_to_index = {folder: idx for idx, folder in enumerate(folders)}
    gallery_folders = torch.tensor(
        [folder_to_index[name_to_folder[name]] for name in index_names],
        dtype=torch.long,
    )

    collected = {
        "candidate_indices": [],
        "candidate_ada": [],
        "id_logits": [],
        "comp_logits": [],
        "backbone_order": [],
        "reference_indices": [],
        "target_indices": [],
        "target_folder_indices": [],
    }
    processed_queries = 0
    progress = tqdm(loader, desc="Extract Product score components", ncols=140, ascii=True)
    for batch in progress:
        if args.max_queries and processed_queries >= args.max_queries:
            break
        reference_names = batch[0]
        target_names = batch[1]
        captions = batch[2]
        reference_bboxes = batch[4] if len(batch) == 5 else batch[3]
        if args.max_queries:
            remaining = args.max_queries - processed_queries
            reference_names = reference_names[:remaining]
            target_names = target_names[:remaining]
            captions = captions[:remaining]
            reference_bboxes = reference_bboxes[:remaining]
        captions = [text_processors["eval"](caption) for caption in captions]
        batch_size = len(captions)
        if batch_size == 0:
            break

        if batch_size == 1:
            reference_raw = itemgetter(*reference_names)(name_to_raw).unsqueeze(0)
        else:
            reference_raw = torch.stack(itemgetter(*reference_names)(name_to_raw))

        global_chunks = []
        for start in range(0, len(index_names), args.gallery_chunk_size):
            feature_chunk = index_features[start:start + args.gallery_chunk_size]
            scores = latent_model.inference_composition_from_raw(
                reference_image_embeds_raw=reference_raw,
                target_features=feature_chunk,
                modification_text=captions,
                reference_bbox=reference_bboxes,
            )
            global_chunks.append(scores.detach().cpu())
        global_scores = torch.cat(global_chunks, dim=1)

        candidate_count = min(args.core_matcher_topk, global_scores.size(1))
        order_size = min(global_scores.size(1), candidate_count + 51)
        backbone_order = global_scores.topk(order_size, dim=1, sorted=True).indices
        candidate_indices = backbone_order[:, :candidate_count]
        candidate_ada = torch.gather(global_scores, 1, candidate_indices)

        id_chunks = []
        comp_chunks = []
        for start in range(0, batch_size, args.core_matcher_query_chunk_size):
            end = min(start + args.core_matcher_query_chunk_size, batch_size)
            query_candidates = candidate_indices[start:end]
            target_raw = index_raw[query_candidates.to(index_raw.device)]
            output = latent_model.inference_core_matcher_candidates(
                reference_image_embeds_raw=reference_raw[start:end],
                target_candidate_embeds_raw=target_raw,
                composition_candidate_logits=candidate_ada[start:end],
                modification_text=captions[start:end],
                reference_bbox=reference_bboxes[start:end],
                return_parts=True,
            )
            id_chunks.append(output["sim_id"].detach().cpu())
            comp_chunks.append(output["sim_region_comp"].detach().cpu())
            del target_raw, output

        collected["candidate_indices"].append(candidate_indices)
        collected["candidate_ada"].append(candidate_ada)
        collected["id_logits"].append(torch.cat(id_chunks, dim=0))
        collected["comp_logits"].append(torch.cat(comp_chunks, dim=0))
        collected["backbone_order"].append(backbone_order)
        collected["reference_indices"].append(
            torch.tensor([name_to_gallery[name] for name in reference_names])
        )
        collected["target_indices"].append(
            torch.tensor([name_to_gallery[name] for name in target_names])
        )
        collected["target_folder_indices"].append(
            torch.tensor([folder_to_index[name_to_folder[name]] for name in target_names])
        )

        processed_queries += batch_size
        progress.set_postfix(queries=processed_queries)
        del global_scores, candidate_indices, candidate_ada, backbone_order

    cache = {key: torch.cat(value, dim=0) for key, value in collected.items()}
    cache["gallery_folders"] = gallery_folders
    cache["meta"] = {
        "dataset": "Product",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "queries": processed_queries,
        "core_matcher_temp": args.core_matcher_temp,
        "core_matcher_text_weight": args.core_matcher_text_weight,
        "core_matcher_cycle_weight": args.core_matcher_cycle_weight,
        "core_matcher_topk": args.core_matcher_topk,
        "region_topk": args.region_topk,
        "region_topk_max": args.region_topk_max,
        "region_area_scale": args.region_area_scale,
        "region_spatial_kernel": args.region_spatial_kernel,
        "region_spatial_weight": args.region_spatial_weight,
        "region_temperature": args.region_temperature,
    }
    return compact_cache(cache)


def validate_loaded_cache(args, cache):
    meta = cache.get("meta", {})
    expected_checkpoint = str(Path(args.checkpoint).resolve())
    if meta.get("dataset") != "Product":
        raise ValueError("The score cache is not a Product cache")
    if meta.get("checkpoint") != expected_checkpoint:
        raise ValueError(
            "Score cache checkpoint mismatch: "
            f"cached={meta.get('checkpoint')}, requested={expected_checkpoint}"
        )
    expected = {
        "core_matcher_temp": args.core_matcher_temp,
        "core_matcher_text_weight": args.core_matcher_text_weight,
        "core_matcher_cycle_weight": args.core_matcher_cycle_weight,
        "core_matcher_topk": args.core_matcher_topk,
        "region_topk": args.region_topk,
        "region_topk_max": args.region_topk_max,
        "region_area_scale": args.region_area_scale,
        "region_spatial_kernel": args.region_spatial_kernel,
        "region_spatial_weight": args.region_spatial_weight,
        "region_temperature": args.region_temperature,
    }
    mismatches = {
        key: (meta.get(key), value)
        for key, value in expected.items()
        if meta.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Score cache setting mismatch: {mismatches}")


def write_results(args, cache, results):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(results, key=rank_key, reverse=True)
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank

    fields = [
        "rank",
        "stage",
        "fusion",
        "lambda_id",
        *RECALL_KEYS,
        "Avg7",
        "R_mean_1_5",
        "primary3",
    ]
    csv_path = output_dir / "product_score_search_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)

    json_path = output_dir / "product_score_search_results.json"
    payload = {
        "dataset": "Product",
        "checkpoint": args.checkpoint,
        "score_cache": str(args.score_cache),
        "score_cache_meta": cache["meta"],
        "ranking_rule": "R@1 descending, then R@5, Avg7, R_ID@1.",
        "results": ordered,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    print("\n" + "=" * 118)
    print("Top Product score-only configurations")
    print(
        f"{'Rank':>4} {'Stage':>7} {'Fusion':>8} {'Lambda':>8} "
        f"{'R_ID@1':>9} {'R@1':>9} {'R@5':>9} {'R@10':>9} "
        f"{'Avg7':>9} {'Primary3':>9}"
    )
    for row in ordered[: max(args.top_results, 1)]:
        print(
            f"{row['rank']:>4} {row['stage']:>7} {row['fusion']:>8.3f} "
            f"{row['lambda_id']:>8.3f} {row['R_ID@1']:>9.3f} "
            f"{row['R@1']:>9.3f} {row['R@5']:>9.3f} "
            f"{row['R@10']:>9.3f} {row['Avg7']:>9.3f} "
            f"{row['primary3']:>9.3f}"
        )
    print("=" * 118)
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")


def main():
    args = parse_args()
    if args.core_matcher_topk < 2:
        raise ValueError("--core-matcher-topk must be at least 2")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.score_cache is None:
        args.score_cache = str(output_dir / "product_score_components.pt")
    score_cache_path = Path(args.score_cache)

    if args.reuse_score_cache:
        if not score_cache_path.exists():
            raise FileNotFoundError(f"Score cache not found: {score_cache_path}")
        cache = torch.load(score_cache_path, map_location="cpu")
        validate_loaded_cache(args, cache)
        print(f"Reusing score cache: {score_cache_path}")
    else:
        cache = extract_score_components(args)
        score_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, score_cache_path)
        print(f"Saved reusable score cache: {score_cache_path}")

    coarse_configs = make_coarse_configs(args)
    coarse_results = evaluate_grid(cache, coarse_configs, "Coarse score grid")
    all_results = list(coarse_results)

    if not args.no_fine_search and args.fine_seeds > 0:
        seeds = sorted(coarse_results, key=rank_key, reverse=True)[: args.fine_seeds]
        fine_configs = make_fine_configs(seeds)
        coarse_keys = {(row["fusion"], row["lambda_id"]) for row in coarse_results}
        fine_configs = [
            config
            for config in fine_configs
            if (config["fusion"], config["lambda_id"]) not in coarse_keys
        ]
        all_results.extend(evaluate_grid(cache, fine_configs, "Fine score grid"))

    write_results(args, cache, all_results)


if __name__ == "__main__":
    main()
