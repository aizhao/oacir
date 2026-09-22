import argparse
import json
import sys
from collections import defaultdict
from operator import itemgetter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_utils import OACIRRDataset, squarepad_transform, targetpad_transform
from evaluate import extract_index_blip_features_from_raw_cache, load_checkpoint_safely
from lavis.models import load_model_and_preprocess
from utils import custom_collate_fn, device


RECALL_KEYS = ("R_ID@1", "R_ID@3", "R_ID@5", "R@1", "R@5", "R@10", "R@50")


def parse_args():
    parser = argparse.ArgumentParser(
        "Diagnose score-scale mismatch and ranking conflict in AdaFocal + CORE"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["Fashion", "Car", "Product", "Landmark"],
    )
    parser.add_argument("--data-root", default="./Datasets/OACIRR")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-feature-cache", required=True)
    parser.add_argument("--blip-model-name", default="oacir_latent")
    parser.add_argument("--vit-backbone", default="pretrain")
    parser.add_argument("--transform", default="targetpad", choices=["targetpad", "squarepad"])
    parser.add_argument("--target-ratio", default=1.25, type=float)
    parser.add_argument("--val-feature-batch-size", default=32, type=int)
    parser.add_argument("--eval-batch-size", default=16, type=int)
    parser.add_argument("--gallery-chunk-size", default=1024, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument(
        "--max-queries",
        default=0,
        type=int,
        help="Optional fast diagnostic limit; 0 evaluates the complete validation set.",
    )
    parser.add_argument(
        "--fusion-weights",
        nargs="+",
        type=float,
        default=[0.10, 0.20, 0.35, 0.50, 0.75, 1.00],
    )
    parser.add_argument("--output", default=None)

    parser.add_argument("--core-matcher-temp", default=0.07, type=float)
    parser.add_argument("--core-matcher-lambda-id", default=1.0, type=float)
    parser.add_argument("--core-matcher-fusion-weight", default=0.35, type=float)
    parser.add_argument("--core-matcher-topk", default=50, type=int)
    parser.add_argument("--core-matcher-cycle-weight", default=0.25, type=float)
    parser.add_argument("--core-matcher-text-weight", default=0.25, type=float)
    parser.add_argument("--core-matcher-bbox-floor", default=0.05, type=float)
    parser.add_argument("--core-matcher-query-chunk-size", default=8, type=int)
    parser.add_argument("--region-topk", default=16, type=int)
    parser.add_argument("--region-topk-max", default=96, type=int)
    parser.add_argument("--region-area-scale", default=1.5, type=float)
    parser.add_argument("--region-spatial-kernel", default=3, type=int)
    parser.add_argument("--region-spatial-weight", default=0.15, type=float)
    parser.add_argument("--region-temperature", default=0.07, type=float)
    return parser.parse_args()


def zscore(scores):
    centered = scores - scores.mean(dim=1, keepdim=True)
    scale = centered.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
    return centered / scale


def rank_correlation(scores_a, scores_b):
    rank_a = scores_a.argsort(dim=1).argsort(dim=1).float()
    rank_b = scores_b.argsort(dim=1).argsort(dim=1).float()
    rank_a = rank_a - rank_a.mean(dim=1, keepdim=True)
    rank_b = rank_b - rank_b.mean(dim=1, keepdim=True)
    return F.cosine_similarity(rank_a, rank_b, dim=1)


def summarize(values):
    if not values:
        return {"mean": None, "p10": None, "median": None, "p90": None}
    tensor = torch.cat(values).float()
    return {
        "mean": tensor.mean().item(),
        "p10": torch.quantile(tensor, 0.10).item(),
        "median": torch.quantile(tensor, 0.50).item(),
        "p90": torch.quantile(tensor, 0.90).item(),
    }


def build_top_indices(
    backbone_order,
    candidate_indices,
    candidate_scores,
    reference_indices,
    metric_topk,
):
    """Merge a reranked candidate set without sorting the full gallery again."""
    reranked_order = candidate_scores.argsort(dim=1, descending=True)
    ordered_candidates = torch.gather(candidate_indices, 1, reranked_order)
    merged_rows = []
    for row in range(candidate_indices.size(0)):
        base_row = backbone_order[row]
        candidate_row = candidate_indices[row]
        is_candidate = base_row.unsqueeze(1).eq(candidate_row.unsqueeze(0)).any(dim=1)
        ordered_non_candidates = base_row[~is_candidate]
        merged = torch.cat([ordered_candidates[row], ordered_non_candidates])
        merged = merged[merged.ne(reference_indices[row])]
        if merged.numel() < metric_topk:
            raise RuntimeError("Insufficient fallback candidates for metric computation")
        merged_rows.append(merged[:metric_topk])
    return torch.stack(merged_rows)


def update_recall_counts(counts, top_indices, target_indices, target_folder_indices, gallery_folders):
    image_hits = top_indices.eq(target_indices.unsqueeze(1))
    class_hits = gallery_folders[top_indices].eq(target_folder_indices.unsqueeze(1))
    counts["queries"] += top_indices.size(0)
    for k in (1, 5, 10, 50):
        counts[f"R@{k}"] += image_hits[:, : min(k, image_hits.size(1))].any(dim=1).sum().item()
    for k in (1, 3, 5):
        counts[f"R_ID@{k}"] += class_hits[:, : min(k, class_hits.size(1))].any(dim=1).sum().item()


def finalize_recall(counts):
    queries = max(int(counts["queries"]), 1)
    return {key: 100.0 * float(counts[key]) / queries for key in RECALL_KEYS}


def primary_score(metrics):
    return (metrics["R_ID@1"] + metrics["R@1"] + metrics["R@5"]) / 3.0


def interpret(result):
    scale_ratio = result["score_diagnostics"]["effective_scale_ratio"]["median"]
    correlation = result["score_diagnostics"]["rank_correlation"]["median"]
    conflict = result["score_diagnostics"]["margin_conflict_rate"]
    correction = result["score_diagnostics"]["correction_rate"]
    damage = result["score_diagnostics"]["damage_rate"]
    raw_score = primary_score(result["metrics"]["raw_fusion"])
    normalized_score = primary_score(result["metrics"]["zscore_fusion"])

    findings = []
    if scale_ratio is not None and (scale_ratio < 0.3 or scale_ratio > 3.0):
        findings.append("The two branches have a severe effective score-scale mismatch.")
    elif scale_ratio is not None and (scale_ratio < 0.5 or scale_ratio > 2.0):
        findings.append("The two branches have a moderate effective score-scale mismatch.")
    else:
        findings.append("The median effective score scales are broadly comparable.")

    if normalized_score > raw_score + 0.2:
        findings.append("Z-score fusion improves the primary metric, supporting scale mismatch as a cause.")
    elif normalized_score < raw_score - 0.2:
        findings.append("Z-score fusion hurts the primary metric, so confidence magnitude currently carries useful information.")
    else:
        findings.append("Z-score fusion changes the primary metric only slightly.")

    if correlation is not None and correlation < 0.3 and conflict > 0.2:
        findings.append("AdaFocal and CORE exhibit substantial ranking conflict.")
    elif correlation is not None and correlation > 0.7 and conflict < 0.1:
        findings.append("AdaFocal and CORE rankings are strongly aligned.")
    else:
        findings.append("AdaFocal and CORE provide partially different rankings.")

    if correction > damage:
        findings.append("CORE corrects more AdaFocal Top-1 errors than it damages.")
    elif damage > correction:
        findings.append("CORE damages more AdaFocal Top-1 decisions than it corrects.")
    else:
        findings.append("CORE correction and damage rates are balanced.")
    return findings


@torch.no_grad()
def main():
    args = parse_args()
    if args.core_matcher_topk < 2:
        raise ValueError("--core-matcher-topk must be at least 2")

    model, _, text_processors = load_model_and_preprocess(
        name=args.blip_model_name,
        model_type=args.vit_backbone,
        is_eval=False,
        device=device,
    )
    load_checkpoint_safely(model, args.checkpoint)
    model.set_latent_config(
        core_matcher_temp=args.core_matcher_temp,
        core_matcher_lambda_id=args.core_matcher_lambda_id,
        core_matcher_fusion_weight=args.core_matcher_fusion_weight,
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

    input_dim = 224
    preprocess = (
        targetpad_transform(args.target_ratio, input_dim)
        if args.transform == "targetpad"
        else squarepad_transform(input_dim)
    )
    relative_dataset = OACIRRDataset(
        data_root=args.data_root,
        variant=args.dataset,
        split="val",
        mode="relative",
        preprocess=preprocess,
        highlight_inference=True,
    )
    classic_dataset = OACIRRDataset(
        data_root=args.data_root,
        variant=args.dataset,
        split="val",
        mode="classic",
        preprocess=preprocess,
    )
    cache_path = args.val_feature_cache.format(
        variant=args.dataset,
        variant_lower=args.dataset.lower(),
    )
    index_features, index_raw, index_names = extract_index_blip_features_from_raw_cache(
        classic_dataset,
        model,
        cache_path,
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

    method_names = ["adafocal", "core_only", "raw_fusion", "zscore_fusion"]
    recall_counts = {name: defaultdict(float) for name in method_names}
    raw_grid_counts = {weight: defaultdict(float) for weight in args.fusion_weights}
    zscore_grid_counts = {weight: defaultdict(float) for weight in args.fusion_weights}
    diagnostics = defaultdict(list)
    diagnostic_counts = defaultdict(float)
    processed_queries = 0

    progress = tqdm(loader, desc=f"Diagnosing [{args.dataset}]", ncols=140, ascii=True)
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
            global_chunk = latent_model.inference_composition_from_raw(
                reference_image_embeds_raw=reference_raw,
                target_features=feature_chunk,
                modification_text=captions,
                reference_bbox=reference_bboxes,
            )
            global_chunks.append(global_chunk.detach().cpu())
        global_scores = torch.cat(global_chunks, dim=1)
        candidate_count = min(args.core_matcher_topk, global_scores.size(1))
        metric_topk = min(50, global_scores.size(1) - 1)
        backbone_order_size = min(
            global_scores.size(1),
            candidate_count + metric_topk + 1,
        )
        backbone_order = global_scores.topk(
            backbone_order_size,
            dim=1,
            largest=True,
            sorted=True,
        ).indices
        candidate_indices = backbone_order[:, :candidate_count]
        candidate_ada = torch.gather(global_scores, 1, candidate_indices)

        core_chunks = []
        for start in range(0, batch_size, args.core_matcher_query_chunk_size):
            end = min(start + args.core_matcher_query_chunk_size, batch_size)
            query_candidates = candidate_indices[start:end]
            target_raw = index_raw[query_candidates.to(index_raw.device)]
            core_output = latent_model.inference_core_matcher_candidates(
                reference_image_embeds_raw=reference_raw[start:end],
                target_candidate_embeds_raw=target_raw,
                composition_candidate_logits=candidate_ada[start:end],
                modification_text=captions[start:end],
                reference_bbox=reference_bboxes[start:end],
                return_parts=True,
            )
            core_chunks.append(core_output["sim_core_matcher"].detach().cpu())
            del target_raw, core_output
        core_scores = torch.cat(core_chunks, dim=0)

        reference_indices = torch.tensor([name_to_gallery[name] for name in reference_names])
        target_indices = torch.tensor([name_to_gallery[name] for name in target_names])
        target_folders = torch.tensor(
            [folder_to_index[name_to_folder[name]] for name in target_names]
        )

        current_weight = args.core_matcher_fusion_weight
        score_sets = {
            "adafocal": candidate_ada,
            "core_only": core_scores,
            "raw_fusion": candidate_ada + current_weight * core_scores,
            "zscore_fusion": zscore(candidate_ada) + current_weight * zscore(core_scores),
        }
        for method_name, candidate_scores in score_sets.items():
            top_indices = build_top_indices(
                backbone_order,
                candidate_indices,
                candidate_scores,
                reference_indices,
                metric_topk,
            )
            update_recall_counts(
                recall_counts[method_name],
                top_indices,
                target_indices,
                target_folders,
                gallery_folders,
            )

        for weight in args.fusion_weights:
            raw_scores = candidate_ada + weight * core_scores
            normalized_scores = zscore(candidate_ada) + weight * zscore(core_scores)
            for scores, counts in (
                (raw_scores, raw_grid_counts[weight]),
                (normalized_scores, zscore_grid_counts[weight]),
            ):
                top_indices = build_top_indices(
                    backbone_order,
                    candidate_indices,
                    scores,
                    reference_indices,
                    metric_topk,
                )
                update_recall_counts(
                    counts,
                    top_indices,
                    target_indices,
                    target_folders,
                    gallery_folders,
                )

        ada_std = candidate_ada.std(dim=1, unbiased=False).clamp_min(1e-4)
        core_std = core_scores.std(dim=1, unbiased=False)
        diagnostics["ada_std"].append(ada_std)
        diagnostics["core_std"].append(core_std)
        diagnostics["effective_scale_ratio"].append(current_weight * core_std / ada_std)
        diagnostics["rank_correlation"].append(rank_correlation(candidate_ada, core_scores))

        target_matches = candidate_indices.eq(target_indices.unsqueeze(1))
        covered = target_matches.any(dim=1)
        diagnostic_counts["queries"] += batch_size
        diagnostic_counts["covered"] += covered.sum().item()
        if covered.any():
            rows = torch.arange(batch_size)[covered]
            positions = target_matches[covered].float().argmax(dim=1)
            ada_covered = candidate_ada[covered]
            core_covered = core_scores[covered]
            ref_covered = reference_indices[covered]
            candidate_covered = candidate_indices[covered]
            invalid = candidate_covered.eq(ref_covered.unsqueeze(1))
            positive_mask = torch.zeros_like(invalid)
            positive_mask[torch.arange(len(rows)), positions] = True
            negative_mask = ~(invalid | positive_mask)

            positive_ada = ada_covered[torch.arange(len(rows)), positions]
            positive_core = core_covered[torch.arange(len(rows)), positions]
            hard_ada = ada_covered.masked_fill(~negative_mask, -torch.inf).max(dim=1).values
            hard_core = core_covered.masked_fill(~negative_mask, -torch.inf).max(dim=1).values
            ada_margin = positive_ada - hard_ada
            core_margin = positive_core - hard_core
            diagnostics["ada_positive_margin"].append(ada_margin)
            diagnostics["core_positive_margin"].append(core_margin)

            ada_valid = ada_covered.masked_fill(invalid, -torch.inf)
            core_valid = core_covered.masked_fill(invalid, -torch.inf)
            ada_correct = ada_valid.argmax(dim=1).eq(positions)
            core_correct = core_valid.argmax(dim=1).eq(positions)
            diagnostic_counts["conflict"] += (ada_margin * core_margin < 0).sum().item()
            diagnostic_counts["correction"] += ((~ada_correct) & core_correct).sum().item()
            diagnostic_counts["damage"] += (ada_correct & (~core_correct)).sum().item()
            diagnostic_counts["covered_count"] += covered.sum().item()

        processed_queries += batch_size
        progress.set_postfix(queries=processed_queries)
        del global_scores, candidate_indices, candidate_ada, core_scores

    metrics = {name: finalize_recall(counts) for name, counts in recall_counts.items()}
    raw_grid = {str(weight): finalize_recall(counts) for weight, counts in raw_grid_counts.items()}
    zscore_grid = {
        str(weight): finalize_recall(counts)
        for weight, counts in zscore_grid_counts.items()
    }
    covered_count = max(int(diagnostic_counts["covered_count"]), 1)
    result = {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "temperature": float(latent_model.temp.detach().cpu().item()),
        "queries": processed_queries,
        "topk_target_coverage": diagnostic_counts["covered"]
        / max(diagnostic_counts["queries"], 1),
        "score_diagnostics": {
            "ada_std": summarize(diagnostics["ada_std"]),
            "core_std": summarize(diagnostics["core_std"]),
            "effective_scale_ratio": summarize(diagnostics["effective_scale_ratio"]),
            "rank_correlation": summarize(diagnostics["rank_correlation"]),
            "ada_positive_margin": summarize(diagnostics["ada_positive_margin"]),
            "core_positive_margin": summarize(diagnostics["core_positive_margin"]),
            "margin_conflict_rate": diagnostic_counts["conflict"] / covered_count,
            "correction_rate": diagnostic_counts["correction"] / covered_count,
            "damage_rate": diagnostic_counts["damage"] / covered_count,
        },
        "metrics": metrics,
        "fusion_weight_grid": {
            "raw": raw_grid,
            "zscore": zscore_grid,
        },
    }
    result["interpretation"] = interpret(result)

    print("\n" + "=" * 80)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 80)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(f"Saved diagnostic report to: {output}")


if __name__ == "__main__":
    main()
