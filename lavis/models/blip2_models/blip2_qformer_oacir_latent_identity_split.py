"""True identity multi-positive loss + text-free identity target mask.

The original latent backbone and composition route are retained. Training
requires real target_identity_ids derived from annotation paths. No inferred
identity labels, hardest-negative margin, gate, or extra identity encoder is
used. Identity-only extra candidates never change the exact-image CE pool.
"""

from pathlib import PurePosixPath

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2_qformer_oacir_latent import (
    Blip2QformerOacirLatent,
)
from lavis.models.blip2_models.oacir_latent_matching import (
    compute_bidirectional_region_scores,
)


def identity_key_from_path(relative_path):
    """Use the annotated identity directory, including its domain namespace."""
    path = PurePosixPath(str(relative_path).replace("\\", "/"))
    if len(path.parts) < 2 or ".." in path.parts:
        raise ValueError(f"Invalid annotated identity path: {relative_path!r}")
    return path.parent.as_posix()


def multi_positive_identity_loss(logits, positive_mask, valid_mask=None):
    """-log(sum(exp(positive logits)) / sum(exp(valid logits))).

    Every same-identity image is a positive; only other identities are
    negatives. Padding is excluded from both numerator and denominator.
    Accumulate in float32 under AMP for numerical stability.
    """
    if logits.ndim != 2 or positive_mask.shape != logits.shape:
        raise ValueError("Expected equally shaped [B, C] logits and positives")
    if valid_mask is None:
        valid_mask = torch.ones_like(positive_mask, dtype=torch.bool)
    if valid_mask.shape != logits.shape:
        raise ValueError("valid_mask must match identity logits")
    valid_mask = valid_mask.to(device=logits.device, dtype=torch.bool)
    positive_mask = positive_mask.to(device=logits.device, dtype=torch.bool)
    if (positive_mask & ~valid_mask).any():
        raise ValueError("An identity positive cannot be a padding candidate")
    if not positive_mask.any(dim=1).all():
        raise ValueError("Every identity query must have a real positive")
    stable_logits = logits.float()
    denominator = torch.logsumexp(
        stable_logits.masked_fill(~valid_mask, -torch.inf), dim=1
    )
    numerator = torch.logsumexp(
        stable_logits.masked_fill(~positive_mask, -torch.inf), dim=1
    )
    return (denominator - numerator).mean()


def expand_identity_candidates(candidate_indices, identity_ids):
    """Append all missing same-ID batch targets without displacing Top-K.

    The original fixed Top-K occupies the first K columns. Additional
    columns are for identity supervision only. A validity mask excludes
    padded indices from the loss, including any padded aligned target.
    """
    batch_size, topk = candidate_indices.shape
    if identity_ids.shape != (batch_size,):
        raise ValueError("target_identity_ids must have shape [batch_size]")
    rows = []
    for query_index in range(batch_size):
        original = candidate_indices[query_index]
        if original.unique().numel() != topk:
            raise ValueError("Original Top-K must contain unique batch indices")
        if not original.eq(query_index).any():
            raise ValueError("Original Top-K must contain the exact target")
        same_id = torch.where(identity_ids.eq(identity_ids[query_index]))[0]
        already_present = same_id[:, None].eq(original[None, :]).any(dim=1)
        rows.append(torch.cat([original, same_id[~already_present]]))
    width = max(row.numel() for row in rows)
    padded = torch.arange(
        batch_size, device=candidate_indices.device, dtype=candidate_indices.dtype
    )[:, None].expand(batch_size, width).clone()
    valid = torch.zeros_like(padded, dtype=torch.bool)
    for query_index, row in enumerate(rows):
        padded[query_index, :row.numel()] = row
        valid[query_index, :row.numel()] = True
    positives = identity_ids[padded].eq(identity_ids[:, None]) & valid
    return padded, positives, valid


def compute_identity_split_scores(
    reference_tokens, candidate_tokens, composed_features, region_sizes,
    spatial_kernel_size=3, spatial_weight=0.15, region_temperature=0.07,
    cycle_weight=0.25, text_weight=0.25, return_heatmap=False,
):
    """Discover M_comp normally and M_id with no text input or text gradient."""
    common = dict(
        spatial_kernel_size=spatial_kernel_size,
        spatial_weight=spatial_weight,
        region_temperature=region_temperature,
        cycle_weight=cycle_weight,
        return_heatmap=return_heatmap,
    )
    composition = compute_bidirectional_region_scores(
        reference_tokens, candidate_tokens, composed_features, region_sizes,
        text_weight=text_weight, **common,
    )
    identity = compute_bidirectional_region_scores(
        reference_tokens, candidate_tokens, torch.zeros_like(composed_features),
        region_sizes, text_weight=0.0, **common,
    )
    result = {
        "score_id": identity["score_id"],
        "score_comp": composition["score_comp"],
        "score_background": composition["score_background"],
        "cycle_ratio": identity["cycle_ratio"],
        "entropy": identity["entropy"],
    }
    if return_heatmap:
        result["target_attention"] = composition["target_attention"]
        result["target_identity_attention"] = identity["target_attention"]
    return result


@registry.register_model("oacir_latent_identity_split")
class Blip2QformerOacirLatentIdentitySplit(Blip2QformerOacirLatent):
    """Combine actual identity positives with independent M_id and M_comp."""

    requires_identity_labels = True

    def _compute_core_matcher_candidate_scores(
        self, reference_image_embeds_raw, target_image_embeds_raw,
        reference_bbox, text_features, candidate_indices=None, return_heatmap=False,
    ):
        reference = self._compute_core_reference_features(
            reference_image_embeds_raw, reference_bbox, text_features
        )
        num_queries = reference_image_embeds_raw.size(0)
        if target_image_embeds_raw.dim() == 3:
            if candidate_indices is None or candidate_indices.size(0) != num_queries:
                raise ValueError("Shared target bank requires per-query indices")
            target_tokens = F.normalize(
                self._project_raw_patch_tokens(target_image_embeds_raw), dim=-1
            )
        elif target_image_embeds_raw.dim() == 4:
            if candidate_indices is not None:
                raise ValueError("Gathered candidates must not receive indices")
            if target_image_embeds_raw.size(0) != num_queries:
                raise ValueError("Gathered candidates must match query batch size")
            target_tokens = None
        else:
            raise ValueError("Target embeddings must be [T,N,D] or [B,C,N,D]")

        keys = ("score_id", "score_comp", "score_background", "cycle_ratio", "entropy")
        collected = {key: [] for key in keys}
        comp_heatmaps, id_heatmaps = [], []

        def match(regions, candidates, composed, sizes, heatmap=False):
            return compute_identity_split_scores(
                regions, candidates, composed, sizes,
                spatial_kernel_size=self.region_spatial_kernel,
                spatial_weight=self.region_spatial_weight,
                region_temperature=self.region_temperature,
                cycle_weight=self.core_matcher_cycle_weight,
                text_weight=self.core_matcher_text_weight,
                return_heatmap=heatmap,
            )

        def match_shared(regions, bank, composed, sizes, indices):
            matched = match(regions, bank[indices], composed, sizes)
            return tuple(matched[key] for key in keys)

        chunk_size = max(int(self.core_matcher_query_chunk_size), 1)
        for start in range(0, num_queries, chunk_size):
            end = min(start + chunk_size, num_queries)
            regions = reference["region_tokens"][start:end]
            composed = reference["composed_feature"][start:end]
            sizes = reference["region_sizes"][start:end]
            if target_tokens is not None:
                indices = candidate_indices[start:end]
                if self.training and torch.is_grad_enabled() and not return_heatmap:
                    values = checkpoint(
                        match_shared, regions, target_tokens, composed, sizes, indices,
                        use_reentrant=False, preserve_rng_state=False,
                    )
                    result = dict(zip(keys, values))
                else:
                    result = match(
                        regions, target_tokens[indices], composed, sizes, return_heatmap
                    )
            else:
                raw = target_image_embeds_raw[start:end].to(
                    reference_image_embeds_raw.device, non_blocking=True
                )
                shape = raw.shape
                candidates = self._project_raw_patch_tokens(
                    raw.reshape(-1, shape[-2], shape[-1])
                )
                candidates = F.normalize(candidates, dim=-1).reshape(
                    shape[0], shape[1], shape[-2] - 1, -1
                )
                result = match(regions, candidates, composed, sizes, return_heatmap)
            for key in keys:
                collected[key].append(result[key])
            if return_heatmap:
                comp_heatmaps.append(result["target_attention"])
                id_heatmaps.append(result["target_identity_attention"])

        outputs = {key: torch.cat(values, dim=0) for key, values in collected.items()}
        outputs["score_final"] = (
            outputs["score_comp"] + float(self.core_matcher_lambda_id) * outputs["score_id"]
        )
        outputs["reference_attention"] = reference["reference_attention"]
        outputs["fusion_scalar"] = reference["fusion_scalar"]
        if return_heatmap:
            outputs["target_attention"] = torch.cat(comp_heatmaps, dim=0)
            outputs["target_identity_attention"] = torch.cat(id_heatmaps, dim=0)
        return outputs

    def forward(self, samples):
        if "target_identity_ids" not in samples:
            raise ValueError(
                "oacir_latent_identity_split requires REAL target_identity_ids "
                "from annotation paths; the training entry must pass them"
            )
        reference_bbox = samples.get("reference_bbox")
        if reference_bbox is None:
            raise ValueError("oacir_latent_identity_split requires reference_bbox")
        reference_raw, target_raw, reference_embeds, target_embeds, batch_size = (
            self._encode_images_from_samples(samples)
        )
        identity_ids = torch.as_tensor(
            samples["target_identity_ids"], dtype=torch.long,
            device=reference_embeds.device,
        )
        if identity_ids.shape != (batch_size,) or (identity_ids < 0).any():
            raise ValueError("Expected one nonnegative real identity ID per target")
        modification_text = samples["modification_text"]
        fusion = self._compute_fusion_features(
            reference_embeds, modification_text, reference_bbox
        )
        targets_features = self._compute_target_features_from_embeds(target_embeds)
        text_features = self._compute_text_features(
            modification_text, reference_embeds.device
        )
        backbone_logits = self._compute_composition_scores(
            fusion, targets_features
        ) / self.temp
        targets = torch.arange(batch_size, device=reference_embeds.device)
        exact_candidates = self._build_candidate_indices(
            backbone_logits, topk=self.core_matcher_topk, force_positive=True
        )
        all_candidates, identity_positives, valid = expand_identity_candidates(
            exact_candidates, identity_ids
        )
        scores = self._compute_core_matcher_candidate_scores(
            reference_raw, target_raw, reference_bbox, text_features,
            all_candidates, return_heatmap=False,
        )

        # Exact-image losses retain the original Top-K and their single target.
        topk = exact_candidates.size(1)
        candidate_targets = exact_candidates.eq(targets[:, None]).long().argmax(1)
        final_logits = scores["score_final"][:, :topk] / self.core_matcher_temp
        composition_logits = scores["score_comp"][:, :topk] / self.core_matcher_temp
        reranking_logits = backbone_logits.gather(1, exact_candidates) + (
            float(self.core_matcher_fusion_weight) * final_logits
        )
        positive_comp = scores["score_comp"][:, :topk].gather(
            1, candidate_targets[:, None]
        ).squeeze(1)
        positive_background = scores["score_background"][:, :topk].gather(
            1, candidate_targets[:, None]
        ).squeeze(1)

        with torch.no_grad():
            counts = identity_positives.sum(dim=1)
            self.identity_training_stats = {
                "mean_positive_candidates": counts.float().mean().item(),
                "multi_positive_query_fraction": counts.gt(1).float().mean().item(),
                "mean_valid_candidates": valid.sum(1).float().mean().item(),
                "original_topk": topk,
            }
        return {
            "loss_align": F.cross_entropy(backbone_logits, targets),
            "loss_core_matcher": F.cross_entropy(reranking_logits, candidate_targets),
            "loss_core_comp": F.cross_entropy(composition_logits, candidate_targets),
            "loss_core_id": multi_positive_identity_loss(
                scores["score_id"] / self.core_matcher_temp, identity_positives, valid
            ),
            "loss_core_region": (2.0 - positive_comp + positive_background).mean(),
        }

    @torch.no_grad()
    def inference_core_matcher_candidates(
        self, reference_image_embeds_raw, target_candidate_embeds_raw,
        composition_candidate_logits, modification_text, reference_bbox,
        return_parts=False, return_heatmap=False,
    ):
        device = self._get_device()
        reference_raw = reference_image_embeds_raw.to(device, non_blocking=True)
        text_features = self._compute_text_features(modification_text, device)
        scores = self._compute_core_matcher_candidate_scores(
            reference_raw, target_candidate_embeds_raw, reference_bbox,
            text_features, return_heatmap=return_heatmap,
        )
        core_logits = scores["score_final"] / self.core_matcher_temp
        backbone = composition_candidate_logits.to(
            device=device, dtype=core_logits.dtype, non_blocking=True
        )
        final = backbone + float(self.core_matcher_fusion_weight) * core_logits
        if not return_parts:
            return final
        outputs = {
            "sim_final": final,
            "sim_core_matcher": core_logits,
            "sim_id": scores["score_id"] / self.core_matcher_temp,
            "sim_region_comp": scores["score_comp"] / self.core_matcher_temp,
            "fusion_scalar": scores["fusion_scalar"],
            "cycle_ratio": scores["cycle_ratio"],
            "region_entropy": scores["entropy"],
        }
        if return_heatmap:
            outputs["target_patch_heatmap"] = scores["target_attention"]
            outputs["target_identity_heatmap"] = scores["target_identity_attention"]
            outputs["reference_patch_attention"] = scores["reference_attention"]
        return outputs
