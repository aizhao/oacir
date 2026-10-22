"""Ablation 2: independent composition and identity target masks.

``M_comp`` keeps the original visual-text response. ``M_id`` is discovered by
identity similarity plus cycle consistency only, so modification text cannot
move the identity region.  No other model or loss change is introduced.
"""

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


def compute_split_mask_scores(
    composition_reference_tokens,
    identity_reference_tokens,
    target_candidate_tokens,
    composed_features,
    region_sizes,
    spatial_kernel_size,
    spatial_weight,
    region_temperature,
    cycle_weight,
    text_weight,
    return_heatmap=False,
):
    """Run target-region discovery twice to obtain M_comp and M_id."""
    composition = compute_bidirectional_region_scores(
        composition_reference_tokens,
        target_candidate_tokens,
        composed_features,
        region_sizes,
        identity_reference_region_tokens=composition_reference_tokens,
        spatial_kernel_size=spatial_kernel_size,
        spatial_weight=spatial_weight,
        region_temperature=region_temperature,
        cycle_weight=cycle_weight,
        text_weight=text_weight,
        return_heatmap=return_heatmap,
    )
    identity = compute_bidirectional_region_scores(
        identity_reference_tokens,
        target_candidate_tokens,
        composed_features,
        region_sizes,
        identity_reference_region_tokens=identity_reference_tokens,
        spatial_kernel_size=spatial_kernel_size,
        spatial_weight=spatial_weight,
        region_temperature=region_temperature,
        cycle_weight=cycle_weight,
        text_weight=0.0,
        return_heatmap=return_heatmap,
    )

    outputs = {
        "score_id": identity["score_id"],
        "score_comp": composition["score_comp"],
        "score_background": composition["score_background"],
        "cycle_ratio": identity["cycle_ratio"],
        "entropy": identity["entropy"],
    }
    if return_heatmap:
        outputs["target_attention"] = composition["target_attention"]
        outputs["target_identity_attention"] = identity["target_attention"]
    return outputs


@registry.register_model("oacir_latent_split_mask")
class Blip2QformerOacirLatentSplitMask(Blip2QformerOacirLatent):
    """Isolate text-free M_id from the original composition mask M_comp."""

    def _compute_core_matcher_candidate_scores(
        self,
        reference_image_embeds_raw,
        target_image_embeds_raw,
        reference_bbox,
        text_features,
        candidate_indices=None,
        return_heatmap=False,
    ):
        core_reference = self._compute_core_reference_features(
            reference_image_embeds_raw, reference_bbox, text_features
        )
        if target_image_embeds_raw.dim() == 3:
            if candidate_indices is None:
                raise ValueError("candidate_indices are required for a shared bank")
            target_tokens = F.normalize(
                self._project_raw_patch_tokens(target_image_embeds_raw), dim=-1
            )
            num_queries = candidate_indices.size(0)
        elif target_image_embeds_raw.dim() == 4:
            if candidate_indices is not None:
                raise ValueError("candidate_indices must be omitted when gathered")
            target_tokens = None
            num_queries = target_image_embeds_raw.size(0)
        else:
            raise ValueError("target embeddings must be [T,N,D] or [B,C,N,D]")

        output_keys = (
            "score_id",
            "score_comp",
            "score_background",
            "cycle_ratio",
            "entropy",
        )
        collected = {key: [] for key in output_keys}
        composition_heatmaps = []
        identity_heatmaps = []
        chunk_size = max(int(self.core_matcher_query_chunk_size), 1)
        identity_reference = core_reference.get(
            "identity_region_tokens", core_reference["region_tokens"]
        )

        def run_split(comp_regions, id_regions, candidates, composed, sizes):
            result = compute_split_mask_scores(
                comp_regions,
                id_regions,
                candidates,
                composed,
                sizes,
                spatial_kernel_size=self.region_spatial_kernel,
                spatial_weight=self.region_spatial_weight,
                region_temperature=self.region_temperature,
                cycle_weight=self.core_matcher_cycle_weight,
                text_weight=self.core_matcher_text_weight,
                return_heatmap=False,
            )
            return tuple(result[key] for key in output_keys)

        for start in range(0, num_queries, chunk_size):
            end = min(start + chunk_size, num_queries)
            comp_regions = core_reference["region_tokens"][start:end]
            id_regions = identity_reference[start:end]
            composed = core_reference["composed_feature"][start:end]
            sizes = core_reference["region_sizes"][start:end]

            if target_tokens is not None:
                current_indices = candidate_indices[start:end]

                def run_shared_bank(
                    comp_r, id_r, all_targets, comp_f, current_sizes, indices
                ):
                    return run_split(
                        comp_r, id_r, all_targets[indices], comp_f, current_sizes
                    )

                if self.training and torch.is_grad_enabled() and not return_heatmap:
                    values = checkpoint(
                        run_shared_bank,
                        comp_regions,
                        id_regions,
                        target_tokens,
                        composed,
                        sizes,
                        current_indices,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                    result = dict(zip(output_keys, values))
                else:
                    candidate_tokens = target_tokens[current_indices]
                    result = compute_split_mask_scores(
                        comp_regions,
                        id_regions,
                        candidate_tokens,
                        composed,
                        sizes,
                        spatial_kernel_size=self.region_spatial_kernel,
                        spatial_weight=self.region_spatial_weight,
                        region_temperature=self.region_temperature,
                        cycle_weight=self.core_matcher_cycle_weight,
                        text_weight=self.core_matcher_text_weight,
                        return_heatmap=return_heatmap,
                    )
            else:
                raw_candidates = target_image_embeds_raw[start:end].to(
                    reference_image_embeds_raw.device, non_blocking=True
                )
                raw_shape = raw_candidates.shape
                candidate_tokens = self._project_raw_patch_tokens(
                    raw_candidates.reshape(-1, raw_shape[-2], raw_shape[-1])
                )
                candidate_tokens = F.normalize(candidate_tokens, dim=-1).reshape(
                    raw_shape[0], raw_shape[1], raw_shape[-2] - 1, -1
                )
                result = compute_split_mask_scores(
                    comp_regions,
                    id_regions,
                    candidate_tokens,
                    composed,
                    sizes,
                    spatial_kernel_size=self.region_spatial_kernel,
                    spatial_weight=self.region_spatial_weight,
                    region_temperature=self.region_temperature,
                    cycle_weight=self.core_matcher_cycle_weight,
                    text_weight=self.core_matcher_text_weight,
                    return_heatmap=return_heatmap,
                )

            for key in output_keys:
                collected[key].append(result[key])
            if return_heatmap:
                composition_heatmaps.append(result["target_attention"])
                identity_heatmaps.append(result["target_identity_attention"])

        outputs = {key: torch.cat(values) for key, values in collected.items()}
        outputs["score_final"] = (
            outputs["score_comp"]
            + float(self.core_matcher_lambda_id) * outputs["score_id"]
        )
        outputs["reference_attention"] = core_reference["reference_attention"]
        outputs["fusion_scalar"] = core_reference["fusion_scalar"]
        if return_heatmap:
            outputs["target_attention"] = torch.cat(composition_heatmaps)
            outputs["target_identity_attention"] = torch.cat(identity_heatmaps)
        return outputs

