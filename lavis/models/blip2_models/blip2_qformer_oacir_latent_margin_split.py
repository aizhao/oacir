"""Single-positive CE + hardest-negative margin with M_id/M_comp separation.

Independent ablation of the original latent model. The identity objective is
CE + 0.50 * softplus(hardest_negative_logit - positive_logit + 0.20).
The margin is in temperature-scaled logit units. Candidate Top-K, exact-image
composition supervision, and latent parameter initialization are unchanged.
No real-identity labels, candidate expansion, gate or new encoder are used.
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


def top1_identity_loss(logits, targets, margin=0.20, margin_weight=0.50):
    """CE plus a hardest-negative margin aimed directly at identity R@1."""
    ce_loss = F.cross_entropy(logits, targets)
    positive = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    negative_mask = torch.ones_like(logits, dtype=torch.bool)
    negative_mask.scatter_(1, targets.unsqueeze(1), False)
    hardest_negative = logits.masked_fill(
        ~negative_mask, torch.finfo(logits.dtype).min
    ).max(dim=1).values
    margin_loss = F.softplus(hardest_negative - positive + float(margin)).mean()
    return ce_loss + float(margin_weight) * margin_loss


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
        torch.zeros_like(composed_features),
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


@registry.register_model("oacir_latent_margin_split")
class Blip2QformerOacirLatentMarginSplit(Blip2QformerOacirLatent):
    """Combine the earlier single-positive margin loss and two target masks."""

    requires_identity_labels = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.identity_loss_margin = 0.20
        self.identity_margin_weight = 0.50

    def set_identity_loss_config(self, margin=None, margin_weight=None):
        if margin is not None:
            self.identity_loss_margin = float(margin)
        if margin_weight is not None:
            self.identity_margin_weight = float(margin_weight)

    def forward(self, samples):
        modification_text = samples["modification_text"]
        reference_bbox = samples.get("reference_bbox")
        if reference_bbox is None:
            raise ValueError("oacir_latent_margin_split requires reference_bbox")

        (
            reference_raw,
            target_raw,
            reference_embeds,
            target_embeds,
            batch_size,
        ) = self._encode_images_from_samples(samples)

        fusion_features = self._compute_fusion_features(
            reference_embeds, modification_text, reference_bbox
        )
        target_features = self._compute_target_features_from_embeds(target_embeds)
        text_features = self._compute_text_features(
            modification_text, reference_embeds.device
        )
        sim_comp_logits = self._compute_composition_scores(
            fusion_features, target_features
        ) / self.temp

        targets = torch.arange(
            batch_size, dtype=torch.long, device=reference_embeds.device
        )
        candidate_indices = self._build_candidate_indices(
            sim_comp_logits, topk=self.core_matcher_topk, force_positive=True
        )
        core_scores = self._compute_core_matcher_candidate_scores(
            reference_raw,
            target_raw,
            reference_bbox,
            text_features,
            candidate_indices,
            return_heatmap=False,
        )
        candidate_targets = candidate_indices.eq(targets.unsqueeze(1)).long().argmax(1)

        final_logits = core_scores["score_final"] / self.core_matcher_temp
        identity_logits = core_scores["score_id"] / self.core_matcher_temp
        composition_logits = core_scores["score_comp"] / self.core_matcher_temp
        backbone_candidate_logits = torch.gather(
            sim_comp_logits, dim=1, index=candidate_indices
        )
        reranking_logits = (
            backbone_candidate_logits
            + float(self.core_matcher_fusion_weight) * final_logits
        )

        positive_composition = torch.gather(
            core_scores["score_comp"], 1, candidate_targets.unsqueeze(1)
        ).squeeze(1)
        positive_background = torch.gather(
            core_scores["score_background"], 1, candidate_targets.unsqueeze(1)
        ).squeeze(1)

        return {
            "loss_align": F.cross_entropy(sim_comp_logits, targets),
            "loss_core_matcher": F.cross_entropy(
                reranking_logits, candidate_targets
            ),
            "loss_core_comp": F.cross_entropy(
                composition_logits, candidate_targets
            ),
            "loss_core_id": top1_identity_loss(
                identity_logits,
                candidate_targets,
                margin=self.identity_loss_margin,
                margin_weight=self.identity_margin_weight,
            ),
            "loss_core_region": (
                2.0 - positive_composition + positive_background
            ).mean(),
        }

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

