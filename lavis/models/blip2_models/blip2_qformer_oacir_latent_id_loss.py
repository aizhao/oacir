"""Ablation 1: original OACIR latent model with a top-1 identity loss.

Only ``loss_core_id`` is changed.  Everything else, including the latent
architecture, target-region discovery and inference, is inherited unchanged.
"""

import torch
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2_qformer_oacir_latent import (
    Blip2QformerOacirLatent,
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


@registry.register_model("oacir_latent_id_loss")
class Blip2QformerOacirLatentIdLoss(Blip2QformerOacirLatent):
    """Isolate the effect of a stronger top-1 identity objective."""

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
            raise ValueError("oacir_latent_id_loss requires reference_bbox")

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

