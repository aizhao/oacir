"""Training objectives for the existing OACIR Q-Former Identity model.

Install this file in lavis/models/blip2_models/ alongside the existing
blip2_qformer_oacir_qformer_identity.py. Import it before requesting the
registry name oacir_qformer_identity_conditional. Model parameters and
full-gallery scoring are inherited from the base class.

Training batches use the base model's image/raw-feature, text and bbox
inputs plus two integer [B] vectors:

    target_identity_ids   IDs encoded from (domain, annotated instance ID)
    target_image_ids      IDs unique per actual target image across domains

Optional composition_relevance is an integer [B, B] matrix in the same batch
order. Row i judges query i; column j is target image j. Use 1 for a verified
match, 0 for a verified mismatch and -1 for unknown. The paired image (and
any repeated copy of it) is positive by construction. When the same target
image appears twice, a known judgment applies to both columns.
With judgments supplied, the composition CE ignores unknown same-ID
alternatives. The final exact-image CE still ranks the paired image against
unknown alternatives, while excluding verified matches from its negatives.

Use the same batches, relevance labels and external loss weights for all
three training controls. Each setting is a separate run:

    model.set_single_stage_config(
        identity_loss_mode="single_positive", enable_text_ranking=False)
    model.set_single_stage_config(
        identity_loss_mode="multi_positive", enable_text_ranking=False)
    model.set_single_stage_config(
        identity_loss_mode="multi_positive", enable_text_ranking=True)

The default is the second setting. The third requires composition_relevance
in every training batch. A batch without verified same-ID mismatches yields
zero text-ranking loss. Inspect model.last_loss_diagnostics for coverage.
Use an ID-grouped sampler if ordinary shuffling rarely puts distinct images
of the same identity in a batch. Include multiple identities per batch and
keep that sampler fixed across controls.

The four loss keys match the V2 training loss dictionary. Its launcher
explicitly instantiates V2, so a separate caller must import/instantiate
this class and pass the relevance matrix to enable text ranking. The model
does not derive judgments from captions, filenames, or similarity scores.
Inference uses the inherited methods and does not need any training labels.
"""

import math
import warnings

import torch
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2_qformer_oacir_qformer_identity import (
    Blip2QformerOacirQformerIdentity,
)


MODEL_NAME = "oacir_qformer_identity_conditional"
INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
IDENTITY_LOSS_MODES = ("single_positive", "multi_positive")
OBJECTIVE_SETTINGS = (
    "identity_loss_mode",
    "enable_text_ranking",
    "composition_margin",
    "composition_margin_temperature",
)


def _checked_ids(value, batch_size, device, name):
    ids = torch.as_tensor(value, device=device)
    if ids.shape != (batch_size,) or ids.dtype not in INTEGER_DTYPES:
        raise ValueError(f"{name} must be an integer [{batch_size}] vector")
    if (ids < 0).any():
        raise ValueError(f"{name} must contain nonnegative annotation IDs")
    return ids.long()


def _check_settings(identity_loss_mode, enable_text_ranking, margin, temperature):
    if identity_loss_mode not in IDENTITY_LOSS_MODES:
        raise ValueError(f"identity_loss_mode must be one of {IDENTITY_LOSS_MODES}")
    if not isinstance(enable_text_ranking, bool):
        raise ValueError("enable_text_ranking must be a boolean")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("composition_margin must be finite and nonnegative")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("composition_margin_temperature must be finite and positive")


def _relevance_masks(relevance, same_identity, same_image, image_inverse, image_count):
    """Return verified matches and verified same-identity mismatches."""
    if relevance is None:
        return same_image, torch.zeros_like(same_image)

    labels = torch.as_tensor(relevance, device=same_image.device)
    if labels.shape != same_image.shape or labels.dtype not in INTEGER_DTYPES:
        raise ValueError("composition_relevance must be an integer [B, B] matrix")
    labels = labels.long()
    if ((labels < -1) | (labels > 1)).any():
        raise ValueError("composition_relevance values must be -1, 0 or 1")

    # Merge judgments by actual image, including when one duplicate is unknown.
    group_index = image_inverse.unsqueeze(0).expand_as(labels)
    counts = labels.new_zeros((len(labels), image_count), dtype=torch.long)
    positive_counts = counts.scatter_add(1, group_index, labels.eq(1).long())
    negative_counts = counts.scatter_add(1, group_index, labels.eq(0).long())
    if ((positive_counts > 0) & (negative_counts > 0)).any():
        raise ValueError("Conflicting composition judgments for the same query and image")
    positive = positive_counts.gather(1, group_index).gt(0)
    negative = negative_counts.gather(1, group_index).gt(0)
    if (negative & same_image).any():
        raise ValueError("The paired target image cannot be labeled a mismatch")
    if (positive & ~same_identity).any():
        raise ValueError("A verified composition match must share the query's identity")
    return positive | same_image, negative & same_identity


def _same_identity_text_ranking(
    cosines, positive, negative, inverse_frequency, margin, temperature
):
    """Rank every verified positive above the hardest verified mismatch."""
    has_negative = negative.any(dim=1)
    hardest = cosines.masked_fill(~negative, -torch.inf).amax(dim=1)
    hardest = torch.where(has_negative, hardest, cosines.diagonal())
    violations = F.relu((hardest[:, None] + margin - cosines) / temperature)
    weights = (positive & has_negative[:, None]) * inverse_frequency
    per_query = (violations * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    return per_query.sum() / has_negative.sum().clamp_min(1)


def training_objectives(
    scores,
    target_identity_ids,
    target_image_ids,
    *,
    identity_loss_mode="multi_positive",
    enable_text_ranking=False,
    composition_relevance=None,
    composition_cosines=None,
    margin=0.05,
    margin_temperature=0.07,
):
    """Compute training losses from aligned score matrices and annotations."""
    _check_settings(identity_loss_mode, enable_text_ranking, margin, margin_temperature)
    identity_logits = scores["identity"].float()
    batch_size = len(identity_logits)
    shape = (batch_size, batch_size)
    if batch_size < 2 or any(scores[key].shape != shape
                            for key in ("identity", "composition", "final")):
        raise ValueError("Training expects square, aligned score matrices with B >= 2")
    if enable_text_ranking and composition_relevance is None:
        raise ValueError("Text ranking requires composition_relevance judgments")
    if enable_text_ranking and (
        composition_cosines is None or composition_cosines.shape != shape
    ):
        raise ValueError("Text ranking requires raw composition cosines of shape [B, B]")

    device = identity_logits.device
    ids = _checked_ids(target_identity_ids, batch_size, device, "target_identity_ids")
    images = _checked_ids(target_image_ids, batch_size, device, "target_image_ids")
    same_identity = ids[:, None].eq(ids[None, :])
    same_image = images[:, None].eq(images[None, :])
    if (same_image & ~same_identity).any():
        raise ValueError("An annotated target image cannot have multiple identity IDs")
    _, image_inverse, image_counts = torch.unique(
        images, return_inverse=True, return_counts=True
    )
    inverse_frequency = image_counts[image_inverse].float().reciprocal()
    positive, text_negative = _relevance_masks(
        composition_relevance, same_identity, same_image, image_inverse, len(image_counts)
    )
    labels = torch.arange(batch_size, device=device)
    diagonal = torch.eye(batch_size, dtype=torch.bool, device=device)

    # Preserve exact-image CE and omit known valid alternatives from its negatives.
    exact_allowed = ~positive | diagonal
    composition_allowed = exact_allowed
    if composition_relevance is not None:
        composition_allowed = exact_allowed & (~same_identity | text_negative | diagonal)
    if identity_loss_mode == "multi_positive":
        # Each distinct candidate image contributes total weight one. This
        # prevents repeat sampling from increasing its identity probability.
        logits = identity_logits + inverse_frequency.log()
        positive_mass = logits.masked_fill(~same_identity, -torch.inf).logsumexp(dim=1)
        identity_loss = (logits.logsumexp(dim=1) - positive_mass).mean()
    else:
        identity_loss = F.cross_entropy(
            identity_logits.masked_fill(same_image & ~diagonal, -torch.inf), labels
        )

    composition_logits = scores["composition"].float()
    text_ranking = composition_logits.diagonal().sum() * 0.0
    if enable_text_ranking:
        text_ranking = _same_identity_text_ranking(
            composition_cosines.float(), positive, text_negative,
            inverse_frequency, margin, margin_temperature,
        )
    losses = {
        "loss_retrieval": F.cross_entropy(
            scores["final"].float().masked_fill(~exact_allowed, -torch.inf), labels
        ),
        "loss_identity": identity_loss,
        "loss_composition": F.cross_entropy(
            composition_logits.masked_fill(~composition_allowed, -torch.inf), labels
        ),
        "loss_composition_hard": text_ranking,
    }
    names = (
        "multi_positive_identity_rows",
        "identity_negative_rows",
        "text_ranking_eligible_rows",
        "verified_alternative_positive_rows",
        "exact_duplicate_offdiag",
    )
    counts = torch.stack((
        (same_identity & ~same_image).any(dim=1).sum(),
        (~same_identity).any(dim=1).sum(),
        text_negative.any(dim=1).sum(),
        (positive & ~same_image).any(dim=1).sum(),
        (same_image & ~diagonal).sum(),
    ))
    diagnostics = dict(zip(names, counts.tolist()))
    diagnostics["batch_queries"] = batch_size
    diagnostics["text_ranking_active_rows"] = (
        diagnostics["text_ranking_eligible_rows"] if enable_text_ranking else 0
    )
    return losses, diagnostics


@registry.register_model(MODEL_NAME)
class Blip2QformerOacirQformerIdentityConditional(Blip2QformerOacirQformerIdentity):
    requires_identity_labels = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.missing_bbox_policy = "full-image"
        self.last_identity_bbox_stats = {"missing": 0, "total": 0}
        self._missing_conditional_bbox_warned = False
        self.identity_loss_mode = "multi_positive"
        self.enable_text_ranking = False
        self.composition_margin = 0.05
        self.composition_margin_temperature = 0.07
        self.last_loss_diagnostics = {}

    def set_single_stage_config(self, **values):
        bbox_policy = values.pop("missing_bbox_policy", self.missing_bbox_policy)
        if bbox_policy not in ("full-image", "error"):
            raise ValueError("missing_bbox_policy must be full-image or error")
        settings = {
            key: values.pop(key) if key in values else getattr(self, key)
            for key in OBJECTIVE_SETTINGS
        }
        _check_settings(
            settings["identity_loss_mode"], settings["enable_text_ranking"],
            settings["composition_margin"], settings["composition_margin_temperature"],
        )
        super().set_single_stage_config(**values)
        self.missing_bbox_policy = bbox_policy
        for key, value in settings.items():
            setattr(self, key, value)

    @classmethod
    def from_config(cls, cfg):
        model = super().from_config(cfg)
        keys = (*OBJECTIVE_SETTINGS, "missing_bbox_policy")
        model.set_single_stage_config(**{key: cfg[key] for key in keys if key in cfg})
        return model

    def encode_identity_reference(self, reference_raw, reference_bbox):
        """Keep the established missing-box fallback on older base revisions."""
        count = len(reference_raw)
        boxes = [None] * count if reference_bbox is None else reference_bbox
        if len(boxes) != count:
            raise ValueError("Expected one bbox entry per query")
        missing = sum(box is None for box in boxes)
        self.last_identity_bbox_stats = {"missing": missing, "total": count}
        if not missing:
            return super().encode_identity_reference(reference_raw, boxes)
        if self.missing_bbox_policy == "error":
            raise ValueError(f"Identity reference missing bbox for {missing}/{count} queries")
        for box in boxes:
            if box is None:
                continue
            coords = torch.as_tensor(box, dtype=torch.float32)
            if coords.shape != (4,) or not torch.isfinite(coords).all() or (coords[2:] <= coords[:2]).any():
                raise ValueError("Annotated reference bbox must be finite, nonempty xyxy")
        if not self._missing_conditional_bbox_warned:
            warnings.warn("Missing reference bbox: using all reference patches for that query", RuntimeWarning)
            self._missing_conditional_bbox_warned = True
        patches = self._project_raw_patch_tokens(reference_raw)
        mask = self._build_anchor_mask(boxes, reference_raw.shape[1], count, patches.device)
        regions, _, attention = self.core_reference_encoder(patches, mask, self.bbox_floor)
        minimum = min(self.region_topk, patches.shape[1])
        maximum = min(max(self.region_topk_max, minimum), patches.shape[1])
        sizes = (mask.sum(-1).float() * self.region_area_scale).round().long().clamp(minimum, maximum)
        return regions, sizes, attention

    def forward(self, samples):
        for key in ("target_identity_ids", "target_image_ids"):
            if key not in samples:
                raise ValueError(f"{MODEL_NAME} requires {key} during training")
        relevance = samples.get("composition_relevance")
        if self.enable_text_ranking and relevance is None:
            raise ValueError("Text ranking requires composition_relevance judgments")

        raw_ref, raw_target, ref_embeds, target_embeds, batch = self._encode_images_from_samples(samples)
        if len(raw_target) != batch or batch < 2:
            raise ValueError("Training requires an aligned batch with at least two pairs")
        query = self.encode_composition(ref_embeds, samples["modification_text"])
        regions, sizes, _ = self.encode_identity_reference(raw_ref, samples.get("reference_bbox"))
        target_features = self._compute_target_features_from_embeds(target_embeds)
        target_patches = F.normalize(self._project_raw_patch_tokens(raw_target), dim=-1)
        scores = self.score_all_pairs(query, regions, sizes, target_features, target_patches)

        # The fixed margin acts on raw cosines, independent of learned CE temperature.
        cosines = None
        if self.enable_text_ranking:
            cosines = self._compute_composition_scores(query, target_features)
        losses, self.last_loss_diagnostics = training_objectives(
            scores, samples["target_identity_ids"], samples["target_image_ids"],
            identity_loss_mode=self.identity_loss_mode,
            enable_text_ranking=self.enable_text_ranking,
            composition_relevance=relevance,
            composition_cosines=cosines,
            margin=self.composition_margin,
            margin_temperature=self.composition_margin_temperature,
        )
        return losses
