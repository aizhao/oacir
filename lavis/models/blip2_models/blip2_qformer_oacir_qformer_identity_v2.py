"""Independent combined-loss experiment; original single-stage model unchanged.

Inference architecture and state-dict parameter names are IDENTICAL to
oacir_qformer_identity: Q-Former composition + text-free local identity.
Only training objectives change. No new encoder, probes, candidate prefilter,
reranking, inferred identities, gallery labels at inference, or multi-positive
identity attraction. Same-ID off-diagonal entries are IGNORED by identity CE,
not promoted to positives. Exact target supervision is retained elsewhere.
"""
import math
import warnings
from pathlib import PurePosixPath

import torch
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2_qformer_oacir_qformer_identity import (
    Blip2QformerOacirQformerIdentity,
)


def identity_key_from_path(relative_path):
    """Preserve the entire annotation parent path, including domain namespace."""
    path = PurePosixPath(str(relative_path).replace('\\', '/'))
    if len(path.parts) < 2 or '..' in path.parts:
        raise ValueError(f'Invalid annotated identity path: {relative_path!r}')
    return path.parent.as_posix()


def checked_ids(value, batch, device, name):
    ids = torch.as_tensor(value, device=device)
    if ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(name + ' must contain integer annotation-derived labels')
    if tuple(ids.shape) != (batch,) or (ids < 0).any():
        raise ValueError(name + ' must be a nonnegative [batch] vector')
    return ids.long()


def masked_ce(logits, allowed):
    """One aligned positive, with specified columns excluded from denominator."""
    labels = torch.arange(len(logits), device=logits.device)
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1] or allowed.shape != logits.shape:
        raise ValueError('Training CE expects aligned square scores/mask')
    if not allowed.diagonal().all():
        raise ValueError('The aligned target must never be excluded')
    return F.cross_entropy(logits.float().masked_fill(~allowed, -torch.inf), labels)


def composition_hard_margin(cosines, same_identity, same_image, margin=.05, temperature=.07):
    """Same-ID/different-image hardest target, else global hardest target.

    Raw cosine margin, scaled by a FIXED temperature: a learned composition
    temperature cannot satisfy this auxiliary loss just by changing scale.
    Exact duplicates are never negatives. Rows without any valid negative
    have zero contribution; zero is graph-connected and finite.
    """
    if not math.isfinite(margin) or margin < 0 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Finite margin >=0 and fixed temperature >0 required')
    if cosines.ndim != 2 or cosines.shape[0] != cosines.shape[1]:
        raise ValueError('Expected square raw composition cosine matrix')
    if same_image.shape != cosines.shape or same_identity.shape != cosines.shape:
        raise ValueError('Pair labels must match score shape')
    global_negatives = ~same_image
    same_id_negatives = same_identity & global_negatives
    has_same_id = same_id_negatives.any(-1)
    valid = global_negatives.any(-1)
    selected = torch.where(has_same_id[:, None], same_id_negatives, global_negatives)
    raw = cosines.float()
    hardest = raw.masked_fill(~selected, -torch.inf).max(-1).values
    # Replace absent rows BEFORE relu/division; never create 0 * infinity.
    hardest = torch.where(valid, hardest, raw.diagonal())
    per_row = F.relu((hardest - raw.diagonal() + margin) / temperature)
    loss = (per_row * valid).sum() / valid.sum().clamp_min(1)
    diagnostics = {
        'same_id_hard_rows': int((has_same_id & valid).sum().detach()),
        'global_hard_rows': int((~has_same_id & valid).sum().detach()),
        'no_negative_rows': int((~valid).sum().detach()),
    }
    return loss, diagnostics


def training_objectives(scores, composition_cosines, identity_ids, image_ids,
                        margin=.05, margin_temperature=.07):
    """Annotation labels affect losses only, NEVER the feature/score functions."""
    batch = len(composition_cosines)
    device = composition_cosines.device
    ids = checked_ids(identity_ids, batch, device, 'target_identity_ids')
    images = checked_ids(image_ids, batch, device, 'target_image_ids')
    same_id = ids[:, None].eq(ids[None, :])
    same_image = images[:, None].eq(images[None, :])
    if (same_image & ~same_id).any():
        raise ValueError('The same annotated image cannot belong to different identities')
    diagonal = torch.eye(batch, device=device, dtype=torch.bool)
    # Ignore duplicate target images but KEEP same-ID different-image negatives
    # for exact-image composition and final retrieval.
    exact_allowed = ~same_image | diagonal
    identity_allowed = ~same_id | diagonal
    hard, diagnostics = composition_hard_margin(
        composition_cosines, same_id, same_image, margin, margin_temperature)
    diagnostics.update(
        id_ignored_offdiag=int((same_id & ~diagonal).sum().detach()),
        exact_duplicate_offdiag=int((same_image & ~diagonal).sum().detach()),
        batch_queries=batch,
    )
    losses = {
        'loss_retrieval': masked_ce(scores['final'], exact_allowed),
        'loss_identity': masked_ce(scores['identity'], identity_allowed),
        'loss_composition': masked_ce(scores['composition'], exact_allowed),
        'loss_composition_hard': hard,
    }
    return losses, diagnostics


@registry.register_model('oacir_qformer_identity_v2')
class Blip2QformerOacirQformerIdentityV2(Blip2QformerOacirQformerIdentity):
    """Same two branches/scoring; independent training-only combined variant."""
    requires_identity_labels = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.composition_margin = .05
        self.composition_margin_temperature = .07
        self.last_loss_diagnostics = {}
        # The old server may still have the strict-bbox original base module.
        # Handle compatibility ONLY in this new subclass; never patch the base.
        self.missing_bbox_policy = 'full-image'
        self.last_identity_bbox_stats = {'missing': 0, 'total': 0}
        self._v2_missing_bbox_warned = False

    def set_single_stage_config(self, **values):
        values = dict(values)
        policy = values.pop('missing_bbox_policy', None)
        if policy is not None:
            self.missing_bbox_policy = policy
        if self.missing_bbox_policy not in {'full-image', 'error'}:
            raise ValueError('missing_bbox_policy must be full-image or error')
        for key in ['composition_margin', 'composition_margin_temperature']:
            if key in values:
                value = values.pop(key)
                if value is not None:
                    setattr(self, key, float(value))
        super().set_single_stage_config(**values)
        if not math.isfinite(self.composition_margin) or self.composition_margin < 0:
            raise ValueError('composition_margin must be finite and nonnegative')
        if not math.isfinite(self.composition_margin_temperature) or self.composition_margin_temperature <= 0:
            raise ValueError('composition_margin_temperature must be finite and positive')

    def encode_identity_reference(self, reference_raw, reference_bbox):
        """Old/new base compatible; genuine boxes unchanged, missing stays None."""
        count = len(reference_raw)
        if reference_bbox is None:
            reference_bbox = [None] * count
        if len(reference_bbox) != count:
            raise ValueError('Expected one bbox entry, possibly None, per query')
        missing = sum(box is None for box in reference_bbox)
        self.last_identity_bbox_stats = {'missing': missing, 'total': count}
        if missing and self.missing_bbox_policy == 'error':
            raise ValueError(f'Identity reference missing bbox for {missing}/{count} queries')
        for box in reference_bbox:
            if box is None:
                continue
            coordinates = torch.as_tensor(box, dtype=torch.float32)
            if tuple(coordinates.shape) != (4,) or not torch.isfinite(coordinates).all():
                raise ValueError('Annotated reference bbox must be finite [4] xyxy coordinates')
            if (coordinates[2:] <= coordinates[:2]).any():
                raise ValueError('Empty/reversed reference bbox is not a valid annotation')
        if missing and not self._v2_missing_bbox_warned:
            warnings.warn('V2: missing bbox uses all reference patches as the identity anchor. '
                          'No bbox is fabricated and no sample is dropped.', RuntimeWarning)
            self._v2_missing_bbox_warned = True
        patches = self._project_raw_patch_tokens(reference_raw)
        mask = self._build_anchor_mask(reference_bbox, reference_raw.shape[1], count, patches.device)
        regions, _, attention = self.core_reference_encoder(patches, mask, self.bbox_floor)
        minimum = min(self.region_topk, patches.shape[1])
        maximum = min(max(self.region_topk_max, minimum), patches.shape[1])
        sizes = (mask.sum(-1).float() * self.region_area_scale).round().long().clamp(minimum, maximum)
        return regions, sizes, attention

    def forward(self, samples):
        for key in ['target_identity_ids', 'target_image_ids']:
            if key not in samples:
                raise ValueError('V2 requires annotation-derived ' + key + '; use the V2 runner')
        raw_ref, raw_target, ref_embeds, target_embeds, batch = self._encode_images_from_samples(samples)
        if len(raw_target) != batch or batch < 2:
            raise ValueError('Aligned batch with at least two queries required')
        composition = self.encode_composition(ref_embeds, samples['modification_text'])
        regions, sizes, _ = self.encode_identity_reference(raw_ref, samples.get('reference_bbox'))
        target_features = self._compute_target_features_from_embeds(target_embeds)
        target_patches = F.normalize(self._project_raw_patch_tokens(raw_target), dim=-1)
        scores = self.score_all_pairs(composition, regions, sizes, target_features, target_patches)
        # Tiny BxB score operation only; all image/text encodings reused.
        raw_cosines = self._compute_composition_scores(composition, target_features).float()
        losses, self.last_loss_diagnostics = training_objectives(
            scores, raw_cosines, samples['target_identity_ids'], samples['target_image_ids'],
            margin=self.composition_margin, margin_temperature=self.composition_margin_temperature)
        return losses
