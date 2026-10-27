"""Single-stage Q-Former composition + text-free local identity matching.

New registry name: oacir_qformer_identity. No CAAM/CRM/probe tokens, CORE
visual-text fusion, composition-region mask, candidate selection or reranking.
Only stateless helpers and the existing CORE reference-region encoder are
reused from the old latent module; the old model constructor is NOT called.
Use tools/run_oacir_qformer_identity.py, NOT the legacy train/evaluate entry.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.blip2_qformer_oacir_latent import (
    Blip2QformerOacirLatent, CoreReferenceRegionEncoder,
)


def identity_region_scores(reference_regions, target_patches, region_sizes,
                           cycle_weight=.25, spatial_kernel=3,
                           spatial_weight=.15, region_temperature=.07,
                           return_attention=False):
    """All Q x G pairs. NO text or composed-feature argument by design.

    Preserves the original text-free M_id selection and coverage/purity/cycle
    score. Top-k below selects SPATIAL PATCHES, never gallery images.
    """
    if reference_regions.ndim != 3 or target_patches.ndim != 3:
        raise ValueError('Expected reference [Q,R,D] and gallery [G,N,D]')
    if reference_regions.shape[-1] != target_patches.shape[-1]:
        raise ValueError('Reference/gallery identity dimensions differ')
    if tuple(region_sizes.shape) != (len(reference_regions),):
        raise ValueError('One region size is required per query')
    if spatial_kernel < 1 or spatial_kernel % 2 == 0 or region_temperature <= 0:
        raise ValueError('Positive odd spatial kernel and positive temperature required')
    with torch.autocast(device_type=reference_regions.device.type, enabled=False):
        refs = F.normalize(reference_regions.float(), dim=-1)
        targets = F.normalize(target_patches.float(), dim=-1)
        sim = torch.einsum('qrd,gnd->qgrn', refs, targets)
        forward_indices = sim.max(-1).indices
        reverse_values, reverse_indices = sim.max(-2)
        patch_count = targets.shape[1]
        patch_indices = torch.arange(patch_count, device=sim.device).view(1, 1, -1)
        cycles = forward_indices.gather(2, reverse_indices).eq(patch_indices)
        response = reverse_values + float(cycle_weight) * cycles.to(sim.dtype)
        grid = math.isqrt(patch_count)
        square = grid * grid == patch_count
        if square and spatial_kernel > 1:
            padding = spatial_kernel // 2
            response = F.avg_pool2d(F.pad(response.reshape(-1, 1, grid, grid),
                (padding,) * 4, mode='replicate'), spatial_kernel, stride=1).reshape_as(response)
        if square:
            yy, xx = torch.meshgrid(torch.arange(grid, device=sim.device),
                                    torch.arange(grid, device=sim.device), indexing='ij')
            coords = torch.stack([yy.flatten(), xx.flatten()], -1).to(sim.dtype)
        selected = torch.zeros_like(response, dtype=torch.bool)
        for query in range(len(refs)):
            size = max(1, min(int(region_sizes[query].item()), patch_count))
            rank = response[query]
            if square and size < patch_count:
                peak = coords[rank.argmax(-1)]
                distance = (coords[None] - peak[:, None]).square().sum(-1)
                rank = rank - float(spatial_weight) * distance / max((grid - 1) ** 2, 1)
            selected[query].scatter_(1, rank.topk(size, dim=-1).indices, True)
        minimum = torch.finfo(sim.dtype).min
        weights = (response / region_temperature).masked_fill(~selected, minimum).softmax(-1)
        coverage = sim.masked_fill(~selected.unsqueeze(2), minimum).max(-1).values.mean(-1)
        purity = (weights * reverse_values).sum(-1)
        cycle_similarity = (reverse_values * cycles).sum(-1) / cycles.sum(-1).clamp_min(1)
        scores = (coverage + purity + float(cycle_weight) * cycle_similarity) / (2. + cycle_weight)
        return (scores, weights) if return_attention else scores


@registry.register_model('oacir_qformer_identity')
class Blip2QformerOacirQformerIdentity(Blip2Base):
    PRETRAINED_MODEL_CONFIG_DICT = Blip2QformerOacirLatent.PRETRAINED_MODEL_CONFIG_DICT.copy()
    requires_reference_bbox = True
    requires_identity_labels = False  # Baseline single-positive CE; NOT multi-positive.
    requires_full_gallery = True

    # Reuse narrow image/token helpers only, not the old architecture/forward.
    _get_device = Blip2QformerOacirLatent._get_device
    _encode_images_from_samples = Blip2QformerOacirLatent._encode_images_from_samples
    _tokenize_text = Blip2QformerOacirLatent._tokenize_text
    _project_raw_patch_tokens = Blip2QformerOacirLatent._project_raw_patch_tokens
    _build_anchor_mask = Blip2QformerOacirLatent._build_anchor_mask
    _compute_target_features_from_embeds = Blip2QformerOacirLatent._compute_target_features_from_embeds
    _compute_composition_scores = Blip2QformerOacirLatent._compute_composition_scores

    def __init__(self, vit_model='eva_clip_g', img_size=224, drop_path_rate=0,
                 use_grad_checkpoint=False, vit_precision='fp16', freeze_vit=True,
                 num_query_token=32, cross_attention_freq=2, embed_dim=256,
                 max_txt_len=32, num_regions=16):
        super().__init__()
        self.tokenizer = self.init_tokenizer()
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision)
        if freeze_vit:
            for parameter in self.visual_encoder.parameters():
                parameter.requires_grad = False
            self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq)
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.config.gradient_checkpointing = True
        hidden = self.Qformer.config.hidden_size
        self.vision_proj = nn.Linear(hidden, embed_dim)
        self.text_proj = nn.Linear(hidden, embed_dim)
        self.instance_proj = nn.Linear(self.ln_vision.normalized_shape[0], embed_dim)
        nn.init.orthogonal_(self.instance_proj.weight)
        nn.init.zeros_(self.instance_proj.bias)
        self.core_reference_encoder = CoreReferenceRegionEncoder(embed_dim, num_regions)
        self.temp = nn.Parameter(torch.tensor(.07))
        self.identity_temperature = .07
        self.identity_weight = 1.25
        self.cycle_weight, self.bbox_floor = .25, .05
        self.region_topk, self.region_topk_max, self.region_area_scale = 16, 96, 1.5
        self.region_spatial_kernel, self.region_spatial_weight = 3, .15
        self.region_temperature = .07
        self.query_chunk_size, self.gallery_chunk_size = 2, 32
        self.max_txt_len, self.num_query_token = max_txt_len, num_query_token
        self.latent_image_size, self.latent_bbox_format = img_size, 'xyxy'

    def set_single_stage_config(self, **values):
        allowed = {'identity_temperature', 'identity_weight', 'cycle_weight', 'bbox_floor',
                   'region_topk', 'region_topk_max', 'region_area_scale',
                   'region_spatial_kernel', 'region_spatial_weight', 'region_temperature',
                   'query_chunk_size', 'gallery_chunk_size', 'max_txt_len'}
        if set(values) - allowed:
            raise ValueError('Unknown single-stage settings: ' + str(set(values) - allowed))
        for name, value in values.items():
            if value is not None:
                setattr(self, name, value)
        if not all(math.isfinite(float(getattr(self, name))) for name in allowed):
            raise ValueError('All single-stage settings must be finite')
        if min(self.identity_temperature, self.region_temperature, self.region_area_scale) <= 0:
            raise ValueError('Temperatures and area scale must be positive')
        if min(self.query_chunk_size, self.gallery_chunk_size, self.region_topk, self.region_topk_max) < 1:
            raise ValueError('Chunk sizes and spatial patch counts must be positive')
        if self.identity_weight < 0 or not 0 < self.bbox_floor <= 1:
            raise ValueError('Nonnegative identity weight and bbox_floor in (0,1] required')
        if self.cycle_weight < 0 or self.region_spatial_weight < 0:
            raise ValueError('Cycle/spatial weights must be nonnegative')
        if self.region_spatial_kernel < 1 or self.region_spatial_kernel % 2 == 0:
            raise ValueError('Spatial kernel must be positive and odd')
        capacity = self.Qformer.bert.embeddings.position_embeddings.num_embeddings
        if not 2 <= self.max_txt_len <= capacity:
            raise ValueError('max_txt_len exceeds Q-Former position capacity')

    def encode_composition(self, reference_embeds, modification_text):
        """Full reference image + text. No bbox, bias, probes or extra text encoder."""
        tokens = self._tokenize_text(modification_text, reference_embeds.device)
        queries = self.query_tokens.expand(len(reference_embeds), -1, -1)
        query_atts = torch.ones(queries.shape[:2], dtype=torch.long, device=queries.device)
        output = self.Qformer.bert(tokens.input_ids, query_embeds=queries,
            attention_mask=torch.cat([query_atts, tokens.attention_mask], dim=1),
            encoder_hidden_states=reference_embeds,
            encoder_attention_mask=torch.ones(reference_embeds.shape[:2], dtype=torch.long,
                                               device=queries.device),
            use_cache=False, return_dict=True)
        return F.normalize(self.text_proj(output.last_hidden_state[:, self.num_query_token]), dim=-1)

    def encode_identity_reference(self, reference_raw, reference_bbox):
        """Pure image/bbox path. No text/fusion feature is accepted."""
        if reference_bbox is None or any(box is None for box in reference_bbox):
            raise ValueError('Identity reference requires an annotated bbox for every query')
        boxes = torch.as_tensor(reference_bbox, dtype=torch.float32)
        if tuple(boxes.shape) != (len(reference_raw), 4) or not torch.isfinite(boxes).all():
            raise ValueError('Reference bbox must be finite [B,4] transformed xyxy coordinates')
        if (boxes[:, 2:] <= boxes[:, :2]).any():
            raise ValueError('Empty/reversed reference bbox is not a valid identity annotation')
        patches = self._project_raw_patch_tokens(reference_raw)
        mask = self._build_anchor_mask(reference_bbox, reference_raw.shape[1],
                                      len(reference_raw), patches.device)
        regions, _, attention = self.core_reference_encoder(patches, mask, self.bbox_floor)
        minimum = min(self.region_topk, patches.shape[1])
        maximum = min(max(self.region_topk_max, minimum), patches.shape[1])
        sizes = (mask.sum(-1).float() * self.region_area_scale).round().long().clamp(minimum, maximum)
        return regions, sizes, attention

    def encode_gallery_from_raw(self, raw):
        raw = raw.to(self._get_device(), non_blocking=True)
        normalized = self.ln_vision(raw.to(self.ln_vision.weight.dtype))
        composition = self._compute_target_features_from_embeds(normalized)
        identity = F.normalize(self._project_raw_patch_tokens(raw), dim=-1)
        return composition, identity

    def encode_query_from_raw(self, raw, modification_text, reference_bbox):
        raw = raw.to(self._get_device(), non_blocking=True)
        normalized = self.ln_vision(raw.to(self.ln_vision.weight.dtype))
        composition = self.encode_composition(normalized, modification_text)
        regions, sizes, _ = self.encode_identity_reference(raw, reference_bbox)
        return composition, regions, sizes

    def _identity_block(self, regions, patches, sizes):
        return identity_region_scores(regions, patches, sizes, self.cycle_weight,
            self.region_spatial_kernel, self.region_spatial_weight, self.region_temperature)

    def score_all_pairs(self, query_features, reference_regions, region_sizes,
                        gallery_features, gallery_patches):
        """Complete Q x G matrices. Chunking NEVER selects gallery candidates."""
        if not len(query_features) or not len(gallery_features) or len(gallery_features) != len(gallery_patches):
            raise ValueError('Nonempty aligned query/gallery features are required')
        comp = self._compute_composition_scores(query_features, gallery_features).float()
        rows = []
        for qs in range(0, len(query_features), self.query_chunk_size):
            qe = qs + self.query_chunk_size
            columns = []
            for gs in range(0, len(gallery_patches), self.gallery_chunk_size):
                ge = gs + self.gallery_chunk_size
                inputs = (reference_regions[qs:qe], gallery_patches[gs:ge], region_sizes[qs:qe])
                if self.training and torch.is_grad_enabled():
                    block = checkpoint(self._identity_block, *inputs, use_reentrant=False,
                                       preserve_rng_state=False)
                else:
                    block = self._identity_block(*inputs)
                columns.append(block)
            rows.append(torch.cat(columns, dim=1))
        identity = torch.cat(rows, dim=0)
        comp_logits = comp / self.temp.clamp(.001, .5)
        id_logits = identity / self.identity_temperature
        return {'final': comp_logits + self.identity_weight * id_logits,
                'composition': comp_logits, 'identity': id_logits}

    def forward(self, samples):
        raw_ref, raw_target, ref_embeds, target_embeds, batch = self._encode_images_from_samples(samples)
        if len(raw_target) != batch or batch < 2:
            raise ValueError('Training requires an aligned batch with at least two pairs')
        query = self.encode_composition(ref_embeds, samples['modification_text'])
        regions, sizes, _ = self.encode_identity_reference(raw_ref, samples.get('reference_bbox'))
        target_features = self._compute_target_features_from_embeds(target_embeds)
        target_patches = F.normalize(self._project_raw_patch_tokens(raw_target), dim=-1)
        scores = self.score_all_pairs(query, regions, sizes, target_features, target_patches)
        labels = torch.arange(batch, device=query.device)
        return {'loss_retrieval': F.cross_entropy(scores['final'], labels),
                'loss_identity': F.cross_entropy(scores['identity'], labels),
                'loss_composition': F.cross_entropy(scores['composition'], labels)}

    @torch.no_grad()
    def extract_target_features_from_raw(self, raw, mode='tokens'):
        if mode != 'tokens':
            raise ValueError('Single-stage gallery needs tokens, not pooled features')
        composition, identity = self.encode_gallery_from_raw(raw)
        # Tensor packing permits use of the EXISTING read-only raw-cache loader.
        return torch.cat([composition.float(), identity.float()], dim=1)

    @torch.no_grad()
    def extract_target_features_with_raw(self, images, mode='tokens'):
        raw = self.visual_encoder(images.to(self._get_device()))
        return self.extract_target_features_from_raw(raw, mode), raw

    @classmethod
    def from_config(cls, cfg):
        model = cls(vit_model=cfg.get('vit_model', 'eva_clip_g'),
            img_size=cfg.get('image_size', 224), num_query_token=cfg.get('num_query_token', 32),
            cross_attention_freq=cfg.get('cross_attention_freq', 2),
            drop_path_rate=cfg.get('drop_path_rate', 0),
            use_grad_checkpoint=cfg.get('use_grad_checkpoint', False),
            vit_precision=cfg.get('vit_precision', 'fp16'), freeze_vit=cfg.get('freeze_vit', True),
            embed_dim=cfg.get('embed_dim', 256), max_txt_len=cfg.get('max_txt_len', 32),
            num_regions=cfg.get('num_regions', 16))
        model.load_checkpoint_from_config(cfg)
        return model
