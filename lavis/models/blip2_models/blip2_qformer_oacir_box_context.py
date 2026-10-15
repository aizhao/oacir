"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    disabled_train,
)
from lavis.models.blip2_models.oacir_latent_matching import (
    bbox_to_patch_mask as latent_bbox_to_patch_mask,
    compute_bidirectional_region_scores,
)


class BoxContextTokenizer(nn.Module):
    """Build fixed reference tokens directly from a box and its surroundings."""

    def __init__(self, embed_dim: int, output_size: int = 4, context_scale: float = 1.5):
        super().__init__()
        self.output_size = int(output_size)
        self.context_scale = float(context_scale)
        self.context_delta = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.geometry_proj = nn.Sequential(
            nn.Linear(6, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.token_norm = nn.LayerNorm(embed_dim)

        # Start as plain box sampling; context and geometry are learned as
        # residual corrections instead of perturbing the pretrained tokens.
        nn.init.zeros_(self.context_delta[-1].weight)
        nn.init.zeros_(self.context_delta[-1].bias)
        nn.init.zeros_(self.geometry_proj[-1].weight)
        nn.init.zeros_(self.geometry_proj[-1].bias)

    @staticmethod
    def _as_box_tensor(reference_bbox, batch_size, image_size, device, dtype):
        boxes = []
        for index in range(batch_size):
            box = reference_bbox[index] if reference_bbox is not None else None
            if box is None:
                values = [0.0, 0.0, float(image_size), float(image_size)]
            elif isinstance(box, torch.Tensor):
                values = box.detach().flatten().tolist()
            else:
                values = list(box)
            if len(values) != 4:
                raise ValueError(f"Expected four bbox values, got {values}")
            boxes.append(values)

        boxes = torch.tensor(boxes, device=device, dtype=dtype)
        x1 = torch.minimum(boxes[:, 0], boxes[:, 2]).clamp(0, image_size)
        y1 = torch.minimum(boxes[:, 1], boxes[:, 3]).clamp(0, image_size)
        x2 = torch.maximum(boxes[:, 0], boxes[:, 2]).clamp(0, image_size)
        y2 = torch.maximum(boxes[:, 1], boxes[:, 3]).clamp(0, image_size)
        minimum_extent = float(image_size) / 1024.0
        x2 = torch.maximum(x2, x1 + minimum_extent).clamp(max=image_size)
        y2 = torch.maximum(y2, y1 + minimum_extent).clamp(max=image_size)
        x1 = torch.minimum(x1, x2 - minimum_extent).clamp(min=0)
        y1 = torch.minimum(y1, y2 - minimum_extent).clamp(min=0)
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def _sample_box_tokens(self, feature_map, boxes, image_size):
        batch_size = feature_map.size(0)
        steps = (
            torch.arange(self.output_size, device=feature_map.device, dtype=feature_map.dtype)
            + 0.5
        ) / float(self.output_size)
        x = boxes[:, 0:1] + steps.unsqueeze(0) * (boxes[:, 2:3] - boxes[:, 0:1])
        y = boxes[:, 1:2] + steps.unsqueeze(0) * (boxes[:, 3:4] - boxes[:, 1:2])
        grid_x = x[:, None, :].expand(-1, self.output_size, -1)
        grid_y = y[:, :, None].expand(-1, -1, self.output_size)
        grid = torch.stack(
            [2.0 * grid_x / float(image_size) - 1.0, 2.0 * grid_y / float(image_size) - 1.0],
            dim=-1,
        )
        sampled = F.grid_sample(
            feature_map,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled.permute(0, 2, 3, 1).reshape(batch_size, -1, feature_map.size(1))

    def _context_features(self, patch_tokens, boxes, grid_size, image_size):
        centers = (
            torch.arange(grid_size, device=patch_tokens.device, dtype=patch_tokens.dtype)
            + 0.5
        ) * (float(image_size) / float(grid_size))
        yy, xx = torch.meshgrid(centers, centers, indexing="ij")
        xx = xx.flatten().unsqueeze(0)
        yy = yy.flatten().unsqueeze(0)

        inside = (
            (xx >= boxes[:, 0:1])
            & (xx <= boxes[:, 2:3])
            & (yy >= boxes[:, 1:2])
            & (yy <= boxes[:, 3:4])
        )
        center_x = (boxes[:, 0] + boxes[:, 2]) * 0.5
        center_y = (boxes[:, 1] + boxes[:, 3]) * 0.5
        half_width = (boxes[:, 2] - boxes[:, 0]) * 0.5 * self.context_scale
        half_height = (boxes[:, 3] - boxes[:, 1]) * 0.5 * self.context_scale
        expanded = torch.stack(
            [
                (center_x - half_width).clamp(0, image_size),
                (center_y - half_height).clamp(0, image_size),
                (center_x + half_width).clamp(0, image_size),
                (center_y + half_height).clamp(0, image_size),
            ],
            dim=-1,
        )
        expanded_mask = (
            (xx >= expanded[:, 0:1])
            & (xx <= expanded[:, 2:3])
            & (yy >= expanded[:, 1:2])
            & (yy <= expanded[:, 3:4])
        )
        ring_mask = expanded_mask & ~inside
        outside_mask = ~inside
        use_ring = ring_mask.any(dim=-1, keepdim=True)
        context_mask = torch.where(use_ring, ring_mask, outside_mask)
        context_mask = torch.where(
            context_mask.any(dim=-1, keepdim=True),
            context_mask,
            torch.ones_like(context_mask),
        )
        weights = context_mask.to(dtype=patch_tokens.dtype)
        context = torch.einsum("bn,bnd->bd", weights, patch_tokens)
        return context / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)

    def forward(self, patch_tokens, reference_bbox, image_size):
        batch_size, num_patches, embed_dim = patch_tokens.shape
        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            raise ValueError(f"Box tokenizer requires a square patch grid, got {num_patches}")

        boxes = self._as_box_tensor(
            reference_bbox,
            batch_size,
            image_size,
            patch_tokens.device,
            patch_tokens.dtype,
        )
        feature_map = patch_tokens.transpose(1, 2).reshape(
            batch_size,
            embed_dim,
            grid_size,
            grid_size,
        )
        object_tokens = self._sample_box_tokens(feature_map, boxes, image_size)
        context = self._context_features(patch_tokens, boxes, grid_size, image_size)

        width = (boxes[:, 2] - boxes[:, 0]).clamp_min(1e-6)
        height = (boxes[:, 3] - boxes[:, 1]).clamp_min(1e-6)
        geometry = torch.stack(
            [
                boxes[:, 0] / float(image_size),
                boxes[:, 1] / float(image_size),
                boxes[:, 2] / float(image_size),
                boxes[:, 3] / float(image_size),
                width * height / float(image_size * image_size),
                torch.log(width / height),
            ],
            dim=-1,
        )
        geometry_features = self.geometry_proj(geometry)
        region_tokens = self.token_norm(
            object_tokens
            + self.context_delta(object_tokens - context.unsqueeze(1))
            + geometry_features.unsqueeze(1)
        )
        region_tokens = F.normalize(region_tokens, dim=-1)

        anchor_mask = (
            (
                torch.arange(grid_size, device=patch_tokens.device, dtype=patch_tokens.dtype)
                + 0.5
            )
            * (float(image_size) / float(grid_size))
        )
        yy, xx = torch.meshgrid(anchor_mask, anchor_mask, indexing="ij")
        xx = xx.flatten().unsqueeze(0)
        yy = yy.flatten().unsqueeze(0)
        attention = (
            (xx >= boxes[:, 0:1])
            & (xx <= boxes[:, 2:3])
            & (yy >= boxes[:, 1:2])
            & (yy <= boxes[:, 3:4])
        ).to(dtype=patch_tokens.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return region_tokens, attention.unsqueeze(1)


class RegionQueryAdapter(nn.Module):
    """Adapt the AdaFocal composition embedding for raw patch matching."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, composition_features):
        residual = self.adapter(self.norm(composition_features))
        adapted = F.normalize(composition_features + residual, dim=-1)
        residual_ratio = residual.norm(dim=-1, keepdim=True) / composition_features.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)
        return adapted, residual_ratio


class ContextualReasoningModule(nn.Module):
    """
    AdaFocal CRM used by CAAM to predict the reference-box attention bias scale.
    """

    def __init__(self, input_dim: int, nhead: int = 8, num_encoder_layers: int = 2, hidden_dim: int = 512):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )
        self.output_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.activation = nn.Softplus()

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        cls_token = self.cls_token.expand(fused_features.shape[0], -1, -1)
        contextual_sequence = torch.cat([cls_token, fused_features], dim=1)
        contextual_output = self.transformer_encoder(contextual_sequence)
        return self.activation(self.output_head(contextual_output[:, 0]))


@registry.register_model("oacir_box_context")
class Blip2QformerOacirBoxContext(Blip2Base):
    """
    AdaFocal backbone with box-context target-region reranking for OACIR.

    AdaFocal provides the full-gallery composition ranking. A box-native
    tokenizer preserves reference parts and local context, while a residual
    query adapter maps the existing Q-Former composition embedding into the
    raw-patch space used for latent target-region discovery.

    Final score:
        S_final = S_adafocal + fusion_weight * S_matcher.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        caam_hidden_dim=512,
        num_probe_token=8,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )

        if freeze_vit:
            for _, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.config.gradient_checkpointing = True

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        qformer_hidden_dim = self.Qformer.config.hidden_size
        self.crm_module = ContextualReasoningModule(
            input_dim=qformer_hidden_dim,
            nhead=8,
            num_encoder_layers=2,
            hidden_dim=caam_hidden_dim or 512,
        )
        self.num_probe_token = num_probe_token or 8
        self.contextual_probe_tokens = nn.Parameter(
            torch.zeros(1, self.num_probe_token, qformer_hidden_dim)
        )
        self.contextual_probe_tokens.data.normal_(
            mean=0.0,
            std=self.Qformer.config.initializer_range,
        )

        vision_width = self.ln_vision.normalized_shape[0]
        self.instance_proj = nn.Linear(vision_width, embed_dim)
        nn.init.orthogonal_(self.instance_proj.weight)
        nn.init.zeros_(self.instance_proj.bias)

        self.box_context_tokenizer = BoxContextTokenizer(
            embed_dim=embed_dim,
            output_size=4,
            context_scale=1.5,
        )
        self.region_query_adapter = RegionQueryAdapter(embed_dim=embed_dim)

        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.core_matcher_temp = 0.07
        self.core_matcher_lambda_id = 0.5
        self.core_matcher_fusion_weight = 0.02
        self.core_matcher_topk = 50
        self.core_matcher_cycle_weight = 0.25
        self.core_matcher_text_weight = 0.25
        self.core_matcher_bbox_floor = 0.05
        self.core_matcher_query_chunk_size = 8
        self.region_topk = 16
        self.region_topk_max = 96
        self.region_area_scale = 1.5
        self.region_spatial_kernel = 3
        self.region_spatial_weight = 0.15
        self.region_temperature = 0.07
        self.latent_image_size = img_size or 224
        self.latent_bbox_format = "xyxy"
        self.requires_reference_bbox = True

        self.max_txt_len = max_txt_len
        self.num_query_token = num_query_token

    def set_latent_config(
        self,
        core_matcher_temp=None,
        core_matcher_lambda_id=None,
        core_matcher_fusion_weight=None,
        core_matcher_topk=None,
        core_matcher_cycle_weight=None,
        core_matcher_text_weight=None,
        core_matcher_bbox_floor=None,
        core_matcher_query_chunk_size=None,
        region_topk=None,
        region_topk_max=None,
        region_area_scale=None,
        region_spatial_kernel=None,
        region_spatial_weight=None,
        region_temperature=None,
    ):
        if core_matcher_temp is not None:
            self.core_matcher_temp = float(core_matcher_temp)
        if core_matcher_lambda_id is not None:
            self.core_matcher_lambda_id = float(core_matcher_lambda_id)
        if core_matcher_fusion_weight is not None:
            self.core_matcher_fusion_weight = float(core_matcher_fusion_weight)
        if core_matcher_topk is not None:
            self.core_matcher_topk = int(core_matcher_topk)
        if core_matcher_cycle_weight is not None:
            self.core_matcher_cycle_weight = float(core_matcher_cycle_weight)
        if core_matcher_text_weight is not None:
            self.core_matcher_text_weight = float(core_matcher_text_weight)
        if core_matcher_bbox_floor is not None:
            self.core_matcher_bbox_floor = float(core_matcher_bbox_floor)
        if core_matcher_query_chunk_size is not None:
            self.core_matcher_query_chunk_size = int(core_matcher_query_chunk_size)
        if region_topk is not None:
            self.region_topk = int(region_topk)
        if region_topk_max is not None:
            self.region_topk_max = int(region_topk_max)
        if region_area_scale is not None:
            self.region_area_scale = float(region_area_scale)
        if region_spatial_kernel is not None:
            self.region_spatial_kernel = int(region_spatial_kernel)
        if region_spatial_weight is not None:
            self.region_spatial_weight = float(region_spatial_weight)
        if region_temperature is not None:
            self.region_temperature = float(region_temperature)

    def _get_device(self):
        return next(self.parameters()).device

    def _encode_images_from_samples(self, samples):
        if "reference_image_embeds_raw" in samples:
            device = self._get_device()
            reference_raw = samples["reference_image_embeds_raw"].to(device, non_blocking=True)
            target_raw = samples["target_image_embeds_raw"].to(device, non_blocking=True)
            reference_embeds = self.ln_vision(reference_raw.to(dtype=self.ln_vision.weight.dtype))
            target_embeds = self.ln_vision(target_raw.to(dtype=self.ln_vision.weight.dtype))
            batch_size = reference_embeds.size(0)
            return reference_raw, target_raw, reference_embeds, target_embeds, batch_size

        reference_image = samples["reference_image"]
        target_image = samples["target_image"]
        reference_raw = self.visual_encoder(reference_image)
        target_raw = self.visual_encoder(target_image)
        reference_embeds = self.ln_vision(reference_raw)
        target_embeds = self.ln_vision(target_raw)
        batch_size = reference_image.size(0)
        return reference_raw, target_raw, reference_embeds, target_embeds, batch_size

    def _tokenize_text(self, modification_text, device):
        return self.tokenizer(
            modification_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

    @staticmethod
    def _has_valid_reference_bbox(reference_bbox):
        return reference_bbox is not None and any(box is not None for box in reference_bbox)

    def _compute_adafocal_attention_bias(
        self,
        reference_image_embeds,
        text_tokens,
        reference_image_atts,
        reference_bbox,
    ):
        if not self._has_valid_reference_bbox(reference_bbox):
            return None

        device = reference_image_embeds.device
        probe_tokens = self.contextual_probe_tokens.expand(
            reference_image_embeds.shape[0],
            -1,
            -1,
        )
        probe_atts = torch.ones(
            probe_tokens.size()[:-1],
            dtype=torch.long,
            device=device,
        )
        pre_fusion_atts = torch.cat(
            [probe_atts, text_tokens.attention_mask],
            dim=1,
        )

        pre_fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=probe_tokens,
            attention_mask=pre_fusion_atts,
            encoder_hidden_states=reference_image_embeds,
            encoder_attention_mask=reference_image_atts,
            use_cache=False,
            return_dict=True,
        )
        pre_fusion_features = pre_fusion_output.last_hidden_state[
            :,
            : self.num_probe_token,
            :,
        ]
        adaptive_bias_scalar = self.crm_module(pre_fusion_features)

        patch_mask = latent_bbox_to_patch_mask(
            reference_bbox,
            num_tokens=reference_image_embeds.size(1),
            image_size=self.latent_image_size,
            has_cls_token=True,
            bbox_format=self.latent_bbox_format,
            device=device,
        ).to(dtype=reference_image_embeds.dtype)
        cls_mask = torch.zeros(
            patch_mask.size(0),
            1,
            dtype=patch_mask.dtype,
            device=device,
        )
        attention_mask = torch.cat([cls_mask, patch_mask], dim=1)
        return (adaptive_bias_scalar * attention_mask).unsqueeze(1).unsqueeze(1)

    def _compute_fusion_features(
        self,
        reference_image_embeds,
        modification_text,
        reference_bbox=None,
        return_token_features=False,
    ):
        device = reference_image_embeds.device
        reference_image_atts = torch.ones(
            reference_image_embeds.size()[:-1],
            dtype=torch.long,
            device=device,
        )

        query_tokens = self.query_tokens.expand(reference_image_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=device)
        text_tokens = self._tokenize_text(modification_text, device)
        fusion_atts = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        attention_bias = self._compute_adafocal_attention_bias(
            reference_image_embeds,
            text_tokens,
            reference_image_atts,
            reference_bbox,
        )

        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=fusion_atts,
            encoder_hidden_states=reference_image_embeds,
            encoder_attention_mask=reference_image_atts,
            use_cache=False,
            return_dict=True,
            attention_bias=attention_bias,
        )

        fusion_features = F.normalize(
            self.text_proj(fusion_output.last_hidden_state[:, self.num_query_token, :]),
            dim=-1,
        )
        if not return_token_features:
            return fusion_features

        fusion_token_features = F.normalize(
            self.text_proj(fusion_output.last_hidden_state[:, : self.num_query_token, :]),
            dim=-1,
        )
        return fusion_features, fusion_token_features

    def _compute_target_features_from_embeds(self, target_image_embeds):
        device = target_image_embeds.device
        target_image_atts = torch.ones(
            target_image_embeds.size()[:-1],
            dtype=torch.long,
            device=device,
        )
        query_tokens = self.query_tokens.expand(target_image_embeds.shape[0], -1, -1)

        target_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=target_image_embeds,
            encoder_attention_mask=target_image_atts,
            use_cache=False,
            return_dict=True,
        )

        return F.normalize(self.vision_proj(target_output.last_hidden_state), dim=-1)

    def _compute_composition_scores(self, fusion_features, target_features):
        if target_features.dtype != fusion_features.dtype:
            target_features = target_features.to(dtype=fusion_features.dtype)
        if target_features.dim() == 2:
            return torch.matmul(fusion_features, target_features.t())
        return torch.einsum("bd,nqd->bnq", fusion_features, target_features).max(dim=-1).values

    def _project_raw_patch_tokens(self, image_embeds_raw):
        normalized_tokens = self.ln_vision(
            image_embeds_raw.to(dtype=self.ln_vision.weight.dtype)
        )
        patch_tokens = normalized_tokens[:, 1:, :].to(dtype=self.instance_proj.weight.dtype)
        return self.instance_proj(patch_tokens)

    def _build_anchor_mask(self, reference_bbox, num_tokens, batch_size, device):
        if reference_bbox is None:
            reference_bbox = [None] * batch_size
        return latent_bbox_to_patch_mask(
            reference_bbox,
            num_tokens=num_tokens,
            image_size=self.latent_image_size,
            has_cls_token=True,
            bbox_format=self.latent_bbox_format,
            device=device,
        )

    @staticmethod
    def _build_candidate_indices(similarity_logits, topk, force_positive=False):
        """Select backbone candidates, optionally forcing aligned positives in."""
        num_queries, num_targets = similarity_logits.shape
        topk = min(max(int(topk), 1), num_targets)
        with torch.no_grad():
            candidate_indices = similarity_logits.detach().topk(topk, dim=1).indices
            if force_positive:
                if num_queries != num_targets:
                    raise ValueError("Aligned positives require a square training score matrix")
                targets = torch.arange(
                    num_queries,
                    dtype=torch.long,
                    device=similarity_logits.device,
                )
                has_positive = (candidate_indices == targets.unsqueeze(1)).any(dim=1)
                if not has_positive.all():
                    candidate_indices = candidate_indices.clone()
                    candidate_indices[~has_positive, -1] = targets[~has_positive]
        return candidate_indices

    def _compute_box_reference_features(
        self,
        reference_image_embeds_raw,
        reference_bbox,
        composition_features,
    ):
        reference_tokens = self._project_raw_patch_tokens(reference_image_embeds_raw)
        anchor_mask = self._build_anchor_mask(
            reference_bbox=reference_bbox,
            num_tokens=reference_image_embeds_raw.size(1),
            batch_size=reference_image_embeds_raw.size(0),
            device=reference_image_embeds_raw.device,
        )
        region_tokens, reference_attention = self.box_context_tokenizer(
            reference_tokens,
            reference_bbox,
            image_size=self.latent_image_size,
        )
        region_query, adapter_ratio = self.region_query_adapter(
            composition_features.to(dtype=reference_tokens.dtype)
        )

        num_target_patches = reference_tokens.size(1)
        min_region_size = min(max(int(self.region_topk), 1), num_target_patches)
        max_region_size = min(
            max(int(self.region_topk_max), min_region_size),
            num_target_patches,
        )
        region_sizes = (
            anchor_mask.sum(dim=-1).to(dtype=torch.float32)
            * float(self.region_area_scale)
        ).round().long().clamp(min=min_region_size, max=max_region_size)
        return {
            "region_tokens": region_tokens,
            "composed_feature": region_query,
            "reference_attention": reference_attention,
            "adapter_ratio": adapter_ratio,
            "region_sizes": region_sizes,
        }

    def _compute_box_matcher_candidate_scores(
        self,
        reference_image_embeds_raw,
        target_image_embeds_raw,
        reference_bbox,
        composition_features,
        candidate_indices=None,
        return_heatmap=False,
    ):
        """Run box-context latent-region discovery on selected candidates.

        Training passes a shared target bank [T, N, D] plus candidate indices.
        Exact gallery reranking passes already gathered candidates [B, C, N, D]
        so only the global AdaFocal Top-K raw tokens need to reach the GPU.
        """
        box_reference = self._compute_box_reference_features(
            reference_image_embeds_raw,
            reference_bbox,
            composition_features,
        )
        if target_image_embeds_raw.dim() == 3:
            if candidate_indices is None:
                raise ValueError("candidate_indices are required for a shared target bank")
            target_tokens = F.normalize(
                self._project_raw_patch_tokens(target_image_embeds_raw),
                dim=-1,
            )
            num_queries = candidate_indices.size(0)
        elif target_image_embeds_raw.dim() == 4:
            if candidate_indices is not None:
                raise ValueError("candidate_indices must be omitted for gathered candidates")
            if target_image_embeds_raw.size(0) != reference_image_embeds_raw.size(0):
                raise ValueError("Gathered candidates must have one candidate bank per query")
            target_tokens = None
            num_queries = target_image_embeds_raw.size(0)
        else:
            raise ValueError("target_image_embeds_raw must be [T,N,D] or [B,C,N,D]")

        matching_output_keys = (
            "score_id",
            "score_comp",
            "cycle_ratio",
            "entropy",
        )
        collected_keys = (
            "score_id",
            "score_comp",
            "cycle_ratio",
            "entropy",
        )
        collected = {key: [] for key in collected_keys}
        heatmaps = []
        query_chunk_size = max(int(self.core_matcher_query_chunk_size), 1)

        for start in range(0, num_queries, query_chunk_size):
            end = min(start + query_chunk_size, num_queries)
            if target_tokens is not None:
                current_indices = candidate_indices[start:end]
                reference_regions = box_reference["region_tokens"][start:end]
                composed_features = box_reference["composed_feature"][start:end]
                region_sizes = box_reference["region_sizes"][start:end]

                if self.training and torch.is_grad_enabled():
                    def checkpointed_matching(
                        current_reference_regions,
                        all_target_tokens,
                        current_composed_features,
                        current_region_sizes,
                        current_candidate_indices,
                    ):
                        candidate_tokens = all_target_tokens[current_candidate_indices]
                        matched = compute_bidirectional_region_scores(
                            current_reference_regions,
                            candidate_tokens,
                            current_composed_features,
                            current_region_sizes,
                            spatial_kernel_size=self.region_spatial_kernel,
                            spatial_weight=self.region_spatial_weight,
                            region_temperature=self.region_temperature,
                            cycle_weight=self.core_matcher_cycle_weight,
                            text_weight=self.core_matcher_text_weight,
                            return_heatmap=False,
                        )
                        return tuple(matched[key] for key in matching_output_keys)

                    checkpoint_outputs = checkpoint(
                        checkpointed_matching,
                        reference_regions,
                        target_tokens,
                        composed_features,
                        region_sizes,
                        current_indices,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                    chunk_outputs = dict(zip(matching_output_keys, checkpoint_outputs))
                else:
                    candidate_tokens = target_tokens[current_indices]
                    chunk_outputs = compute_bidirectional_region_scores(
                        reference_regions,
                        candidate_tokens,
                        composed_features,
                        region_sizes,
                        spatial_kernel_size=self.region_spatial_kernel,
                        spatial_weight=self.region_spatial_weight,
                        region_temperature=self.region_temperature,
                        cycle_weight=self.core_matcher_cycle_weight,
                        text_weight=self.core_matcher_text_weight,
                        return_heatmap=return_heatmap,
                    )
            else:
                raw_candidates = target_image_embeds_raw[start:end].to(
                    reference_image_embeds_raw.device,
                    non_blocking=True,
                )
                raw_shape = raw_candidates.shape
                candidate_tokens = self._project_raw_patch_tokens(
                    raw_candidates.reshape(-1, raw_shape[-2], raw_shape[-1])
                )
                candidate_tokens = F.normalize(candidate_tokens, dim=-1).reshape(
                    raw_shape[0],
                    raw_shape[1],
                    raw_shape[-2] - 1,
                    -1,
                )
                chunk_outputs = compute_bidirectional_region_scores(
                    box_reference["region_tokens"][start:end],
                    candidate_tokens,
                    box_reference["composed_feature"][start:end],
                    box_reference["region_sizes"][start:end],
                    spatial_kernel_size=self.region_spatial_kernel,
                    spatial_weight=self.region_spatial_weight,
                    region_temperature=self.region_temperature,
                    cycle_weight=self.core_matcher_cycle_weight,
                    text_weight=self.core_matcher_text_weight,
                    return_heatmap=return_heatmap,
                )

            for key in collected_keys:
                collected[key].append(chunk_outputs[key])
            if return_heatmap:
                heatmaps.append(chunk_outputs["target_attention"])

        outputs = {key: torch.cat(value, dim=0) for key, value in collected.items()}
        outputs["score_final"] = (
            outputs["score_comp"]
            + float(self.core_matcher_lambda_id) * outputs["score_id"]
        )
        outputs["reference_attention"] = box_reference["reference_attention"]
        outputs["adapter_ratio"] = box_reference["adapter_ratio"]
        if return_heatmap:
            outputs["target_attention"] = torch.cat(heatmaps, dim=0)
        return outputs

    @staticmethod
    def merge_topk_ranking(
        composition_logits,
        candidate_indices,
        reranked_candidate_logits,
    ):
        """Merge reranked candidates while preserving the original Top-K set."""
        if candidate_indices.shape != reranked_candidate_logits.shape:
            raise ValueError("Candidate indices and reranked logits must have the same shape")
        if composition_logits.size(0) != candidate_indices.size(0):
            raise ValueError("Composition and candidate batch sizes must match")

        num_queries, num_targets = composition_logits.shape
        candidate_mask = torch.zeros_like(composition_logits, dtype=torch.bool)
        candidate_mask.scatter_(1, candidate_indices, True)

        reranked_order = reranked_candidate_logits.argsort(dim=1, descending=True)
        ordered_candidates = torch.gather(candidate_indices, 1, reranked_order)

        backbone_order = composition_logits.argsort(dim=1, descending=True)
        ordered_candidate_mask = torch.gather(candidate_mask, 1, backbone_order)
        ordered_non_candidates = backbone_order[~ordered_candidate_mask].reshape(
            num_queries,
            num_targets - candidate_indices.size(1),
        )
        final_order = torch.cat([ordered_candidates, ordered_non_candidates], dim=1)

        # Validation consumes only ranking. Integer-spaced values avoid any
        # accidental tie at the Top-K boundary while retaining that exact order.
        rank_values = torch.arange(
            num_targets,
            0,
            -1,
            device=composition_logits.device,
            dtype=composition_logits.dtype,
        ).unsqueeze(0).expand(num_queries, -1)
        merged_scores = torch.empty_like(composition_logits)
        merged_scores.scatter_(1, final_order, rank_values)
        return merged_scores

    def forward(self, samples):
        modification_text = samples["modification_text"]
        reference_bbox = samples.get("reference_bbox", None)
        if reference_bbox is None:
            raise ValueError("oacir_box_context requires reference_bbox")

        (
            reference_raw,
            target_raw,
            reference_embeds,
            target_embeds,
            batch_size,
        ) = self._encode_images_from_samples(samples)

        fusion_features = self._compute_fusion_features(
            reference_embeds,
            modification_text,
            reference_bbox,
        )
        target_features = self._compute_target_features_from_embeds(target_embeds)

        sim_comp = self._compute_composition_scores(fusion_features, target_features)
        sim_comp_logits = sim_comp / self.temp

        targets = torch.arange(batch_size, dtype=torch.long, device=reference_embeds.device)
        candidate_indices = self._build_candidate_indices(
            sim_comp_logits,
            topk=self.core_matcher_topk,
            force_positive=True,
        )
        matcher_scores = self._compute_box_matcher_candidate_scores(
            reference_raw,
            target_raw,
            reference_bbox,
            fusion_features,
            candidate_indices,
            return_heatmap=False,
        )
        candidate_targets = (
            candidate_indices == targets.unsqueeze(1)
        ).long().argmax(dim=1)
        final_logits = matcher_scores["score_final"] / self.core_matcher_temp
        identity_logits = matcher_scores["score_id"] / self.core_matcher_temp
        composition_logits = matcher_scores["score_comp"] / self.core_matcher_temp
        backbone_candidate_logits = torch.gather(
            sim_comp_logits,
            dim=1,
            index=candidate_indices,
        )
        reranking_logits = (
            backbone_candidate_logits
            + float(self.core_matcher_fusion_weight) * final_logits
        )

        return {
            "loss_align": F.cross_entropy(sim_comp_logits, targets),
            "loss_core_matcher": F.cross_entropy(
                reranking_logits,
                candidate_targets,
            ),
            "loss_core_comp": F.cross_entropy(
                composition_logits,
                candidate_targets,
            ),
            "loss_core_id": F.cross_entropy(
                identity_logits,
                candidate_targets,
            ),
        }

    @torch.no_grad()
    def extract_target_features(self, image, mode="tokens"):
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_features = self._compute_target_features_from_embeds(image_embeds)

        return image_features, image_embeds

    @torch.no_grad()
    def extract_target_features_with_raw(self, image, mode="tokens"):
        with self.maybe_autocast():
            raw_embeds = self.visual_encoder(image)
            image_embeds = self.ln_vision(raw_embeds)
        image_embeds = image_embeds.float()
        image_features = self._compute_target_features_from_embeds(image_embeds)

        if mode == "mean":
            image_features = F.normalize(image_features.mean(dim=1), dim=-1)

        return image_features, raw_embeds

    @torch.no_grad()
    def extract_target_features_from_raw(self, raw_embeds, mode="tokens"):
        """Build current Q-Former gallery features from cached frozen-ViT tokens."""
        device = self._get_device()
        raw_embeds = raw_embeds.to(device, non_blocking=True)

        with self.maybe_autocast():
            image_embeds = self.ln_vision(
                raw_embeds.to(dtype=self.ln_vision.weight.dtype)
            )
            image_features = self._compute_target_features_from_embeds(image_embeds)

        if mode == "mean":
            image_features = F.normalize(image_features.mean(dim=1), dim=-1)
        return image_features

    @torch.no_grad()
    def inference(self, reference_image_embeds, target_features, modification_text, reference_bbox=None):
        if target_features.device != reference_image_embeds.device:
            target_features = target_features.to(reference_image_embeds.device)

        fusion_features = self._compute_fusion_features(
            reference_image_embeds,
            modification_text,
            reference_bbox,
        )
        return self._compute_composition_scores(fusion_features, target_features)

    @torch.no_grad()
    def inference_composition_from_raw(
        self,
        reference_image_embeds_raw,
        target_features,
        modification_text,
        reference_bbox=None,
    ):
        """Compute AdaFocal logits without running a latent matcher."""
        device = self._get_device()
        reference_image_embeds_raw = reference_image_embeds_raw.to(
            device,
            non_blocking=True,
        )
        target_features = target_features.to(device, non_blocking=True)
        reference_image_embeds = self.ln_vision(
            reference_image_embeds_raw.to(dtype=self.ln_vision.weight.dtype)
        )
        fusion_features = self._compute_fusion_features(
            reference_image_embeds,
            modification_text,
            reference_bbox,
        )
        return self._compute_composition_scores(fusion_features, target_features) / self.temp

    @torch.no_grad()
    def inference_core_matcher_candidates(
        self,
        reference_image_embeds_raw,
        target_candidate_embeds_raw,
        composition_candidate_logits,
        modification_text,
        reference_bbox,
        return_parts=False,
        return_heatmap=False,
    ):
        """Rerank already gathered global AdaFocal Top-K candidates."""
        device = self._get_device()
        reference_image_embeds_raw = reference_image_embeds_raw.to(
            device,
            non_blocking=True,
        )
        reference_image_embeds = self.ln_vision(
            reference_image_embeds_raw.to(dtype=self.ln_vision.weight.dtype)
        )
        composition_features = self._compute_fusion_features(
            reference_image_embeds,
            modification_text,
            reference_bbox,
        )
        matcher_scores = self._compute_box_matcher_candidate_scores(
            reference_image_embeds_raw,
            target_candidate_embeds_raw,
            reference_bbox,
            composition_features,
            candidate_indices=None,
            return_heatmap=return_heatmap,
        )
        core_matcher_logits = matcher_scores["score_final"] / self.core_matcher_temp
        composition_candidate_logits = composition_candidate_logits.to(
            device=device,
            dtype=core_matcher_logits.dtype,
            non_blocking=True,
        )
        reranked_logits = (
            composition_candidate_logits
            + float(self.core_matcher_fusion_weight) * core_matcher_logits
        )

        if not return_parts:
            return reranked_logits

        outputs = {
            "sim_final": reranked_logits,
            "sim_core_matcher": core_matcher_logits,
            "sim_id": matcher_scores["score_id"] / self.core_matcher_temp,
            "sim_region_comp": matcher_scores["score_comp"] / self.core_matcher_temp,
            # Compatibility alias consumed by the existing evaluator.
            "fusion_scalar": matcher_scores["adapter_ratio"],
            "query_adapter_ratio": matcher_scores["adapter_ratio"],
            "cycle_ratio": matcher_scores["cycle_ratio"],
            "region_entropy": matcher_scores["entropy"],
        }
        if return_heatmap:
            outputs["target_patch_heatmap"] = matcher_scores["target_attention"]
            outputs["reference_patch_attention"] = matcher_scores[
                "reference_attention"
            ]
        return outputs

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        max_txt_len = cfg.get("max_txt_len", 32)
        caam_hidden_dim = cfg.get("caam_hidden_dim", 512)
        num_probe_token = cfg.get("num_probe_token", 8)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
            caam_hidden_dim=caam_hidden_dim,
            num_probe_token=num_probe_token,
        )
        model.load_checkpoint_from_config(cfg)
        return model
