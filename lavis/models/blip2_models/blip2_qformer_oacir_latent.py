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

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    disabled_train,
)
from lavis.models.blip2_models.oacir_latent_matching import (
    bbox_to_patch_mask as latent_bbox_to_patch_mask,
    compute_anchor_target_maxsim,
    compute_anchor_target_ot,
)


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


@registry.register_model("oacir_latent")
class Blip2QformerOacirLatent(Blip2Base):
    """
    AdaFocal backbone plus target-side latent region discovery for OACIR.

    The composition branch restores AdaFocal's CAAM/CRM reference-side
    attention bias. The latent branch adds target-side BICM/MaxSim/OT evidence.

    Final score:
        S_final = S_adafocal + residual latent evidence.
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
        self.ot_ref_dustbin_mlp = self._build_ot_scalar_head(embed_dim)
        self.ot_target_dustbin_mlp = self._build_ot_scalar_head(embed_dim)
        self.ot_ref_vote_mlp = self._build_ot_scalar_head(embed_dim)
        self.bicm_identity_query = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.bicm_anchor_key = nn.Linear(embed_dim, embed_dim)
        self.bicm_identity_query_proj = nn.Linear(embed_dim, embed_dim)
        self.bicm_target_key = nn.Linear(embed_dim, embed_dim)
        self.bicm_target_value = nn.Linear(embed_dim, embed_dim)
        self.bicm_ref_edit_proj = nn.Linear(embed_dim, embed_dim)
        self.bicm_target_edit_proj = nn.Linear(embed_dim, embed_dim)
        self.bicm_text_edit_proj = nn.Linear(embed_dim, embed_dim)
        self.bicm_preserve_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self._init_identity_linear(self.bicm_identity_query_proj)
        self._init_identity_linear(self.bicm_target_value)
        self._init_identity_linear(self.bicm_ref_edit_proj)
        self._init_identity_linear(self.bicm_target_edit_proj)
        nn.init.zeros_(self.bicm_preserve_mlp[-1].weight)
        nn.init.zeros_(self.bicm_preserve_mlp[-1].bias)

        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.global_token_weights = nn.Parameter(torch.ones(num_query_token))
        self.global_margin = 0.1
        self.global_hard_weight = 1.0
        self.global_hard_topk = 16
        self.temp_ins = 0.07
        self.lambda_ins = 0.2
        self.latent_matcher = "bicm"
        self.latent_chunk_size = 256
        self.ot_sinkhorn_iters = 20
        self.ot_temperature = 0.07
        self.ot_vote_temperature = 0.07
        self.ot_text_weight = nn.Parameter(torch.log(torch.tensor(0.1)))
        self.ot_bbox_gamma = nn.Parameter(torch.log(torch.tensor(1.0)))
        self.ot_topk = 50
        self.bicm_temp = 0.07
        self.bicm_lambda_id = 0.5
        self.bicm_lambda_ent = 0.01
        self.bicm_mask_floor = 0.1
        self.bicm_fusion_weight = 0.1
        self.region_topk = 16
        self.region_topk_max = 96
        self.region_area_scale = 1.5
        self.region_spatial_kernel = 3
        self.region_spatial_weight = 0.15
        self.region_temperature = 0.07
        self.spatial_variance_margin = 0.08
        self.typed_comp_gamma = 0.5
        self.latent_image_size = img_size or 224
        self.latent_bbox_format = "xyxy"
        self.use_latent_matching = True
        self.use_typed_contrastive = True
        self.return_aux_losses = False
        self.requires_reference_bbox = True

        self.max_txt_len = max_txt_len
        self.num_query_token = num_query_token

    @staticmethod
    def _build_ot_scalar_head(embed_dim):
        head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        return head

    @staticmethod
    def _init_identity_linear(layer):
        nn.init.eye_(layer.weight)
        nn.init.zeros_(layer.bias)

    def set_latent_config(
        self,
        lambda_ins=None,
        temp_ins=None,
        latent_matcher=None,
        latent_chunk_size=None,
        ot_sinkhorn_iters=None,
        ot_temperature=None,
        ot_vote_temperature=None,
        ot_text_weight=None,
        ot_bbox_gamma=None,
        ot_topk=None,
        bicm_temp=None,
        bicm_lambda_id=None,
        bicm_lambda_ent=None,
        bicm_mask_floor=None,
        bicm_fusion_weight=None,
        region_topk=None,
        region_topk_max=None,
        region_area_scale=None,
        region_spatial_kernel=None,
        region_spatial_weight=None,
        region_temperature=None,
        spatial_variance_margin=None,
        typed_comp_gamma=None,
        global_margin=None,
        global_hard_weight=None,
        global_hard_topk=None,
        use_typed_contrastive=None,
        return_aux_losses=None,
    ):
        if lambda_ins is not None:
            self.lambda_ins = float(lambda_ins)
        if temp_ins is not None:
            self.temp_ins = float(temp_ins)
        if latent_matcher is not None:
            if latent_matcher not in {"maxsim", "ot", "bicm"}:
                raise ValueError("latent_matcher must be 'maxsim', 'ot', or 'bicm'")
            self.latent_matcher = latent_matcher
        if latent_chunk_size is not None:
            self.latent_chunk_size = int(latent_chunk_size)
        if ot_sinkhorn_iters is not None:
            self.ot_sinkhorn_iters = int(ot_sinkhorn_iters)
        if ot_temperature is not None:
            self.ot_temperature = float(ot_temperature)
        if ot_vote_temperature is not None:
            self.ot_vote_temperature = float(ot_vote_temperature)
        if ot_text_weight is not None:
            with torch.no_grad():
                value = max(float(ot_text_weight), 1e-8)
                self.ot_text_weight.fill_(torch.log(torch.tensor(value)).item())
        if ot_bbox_gamma is not None:
            with torch.no_grad():
                value = max(float(ot_bbox_gamma), 1e-8)
                self.ot_bbox_gamma.fill_(torch.log(torch.tensor(value)).item())
        if ot_topk is not None:
            self.ot_topk = int(ot_topk)
        if bicm_temp is not None:
            self.bicm_temp = float(bicm_temp)
        if bicm_lambda_id is not None:
            self.bicm_lambda_id = float(bicm_lambda_id)
        if bicm_lambda_ent is not None:
            self.bicm_lambda_ent = float(bicm_lambda_ent)
        if bicm_mask_floor is not None:
            self.bicm_mask_floor = float(bicm_mask_floor)
        if bicm_fusion_weight is not None:
            self.bicm_fusion_weight = float(bicm_fusion_weight)
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
        if spatial_variance_margin is not None:
            self.spatial_variance_margin = float(spatial_variance_margin)
        if typed_comp_gamma is not None:
            self.typed_comp_gamma = float(typed_comp_gamma)
        if global_margin is not None:
            self.global_margin = float(global_margin)
        if global_hard_weight is not None:
            self.global_hard_weight = float(global_hard_weight)
        if global_hard_topk is not None:
            self.global_hard_topk = int(global_hard_topk)
        if use_typed_contrastive is not None:
            self.use_typed_contrastive = bool(use_typed_contrastive)
        if return_aux_losses is not None:
            self.return_aux_losses = bool(return_aux_losses)

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

    def _compute_text_features(self, modification_text, device):
        text_tokens = self._tokenize_text(modification_text, device)
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

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
            use_cache=True,
            return_dict=True,
        )

        return F.normalize(self.vision_proj(target_output.last_hidden_state), dim=-1)

    def _compute_composition_scores(self, fusion_features, target_features):
        if target_features.dtype != fusion_features.dtype:
            target_features = target_features.to(dtype=fusion_features.dtype)
        if target_features.dim() == 2:
            return torch.matmul(fusion_features, target_features.t())
        return torch.einsum("bd,nqd->bnq", fusion_features, target_features).max(dim=-1).values

    def _compute_adaptive_cosine_loss(
        self,
        fusion_features,
        target_features,
        hard_negative_scores=None,
    ):
        if target_features.dtype != fusion_features.dtype:
            target_features = target_features.to(dtype=fusion_features.dtype)
        if target_features.dim() == 2:
            target_pooled = target_features
        else:
            num_tokens = target_features.size(1)
            token_weights = self.global_token_weights[:num_tokens].to(
                dtype=target_features.dtype,
                device=target_features.device,
            )
            target_pooled = (
                target_features * token_weights.view(1, num_tokens, 1)
            ).sum(dim=1) / float(num_tokens)

        target_pooled = F.normalize(target_pooled, dim=-1)
        fusion_features = F.normalize(fusion_features, dim=-1)
        cosine_matrix = torch.matmul(fusion_features, target_pooled.t())
        batch_size = cosine_matrix.size(0)
        targets = torch.arange(batch_size, dtype=torch.long, device=cosine_matrix.device)
        positive_cosine = cosine_matrix[targets, targets]
        positive_loss = (1.0 - positive_cosine).mean()

        if batch_size <= 1 or self.global_hard_weight <= 0:
            return positive_loss

        if hard_negative_scores is None:
            hard_negative_scores = cosine_matrix.detach()
        else:
            hard_negative_scores = hard_negative_scores.detach().to(cosine_matrix.device)

        hard_negative_scores = hard_negative_scores.clone()
        hard_negative_scores[targets, targets] = torch.finfo(hard_negative_scores.dtype).min
        hard_topk = min(max(int(self.global_hard_topk), 1), batch_size - 1)
        hard_indices = hard_negative_scores.topk(hard_topk, dim=1).indices
        hard_negative_cosine = torch.gather(cosine_matrix, dim=1, index=hard_indices)
        hard_loss = F.relu(
            float(self.global_margin)
            + hard_negative_cosine
            - positive_cosine.unsqueeze(1)
        ).mean()

        return positive_loss + float(self.global_hard_weight) * hard_loss

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

    def _compute_instance_scores(
        self,
        reference_image_embeds_raw,
        target_image_embeds_raw,
        reference_bbox,
        text_features=None,
        return_heatmap=False,
    ):
        batch_size = reference_image_embeds_raw.size(0)
        device = reference_image_embeds_raw.device

        reference_inst_tokens = self._project_raw_patch_tokens(reference_image_embeds_raw)
        target_inst_tokens = self._project_raw_patch_tokens(target_image_embeds_raw)
        anchor_mask = self._build_anchor_mask(
            reference_bbox=reference_bbox,
            num_tokens=reference_image_embeds_raw.size(1),
            batch_size=batch_size,
            device=device,
        )

        if self.latent_matcher == "ot":
            if text_features is None:
                raise ValueError("text_features is required when latent_matcher='ot'")
            return compute_anchor_target_ot(
                reference_inst_tokens,
                target_inst_tokens,
                anchor_mask,
                text_features,
                self.ot_ref_dustbin_mlp(reference_inst_tokens).squeeze(-1),
                self.ot_target_dustbin_mlp(target_inst_tokens).squeeze(-1),
                self.ot_ref_vote_mlp(reference_inst_tokens).squeeze(-1),
                chunk_size=self.latent_chunk_size,
                return_heatmap=return_heatmap,
                sinkhorn_iters=self.ot_sinkhorn_iters,
                ot_temperature=self.ot_temperature,
                vote_temperature=self.ot_vote_temperature,
                text_weight=self.ot_text_weight.exp(),
                bbox_gamma=self.ot_bbox_gamma.exp(),
            )

        return compute_anchor_target_maxsim(
            reference_inst_tokens,
            target_inst_tokens,
            anchor_mask,
            chunk_size=self.latent_chunk_size,
            return_heatmap=return_heatmap,
            region_topk=self.region_topk,
            region_topk_max=self.region_topk_max,
            region_area_scale=self.region_area_scale,
            spatial_kernel_size=self.region_spatial_kernel,
            spatial_weight=self.region_spatial_weight,
        )

    def _build_ot_candidate_indices(self, sim_comp_logits):
        """Select top-K backbone candidates per query and force the positive in."""
        batch_size = sim_comp_logits.size(0)
        topk = min(max(int(self.ot_topk), 1), batch_size)
        if topk >= batch_size:
            return None

        with torch.no_grad():
            candidate_indices = sim_comp_logits.detach().topk(topk, dim=1).indices
            targets = torch.arange(
                batch_size,
                dtype=torch.long,
                device=sim_comp_logits.device,
            )
            has_positive = (candidate_indices == targets.unsqueeze(1)).any(dim=1)
            if not has_positive.all():
                candidate_indices = candidate_indices.clone()
                candidate_indices[~has_positive, -1] = targets[~has_positive]

        return candidate_indices

    def _compute_candidate_instance_scores(
        self,
        reference_image_embeds_raw,
        target_image_embeds_raw,
        reference_bbox,
        text_features,
        candidate_indices,
    ):
        batch_size = reference_image_embeds_raw.size(0)
        device = reference_image_embeds_raw.device

        if self.latent_matcher != "ot":
            sim_ins = self._compute_instance_scores(
                reference_image_embeds_raw,
                target_image_embeds_raw,
                reference_bbox,
                text_features=text_features,
                return_heatmap=False,
            )
            return torch.gather(sim_ins, dim=1, index=candidate_indices)

        reference_inst_tokens = self._project_raw_patch_tokens(reference_image_embeds_raw)
        target_inst_tokens = self._project_raw_patch_tokens(target_image_embeds_raw)
        anchor_mask = self._build_anchor_mask(
            reference_bbox=reference_bbox,
            num_tokens=reference_image_embeds_raw.size(1),
            batch_size=batch_size,
            device=device,
        )

        ref_dustbin_logits = self.ot_ref_dustbin_mlp(reference_inst_tokens).squeeze(-1)
        target_dustbin_logits = self.ot_target_dustbin_mlp(target_inst_tokens).squeeze(-1)
        ref_vote_logits = self.ot_ref_vote_mlp(reference_inst_tokens).squeeze(-1)

        rows = []
        for qi in range(batch_size):
            cur_indices = candidate_indices[qi]
            cur_scores = compute_anchor_target_ot(
                reference_inst_tokens[qi : qi + 1],
                target_inst_tokens.index_select(0, cur_indices),
                anchor_mask[qi : qi + 1],
                text_features[qi : qi + 1],
                ref_dustbin_logits[qi : qi + 1],
                target_dustbin_logits.index_select(0, cur_indices),
                ref_vote_logits[qi : qi + 1],
                chunk_size=self.latent_chunk_size,
                return_heatmap=False,
                sinkhorn_iters=self.ot_sinkhorn_iters,
                ot_temperature=self.ot_temperature,
                vote_temperature=self.ot_vote_temperature,
                text_weight=self.ot_text_weight.exp(),
                bbox_gamma=self.ot_bbox_gamma.exp(),
            )
            rows.append(cur_scores.squeeze(0))

        return torch.stack(rows, dim=0)

    def _encode_box_identity_anchor(self, reference_patch_tokens, anchor_mask):
        anchor_logits = torch.matmul(
            self.bicm_anchor_key(reference_patch_tokens),
            self.bicm_identity_query.to(dtype=reference_patch_tokens.dtype),
        )
        anchor_logits = anchor_logits / math.sqrt(reference_patch_tokens.size(-1))
        anchor_logits = anchor_logits.masked_fill(
            ~anchor_mask.bool(),
            torch.finfo(anchor_logits.dtype).min,
        )
        anchor_weights = F.softmax(anchor_logits, dim=-1)
        z_ref_id = torch.einsum(
            "bn,bnd->bd",
            anchor_weights,
            reference_patch_tokens,
        )
        return F.normalize(z_ref_id, dim=-1), anchor_weights

    def _compute_preservation_mask(self, text_features):
        mask = torch.sigmoid(self.bicm_preserve_mlp(text_features))
        floor = min(max(float(self.bicm_mask_floor), 0.0), 1.0)
        return floor + (1.0 - floor) * mask

    def _compute_bicm_scores(
        self,
        reference_image_embeds_raw,
        target_image_embeds_raw,
        reference_bbox,
        text_features,
        return_heatmap=False,
    ):
        batch_size = reference_image_embeds_raw.size(0)
        device = reference_image_embeds_raw.device

        reference_tokens = F.normalize(
            self._project_raw_patch_tokens(reference_image_embeds_raw),
            dim=-1,
        )
        target_tokens = F.normalize(
            self._project_raw_patch_tokens(target_image_embeds_raw),
            dim=-1,
        )
        anchor_mask = self._build_anchor_mask(
            reference_bbox=reference_bbox,
            num_tokens=reference_image_embeds_raw.size(1),
            batch_size=batch_size,
            device=device,
        )

        z_ref_id, _ = self._encode_box_identity_anchor(reference_tokens, anchor_mask)
        preserve_mask = self._compute_preservation_mask(text_features)
        target_keys = F.normalize(self.bicm_target_key(target_tokens), dim=-1)
        target_values = self.bicm_target_value(target_tokens)

        id_query = F.normalize(self.bicm_identity_query_proj(z_ref_id), dim=-1)
        text_edit = F.normalize(self.bicm_text_edit_proj(text_features), dim=-1)
        z_ref_edit = self.bicm_ref_edit_proj(z_ref_id)

        score_chunks = []
        id_chunks = []
        edit_chunks = []
        entropy_chunks = []
        heatmap_chunks = []

        for start in range(0, target_tokens.size(0), self.latent_chunk_size):
            end = min(start + self.latent_chunk_size, target_tokens.size(0))
            cur_keys = target_keys[start:end]
            cur_values = target_values[start:end]

            attention_logits = torch.einsum("bd,cnd->bcn", id_query, cur_keys)
            attention_logits = attention_logits / math.sqrt(cur_keys.size(-1))
            target_attention = F.softmax(attention_logits, dim=-1)
            z_target_id = torch.einsum(
                "bcn,cnd->bcd",
                target_attention,
                cur_values,
            )
            z_target_id = F.normalize(z_target_id, dim=-1)

            masked_ref = F.normalize(
                preserve_mask.unsqueeze(1) * z_ref_id.unsqueeze(1),
                dim=-1,
            )
            masked_target = F.normalize(
                preserve_mask.unsqueeze(1) * z_target_id,
                dim=-1,
            )
            score_id = (masked_ref * masked_target).sum(dim=-1)

            z_target_edit = self.bicm_target_edit_proj(z_target_id)
            visual_delta = F.normalize(
                z_target_edit - z_ref_edit.unsqueeze(1),
                dim=-1,
            )
            score_edit = torch.einsum("bcd,bd->bc", visual_delta, text_edit)

            eps = torch.finfo(target_attention.dtype).eps
            entropy = -(
                target_attention.clamp_min(eps)
                * target_attention.clamp_min(eps).log()
            ).sum(dim=-1)
            entropy = entropy / math.log(target_attention.size(-1))

            score_final = (
                score_edit
                + float(self.bicm_lambda_id) * score_id
                - float(self.bicm_lambda_ent) * entropy
            )
            score_chunks.append(score_final)
            id_chunks.append(score_id)
            edit_chunks.append(score_edit)
            entropy_chunks.append(entropy)

            if return_heatmap:
                heatmap_chunks.append(target_attention)

        outputs = {
            "score_final": torch.cat(score_chunks, dim=1),
            "score_id": torch.cat(id_chunks, dim=1),
            "score_edit": torch.cat(edit_chunks, dim=1),
            "entropy": torch.cat(entropy_chunks, dim=1),
        }
        if return_heatmap:
            outputs["target_attention"] = torch.cat(heatmap_chunks, dim=1)
        return outputs

    @staticmethod
    def _masked_multi_positive_nce(logits, positive_mask, negative_mask):
        """InfoNCE over explicitly typed positives and negatives."""
        positive_mask = positive_mask.bool()
        negative_mask = negative_mask.bool() & ~positive_mask
        valid_rows = positive_mask.any(dim=1) & negative_mask.any(dim=1)

        if not valid_rows.any():
            return logits.sum() * 0.0

        selected_mask = positive_mask | negative_mask
        min_value = torch.finfo(logits.dtype).min
        numerator = torch.logsumexp(
            logits.masked_fill(~positive_mask, min_value),
            dim=1,
        )
        denominator = torch.logsumexp(
            logits.masked_fill(~selected_mask, min_value),
            dim=1,
        )
        return (denominator[valid_rows] - numerator[valid_rows]).mean()

    @staticmethod
    def _build_typed_masks(instance_ids, object_categories, device):
        batch_size = len(instance_ids)
        same_instance = torch.tensor(
            [
                [instance_ids[i] == instance_ids[j] for j in range(batch_size)]
                for i in range(batch_size)
            ],
            dtype=torch.bool,
            device=device,
        )
        same_category = torch.tensor(
            [
                [object_categories[i] == object_categories[j] for j in range(batch_size)]
                for i in range(batch_size)
            ],
            dtype=torch.bool,
            device=device,
        )
        diagonal = torch.eye(batch_size, dtype=torch.bool, device=device)

        return {
            "exact_positive": diagonal,
            "instance_positive": same_instance,
            "wrong_composition": same_instance & ~diagonal,
            "wrong_instance": same_category & ~same_instance,
            "mixed_negative": ~same_category & ~same_instance,
        }

    def _compute_spatial_consistency_loss(self, heatmaps):
        """Encourage the positive target response to form one compact region."""
        batch_size, _, num_patches = heatmaps.shape
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size != num_patches:
            return heatmaps.sum() * 0.0

        positive_heatmaps = heatmaps[
            torch.arange(batch_size, device=heatmaps.device),
            torch.arange(batch_size, device=heatmaps.device),
        ]
        temperature = max(float(self.region_temperature), 1e-4)
        probabilities = F.softmax(positive_heatmaps / temperature, dim=-1)

        coordinates = torch.linspace(
            0.0,
            1.0,
            grid_size,
            device=heatmaps.device,
            dtype=probabilities.dtype,
        )
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        coords = torch.stack([yy.flatten(), xx.flatten()], dim=-1)
        centers = torch.einsum("bn,nd->bd", probabilities, coords)
        distance_sq = (
            coords.unsqueeze(0) - centers.unsqueeze(1)
        ).pow(2).sum(dim=-1)
        spatial_variance = (probabilities * distance_sq).sum(dim=-1)
        margin = float(self.spatial_variance_margin)
        return F.relu(spatial_variance - margin).mean()

    def forward(self, samples):
        modification_text = samples["modification_text"]
        reference_bbox = samples.get("reference_bbox", None)
        instance_ids = samples.get("target_instance_ids")
        object_categories = samples.get("object_categories")

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
        text_features = self._compute_text_features(modification_text, reference_embeds.device)
        target_features = self._compute_target_features_from_embeds(target_embeds)

        sim_comp = self._compute_composition_scores(fusion_features, target_features)
        sim_comp_logits = sim_comp / self.temp

        targets = torch.arange(batch_size, dtype=torch.long, device=reference_embeds.device)
        loss_comp_global = F.cross_entropy(sim_comp_logits, targets)
        loss_global = self._compute_adaptive_cosine_loss(
            fusion_features,
            target_features,
            hard_negative_scores=sim_comp,
        )
        typed_masks = None
        if (
            self.use_typed_contrastive
            and instance_ids is not None
            and object_categories is not None
        ):
            typed_masks = self._build_typed_masks(
                instance_ids,
                object_categories,
                reference_embeds.device,
            )
            loss_comp_typed = self._masked_multi_positive_nce(
                sim_comp_logits,
                typed_masks["exact_positive"],
                typed_masks["wrong_composition"],
            )
            loss_comp = loss_comp_global + self.typed_comp_gamma * loss_comp_typed
        else:
            loss_comp = loss_comp_global

        loss_dict = {
            "loss_comp": loss_comp,
            "loss_global": loss_global,
        }

        if self.use_latent_matching and reference_bbox is not None:
            loss_align = loss_comp
            loss_dict["loss_align"] = loss_align

            if self.latent_matcher == "bicm":
                bicm_scores = self._compute_bicm_scores(
                    reference_raw,
                    target_raw,
                    reference_bbox,
                    text_features,
                    return_heatmap=False,
                )
                bicm_branch_logits = bicm_scores["score_final"] / self.bicm_temp
                bicm_id_logits = bicm_scores["score_id"] / self.bicm_temp
                bicm_edit_logits = bicm_scores["score_edit"] / self.bicm_temp
                positive_entropy = bicm_scores["entropy"][
                    torch.arange(batch_size, device=reference_embeds.device),
                    targets,
                ]

                loss_dict["loss_final"] = F.cross_entropy(
                    bicm_branch_logits,
                    targets,
                )
                loss_dict["loss_id"] = F.cross_entropy(bicm_id_logits, targets)
                loss_dict["loss_edit"] = F.cross_entropy(bicm_edit_logits, targets)
                loss_dict["loss_ent"] = positive_entropy.mean()
                return loss_dict

            candidate_indices = self._build_ot_candidate_indices(sim_comp_logits)
            if candidate_indices is None or self.return_aux_losses:
                instance_output = self._compute_instance_scores(
                    reference_raw,
                    target_raw,
                    reference_bbox,
                    text_features=text_features,
                    return_heatmap=self.return_aux_losses,
                )
                if self.return_aux_losses:
                    sim_ins, region_heatmaps = instance_output
                else:
                    sim_ins = instance_output
                    region_heatmaps = None
                sim_ins_logits = sim_ins / self.temp_ins
                loss_ot = F.cross_entropy(sim_ins_logits, targets)
            else:
                candidate_sim_ins = self._compute_candidate_instance_scores(
                    reference_raw,
                    target_raw,
                    reference_bbox,
                    text_features,
                    candidate_indices,
                )
                sim_ins_logits = candidate_sim_ins / self.temp_ins
                candidate_targets = (
                    candidate_indices == targets.unsqueeze(1)
                ).long().argmax(dim=1)
                loss_ot = F.cross_entropy(sim_ins_logits, candidate_targets)
                region_heatmaps = None

            loss_dict["loss_ot"] = loss_ot
            if self.return_aux_losses:
                if typed_masks is not None:
                    loss_ins = self._masked_multi_positive_nce(
                        sim_ins_logits,
                        typed_masks["instance_positive"],
                        typed_masks["wrong_instance"],
                    )
                else:
                    loss_ins = F.cross_entropy(sim_ins_logits, targets)
                loss_spatial = self._compute_spatial_consistency_loss(region_heatmaps)
                loss_dict["loss_ins"] = loss_ins
                loss_dict["loss_spatial"] = loss_spatial
        else:
            loss_dict["loss_align"] = loss_comp

        return loss_dict

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
    def inference_with_latent_matching(
        self,
        reference_image_embeds_raw,
        target_features,
        target_image_embeds_raw,
        modification_text,
        reference_bbox=None,
        return_parts=False,
        return_heatmap=False,
    ):
        device = self._get_device()
        reference_image_embeds_raw = reference_image_embeds_raw.to(device, non_blocking=True)
        target_image_embeds_raw = target_image_embeds_raw.to(device, non_blocking=True)
        target_features = target_features.to(device, non_blocking=True)

        reference_image_embeds = self.ln_vision(reference_image_embeds_raw.to(dtype=self.ln_vision.weight.dtype))
        fusion_features = self._compute_fusion_features(
            reference_image_embeds,
            modification_text,
            reference_bbox,
        )
        text_features = self._compute_text_features(modification_text, device)

        sim_comp = self._compute_composition_scores(fusion_features, target_features)
        sim_comp_logits = sim_comp / self.temp

        if self.latent_matcher == "bicm":
            bicm_scores = self._compute_bicm_scores(
                reference_image_embeds_raw,
                target_image_embeds_raw,
                reference_bbox,
                text_features,
                return_heatmap=return_heatmap,
            )
            bicm_branch_logits = bicm_scores["score_final"] / self.bicm_temp
            sim_final_logits = (
                sim_comp_logits
                + float(self.bicm_fusion_weight) * bicm_branch_logits
            )
            if return_parts:
                outputs = {
                    "sim_final": sim_final_logits,
                    "sim_comp": sim_comp_logits,
                    "sim_bicm": bicm_branch_logits,
                    "sim_id": bicm_scores["score_id"] / self.bicm_temp,
                    "sim_edit": bicm_scores["score_edit"] / self.bicm_temp,
                    "attention_entropy": bicm_scores["entropy"],
                }
                if return_heatmap:
                    outputs["target_patch_heatmap"] = bicm_scores["target_attention"]
                return outputs

            return sim_final_logits

        instance_output = self._compute_instance_scores(
            reference_image_embeds_raw,
            target_image_embeds_raw,
            reference_bbox,
            text_features=text_features,
            return_heatmap=return_heatmap,
        )
        if return_heatmap:
            sim_ins, heatmap = instance_output
        else:
            sim_ins = instance_output
            heatmap = None

        sim_ins_logits = sim_ins / self.temp_ins
        sim_final_logits = sim_comp_logits + self.lambda_ins * sim_ins_logits

        if return_parts:
            outputs = {
                "sim_final": sim_final_logits,
                "sim_comp": sim_comp_logits,
                "sim_ins": sim_ins_logits,
            }
            if heatmap is not None:
                outputs["target_patch_heatmap"] = heatmap
            return outputs

        return sim_final_logits

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
