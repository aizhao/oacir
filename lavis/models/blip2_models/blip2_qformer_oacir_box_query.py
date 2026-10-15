"""Box-native region-query variant of the OACIR latent matcher.

This model keeps the validated AdaFocal and candidate-reranking pipeline from
``oacir_latent`` while replacing only its reference region encoder. The new
encoder uses ConvNeXt-V2-style GRN refinement, SAM/DAB-style continuous box
prompts, box-to-patch relative bias, learned region queries, and PMA pooling.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint_sequential

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2_qformer_oacir_latent import (
    Blip2QformerOacirLatent,
)


class GlobalResponseNorm(nn.Module):
    """Global Response Normalization for channels-last feature maps."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = float(eps)

    def forward(self, features):
        response = torch.norm(features, p=2, dim=(1, 2), keepdim=True)
        normalized = response / (response.mean(dim=-1, keepdim=True) + self.eps)
        return features + self.gamma * (features * normalized) + self.beta


class GRNPatchRefinementBlock(nn.Module):
    """ConvNeXt-V2-style spatial refinement with GRN channel competition."""

    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.depthwise = nn.Conv2d(
            dim,
            dim,
            kernel_size=7,
            padding=3,
            groups=dim,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.expand = nn.Linear(dim, expansion * dim)
        self.grn = GlobalResponseNorm(expansion * dim)
        self.project = nn.Linear(expansion * dim, dim)

    def forward(self, feature_map):
        residual = feature_map
        features = self.depthwise(feature_map).permute(0, 2, 3, 1)
        features = self.expand(self.norm(features))
        features = self.grn(F.gelu(features))
        features = self.project(features).permute(0, 3, 1, 2)
        return residual + features


class FourierPositionEncoding(nn.Module):
    """SAM-style random Fourier features for sparse box corner prompts."""

    def __init__(self, num_features: int = 64):
        super().__init__()
        if num_features < 1:
            raise ValueError("num_features must be positive")
        gaussian_matrix = torch.randn(2, num_features, dtype=torch.float32)
        self.register_buffer(
            "gaussian_matrix",
            gaussian_matrix,
            persistent=True,
        )

    @property
    def output_dim(self):
        return 2 * self.gaussian_matrix.size(1)

    def forward(self, coordinates):
        coordinates = 2.0 * coordinates.float() - 1.0
        angles = 2.0 * math.pi * torch.matmul(
            coordinates,
            self.gaussian_matrix,
        )
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class BoxPromptEncoder(nn.Module):
    """Encode box corners, size, and box-to-patch geometry continuously."""

    def __init__(self, embed_dim: int, num_regions: int):
        super().__init__()
        fourier_features = max(embed_dim // 8, 16)
        self.position_encoding = FourierPositionEncoding(fourier_features)
        self.corner_types = nn.Parameter(torch.empty(2, embed_dim))
        self.corner_proj = nn.Linear(self.position_encoding.output_dim, embed_dim)
        self.geometry_proj = nn.Sequential(
            nn.Linear(8, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.prompt_norm = nn.LayerNorm(embed_dim)
        self.relative_bias = nn.Sequential(
            nn.Linear(6, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_regions),
        )
        nn.init.normal_(self.corner_types, std=0.02)

    @staticmethod
    def _as_normalized_boxes(reference_bbox, batch_size, image_size, device, dtype):
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
        boxes = boxes / float(image_size)
        x1 = torch.minimum(boxes[:, 0], boxes[:, 2]).clamp(0.0, 1.0)
        y1 = torch.minimum(boxes[:, 1], boxes[:, 3]).clamp(0.0, 1.0)
        x2 = torch.maximum(boxes[:, 0], boxes[:, 2]).clamp(0.0, 1.0)
        y2 = torch.maximum(boxes[:, 1], boxes[:, 3]).clamp(0.0, 1.0)
        minimum_extent = 1.0 / 1024.0
        x2 = torch.maximum(x2, x1 + minimum_extent).clamp(max=1.0)
        y2 = torch.maximum(y2, y1 + minimum_extent).clamp(max=1.0)
        x1 = torch.minimum(x1, x2 - minimum_extent).clamp(min=0.0)
        y1 = torch.minimum(y1, y2 - minimum_extent).clamp(min=0.0)
        return torch.stack([x1, y1, x2, y2], dim=-1)

    @staticmethod
    def _patch_centers(grid_size, device, dtype):
        coordinates = (
            torch.arange(grid_size, device=device, dtype=dtype) + 0.5
        ) / float(grid_size)
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        return torch.stack([xx.flatten(), yy.flatten()], dim=-1)

    def forward(
        self,
        reference_bbox,
        batch_size,
        grid_size,
        image_size,
        device,
        dtype,
        bbox_floor=0.05,
    ):
        boxes = self._as_normalized_boxes(
            reference_bbox,
            batch_size,
            image_size,
            device,
            dtype,
        )
        top_left = self.corner_proj(self.position_encoding(boxes[:, :2]))
        bottom_right = self.corner_proj(self.position_encoding(boxes[:, 2:]))
        corner_prompt = 0.5 * (
            top_left
            + self.corner_types[0]
            + bottom_right
            + self.corner_types[1]
        )

        width = (boxes[:, 2] - boxes[:, 0]).clamp_min(1e-6)
        height = (boxes[:, 3] - boxes[:, 1]).clamp_min(1e-6)
        center_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
        center_y = 0.5 * (boxes[:, 1] + boxes[:, 3])
        geometry = torch.stack(
            [
                center_x,
                center_y,
                width,
                height,
                width * height,
                torch.log(width),
                torch.log(height),
                torch.log(width / height),
            ],
            dim=-1,
        )
        box_prompt = self.prompt_norm(corner_prompt + self.geometry_proj(geometry))

        patch_centers = self._patch_centers(grid_size, device, dtype)
        patch_x = patch_centers[:, 0].view(1, -1)
        patch_y = patch_centers[:, 1].view(1, -1)
        relative_geometry = torch.stack(
            [
                (patch_x - boxes[:, 0:1]) / width.unsqueeze(1),
                (patch_y - boxes[:, 1:2]) / height.unsqueeze(1),
                (boxes[:, 2:3] - patch_x) / width.unsqueeze(1),
                (boxes[:, 3:4] - patch_y) / height.unsqueeze(1),
                torch.log(width).unsqueeze(1).expand(-1, patch_x.size(1)),
                torch.log(height).unsqueeze(1).expand(-1, patch_y.size(1)),
            ],
            dim=-1,
        )
        learned_bias = self.relative_bias(relative_geometry).permute(0, 2, 1)

        # A differentiable box prior replaces the old binary mask/floor map.
        softness = max(1.0 / float(grid_size), 1e-3)
        soft_inside = (
            torch.sigmoid((patch_x - boxes[:, 0:1]) / softness)
            * torch.sigmoid((boxes[:, 2:3] - patch_x) / softness)
            * torch.sigmoid((patch_y - boxes[:, 1:2]) / softness)
            * torch.sigmoid((boxes[:, 3:4] - patch_y) / softness)
        )
        floor = min(max(float(bbox_floor), 1e-6), 1.0)
        spatial_prior = floor + (1.0 - floor) * soft_inside
        attention_bias = learned_bias + spatial_prior.log().unsqueeze(1)
        return box_prompt, attention_bias


class BoxConditionedRegionAttention(nn.Module):
    """Learned region queries with box-to-patch relative attention bias."""

    def __init__(self, embed_dim: int, num_regions: int, num_heads: int = 8):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = int(embed_dim)
        self.num_regions = int(num_regions)
        self.num_heads = int(num_heads)
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.region_queries = nn.Parameter(torch.empty(num_regions, embed_dim))
        self.box_to_queries = nn.Linear(embed_dim, num_regions * embed_dim)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.query_norm = nn.LayerNorm(embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        nn.init.normal_(self.region_queries, std=0.02)

    def forward(self, patch_tokens, box_prompt, attention_bias):
        batch_size, num_patches, _ = patch_tokens.shape
        queries = self.region_queries.unsqueeze(0) + self.box_to_queries(
            box_prompt
        ).reshape(batch_size, self.num_regions, self.embed_dim)
        queries = self.query_norm(queries)

        query = self.query_proj(queries).reshape(
            batch_size,
            self.num_regions,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        key = self.key_proj(patch_tokens).reshape(
            batch_size,
            num_patches,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        value = self.value_proj(patch_tokens).reshape(
            batch_size,
            num_patches,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)

        logits = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        logits = logits + attention_bias.unsqueeze(1)
        weights = F.softmax(logits, dim=-1)
        attended = torch.matmul(weights, value).permute(0, 2, 1, 3).reshape(
            batch_size,
            self.num_regions,
            self.embed_dim,
        )
        region_tokens = self.output_norm(queries + self.output_proj(attended))
        region_tokens = self.output_norm(region_tokens + self.ffn(region_tokens))
        return region_tokens, weights.mean(dim=1)


class PMAReferencePooling(nn.Module):
    """Pooling by multi-head attention over the discovered reference regions."""

    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.seed = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
        )
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.norm_2 = nn.LayerNorm(embed_dim)
        nn.init.normal_(self.seed, std=0.02)

    def forward(self, region_tokens):
        seed = self.seed.expand(region_tokens.size(0), -1, -1)
        pooled, _ = self.attention(
            seed,
            region_tokens,
            region_tokens,
            need_weights=False,
        )
        pooled = self.norm_1(seed + pooled)
        pooled = self.norm_2(pooled + self.ffn(pooled))
        return F.normalize(pooled[:, 0], dim=-1)


class BoxQueryReferenceEncoder(nn.Module):
    """Box-native GRN/query/PMA replacement for the original CORE RRE."""

    def __init__(
        self,
        embed_dim: int,
        num_regions: int = 16,
        num_heads: int = 8,
        num_refinement_blocks: int = 2,
    ):
        super().__init__()
        self.refinement = nn.Sequential(
            *[
                GRNPatchRefinementBlock(embed_dim)
                for _ in range(num_refinement_blocks)
            ]
        )
        self.box_prompt_encoder = BoxPromptEncoder(embed_dim, num_regions)
        self.region_attention = BoxConditionedRegionAttention(
            embed_dim,
            num_regions,
            num_heads,
        )
        self.pooling = PMAReferencePooling(embed_dim, num_heads)

    def forward(
        self,
        patch_tokens,
        reference_bbox,
        image_size,
        bbox_floor=0.05,
    ):
        batch_size, num_patches, embed_dim = patch_tokens.shape
        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            raise ValueError(
                f"Box-query encoder requires a square patch grid, got {num_patches}"
            )

        feature_map = patch_tokens.transpose(1, 2).reshape(
            batch_size,
            embed_dim,
            grid_size,
            grid_size,
        )
        if self.training and torch.is_grad_enabled():
            refined_map = checkpoint_sequential(
                self.refinement,
                len(self.refinement),
                feature_map,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            refined_map = self.refinement(feature_map)
        refined_tokens = refined_map.flatten(2).transpose(1, 2)

        box_prompt, attention_bias = self.box_prompt_encoder(
            reference_bbox=reference_bbox,
            batch_size=batch_size,
            grid_size=grid_size,
            image_size=image_size,
            device=patch_tokens.device,
            dtype=patch_tokens.dtype,
            bbox_floor=bbox_floor,
        )
        region_tokens, attention_weights = self.region_attention(
            refined_tokens,
            box_prompt,
            attention_bias,
        )
        pooled_feature = self.pooling(region_tokens)
        region_tokens = F.normalize(region_tokens, dim=-1)
        return region_tokens, pooled_feature, attention_weights


@registry.register_model("oacir_box_query")
class Blip2QformerOacirBoxQuery(Blip2QformerOacirLatent):
    """OACIR latent matcher with a box-conditioned region-query encoder."""

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        embed_dim = self.instance_proj.out_features
        self.core_reference_encoder = BoxQueryReferenceEncoder(
            embed_dim=embed_dim,
            num_regions=16,
            num_heads=8,
            num_refinement_blocks=2,
        )

    def _compute_core_reference_features(
        self,
        reference_image_embeds_raw,
        reference_bbox,
        text_features,
    ):
        reference_tokens = self._project_raw_patch_tokens(reference_image_embeds_raw)
        anchor_mask = self._build_anchor_mask(
            reference_bbox=reference_bbox,
            num_tokens=reference_image_embeds_raw.size(1),
            batch_size=reference_image_embeds_raw.size(0),
            device=reference_image_embeds_raw.device,
        )
        region_tokens, reference_feature, reference_attention = (
            self.core_reference_encoder(
                reference_tokens,
                reference_bbox,
                image_size=self.latent_image_size,
                bbox_floor=self.core_matcher_bbox_floor,
            )
        )
        composed_feature, fusion_scalar = self.core_adaptive_fusion(
            reference_feature,
            text_features,
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
            "reference_feature": reference_feature,
            "composed_feature": composed_feature,
            "reference_attention": reference_attention,
            "fusion_scalar": fusion_scalar,
            "region_sizes": region_sizes,
        }
