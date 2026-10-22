"""Ablation 3: original OACIR latent model with a simple identity branch.

The composition route remains exactly the original latent route.  Identity
matching alone receives eight text-free BBox-grid tokens.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2_qformer_oacir_latent import (
    Blip2QformerOacirLatent,
)


class BBoxIdentityBranch(nn.Module):
    """Pool BBox patches into a 2 x 4 grid without text or Q-Former."""

    def __init__(self, embed_dim, num_regions=8):
        super().__init__()
        if int(num_regions) != 8:
            raise ValueError("BBoxIdentityBranch requires num_regions=8")
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, patch_tokens, anchor_mask):
        batch_size, num_patches, _ = patch_tokens.shape
        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            raise ValueError("identity branch requires a square patch grid")

        anchor_mask = anchor_mask.to(device=patch_tokens.device, dtype=torch.bool)
        batch_weights = []
        for batch_index in range(batch_size):
            mask_2d = anchor_mask[batch_index].reshape(grid_size, grid_size)
            coords = mask_2d.nonzero(as_tuple=False)
            if coords.numel() == 0:
                mask_2d = torch.ones_like(mask_2d)
                coords = mask_2d.nonzero(as_tuple=False)

            y_min, x_min = coords.min(dim=0).values
            y_max, x_max = coords.max(dim=0).values + 1
            height = max(int((y_max - y_min).item()), 1)
            width = max(int((x_max - x_min).item()), 1)
            fallback = mask_2d.flatten().to(patch_tokens.dtype)
            fallback = fallback / fallback.sum().clamp_min(1.0)

            region_weights = []
            for row in range(2):
                for col in range(4):
                    y0 = int(y_min.item()) + row * height // 2
                    y1 = int(y_min.item()) + (row + 1) * height // 2
                    x0 = int(x_min.item()) + col * width // 4
                    x1 = int(x_min.item()) + (col + 1) * width // 4
                    y1, x1 = max(y1, y0 + 1), max(x1, x0 + 1)
                    region = torch.zeros_like(mask_2d)
                    region[y0:y1, x0:x1] = True
                    weights = (region & mask_2d).flatten().to(patch_tokens.dtype)
                    weights = (
                        fallback if weights.sum() == 0 else weights / weights.sum()
                    )
                    region_weights.append(weights)
            batch_weights.append(torch.stack(region_weights))

        attention = torch.stack(batch_weights)
        pooled = torch.einsum("brn,bnd->brd", attention, patch_tokens)
        tokens = pooled + self.proj(self.norm(pooled))
        return F.normalize(tokens, dim=-1), attention


@registry.register_model("oacir_latent_dual_branch")
class Blip2QformerOacirLatentDualBranch(Blip2QformerOacirLatent):
    """Add only the independent BBox identity-token route."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.identity_branch = BBoxIdentityBranch(
            self.instance_proj.out_features, num_regions=8
        )

    def _compute_core_reference_features(
        self, reference_image_embeds_raw, reference_bbox, text_features
    ):
        outputs = super()._compute_core_reference_features(
            reference_image_embeds_raw, reference_bbox, text_features
        )
        reference_tokens = self._project_raw_patch_tokens(reference_image_embeds_raw)
        anchor_mask = self._build_anchor_mask(
            reference_bbox=reference_bbox,
            num_tokens=reference_image_embeds_raw.size(1),
            batch_size=reference_image_embeds_raw.size(0),
            device=reference_image_embeds_raw.device,
        )
        identity_tokens, identity_attention = self.identity_branch(
            reference_tokens, anchor_mask
        )
        outputs["identity_region_tokens"] = identity_tokens
        outputs["identity_reference_attention"] = identity_attention
        return outputs

