import math
import torch
import torch.nn.functional as F


def infer_patch_grid(num_tokens: int, has_cls_token: bool = True):
    """
    Infer patch grid size from visual tokens.

    For BLIP-2 ViT-G with 224 input:
        num_tokens = 257
        patch tokens = 256
        grid = 16 x 16
    """
    num_patch_tokens = num_tokens - 1 if has_cls_token else num_tokens
    grid_size = int(math.sqrt(num_patch_tokens))

    if grid_size * grid_size != num_patch_tokens:
        raise ValueError(
            f"Cannot infer square patch grid from num_tokens={num_tokens}, "
            f"num_patch_tokens={num_patch_tokens}"
        )

    return grid_size, num_patch_tokens


def _to_xyxy(box, bbox_format="xyxy"):
    """
    box can be list / tuple / tensor.
    Default assumes [x1, y1, x2, y2].
    If bbox_format='xywh', converts [x, y, w, h] to [x1, y1, x2, y2].
    """
    if isinstance(box, torch.Tensor):
        box = box.detach().cpu().tolist()

    x1, y1, x2, y2 = [float(v) for v in box[:4]]

    if bbox_format == "xywh":
        x2 = x1 + x2
        y2 = y1 + y2

    return x1, y1, x2, y2


def bbox_to_patch_mask(
    bboxes,
    num_tokens: int,
    image_size: int = 224,
    has_cls_token: bool = True,
    bbox_format: str = "xyxy",
    device=None,
):
    """
    Convert transformed reference bboxes to patch masks.

    Args:
        bboxes: list of bbox, each bbox is [x1,y1,x2,y2] in 224x224 transformed image space.
        num_tokens: visual token number, e.g. 257.
        image_size: usually 224.
    Returns:
        patch_mask: [B, num_patch_tokens], bool.
    """
    grid_size, num_patch_tokens = infer_patch_grid(num_tokens, has_cls_token=has_cls_token)
    patch_size = image_size / grid_size

    masks = []

    for box in bboxes:
        mask = torch.zeros(num_patch_tokens, dtype=torch.bool)

        if box is None:
            # fallback: use all patches
            mask[:] = True
            masks.append(mask)
            continue

        x1, y1, x2, y2 = _to_xyxy(box, bbox_format=bbox_format)

        # sanitize
        x1 = max(0.0, min(float(image_size), x1))
        y1 = max(0.0, min(float(image_size), y1))
        x2 = max(0.0, min(float(image_size), x2))
        y2 = max(0.0, min(float(image_size), y2))

        if x2 <= x1 or y2 <= y1:
            mask[:] = True
            masks.append(mask)
            continue

        px1 = int(math.floor(x1 / patch_size))
        py1 = int(math.floor(y1 / patch_size))
        px2 = int(math.ceil(x2 / patch_size)) - 1
        py2 = int(math.ceil(y2 / patch_size)) - 1

        px1 = max(0, min(grid_size - 1, px1))
        py1 = max(0, min(grid_size - 1, py1))
        px2 = max(0, min(grid_size - 1, px2))
        py2 = max(0, min(grid_size - 1, py2))

        for yy in range(py1, py2 + 1):
            for xx in range(px1, px2 + 1):
                mask[yy * grid_size + xx] = True

        if not mask.any():
            mask[:] = True

        masks.append(mask)

    patch_mask = torch.stack(masks, dim=0)

    if device is not None:
        patch_mask = patch_mask.to(device)

    return patch_mask


def compute_bidirectional_region_scores(
    reference_region_tokens,
    target_candidate_tokens,
    composed_features,
    region_sizes,
    spatial_kernel_size: int = 3,
    spatial_weight: float = 0.15,
    region_temperature: float = 0.07,
    cycle_weight: float = 0.25,
    text_weight: float = 0.25,
    return_heatmap: bool = False,
):
    """Matcher-style bidirectional discovery over per-query target candidates.

    Args:
        reference_region_tokens: [B, R, D] CORE semantic region tokens.
        target_candidate_tokens: [B, C, N, D] target patch tokens.
        composed_features: [B, D] CORE visual-text composed representation.
        region_sizes: [B] adaptive number of target patches to retain.

    Returns:
        A dictionary containing identity/composition scores and the discovered
        latent-region features for every query-candidate pair.
    """
    if reference_region_tokens.dim() != 3:
        raise ValueError("reference_region_tokens must have shape [B, R, D]")
    if target_candidate_tokens.dim() != 4:
        raise ValueError("target_candidate_tokens must have shape [B, C, N, D]")
    if spatial_kernel_size < 1 or spatial_kernel_size % 2 == 0:
        raise ValueError("spatial_kernel_size must be a positive odd integer")
    if region_temperature <= 0:
        raise ValueError("region_temperature must be positive")

    reference_region_tokens = F.normalize(reference_region_tokens, dim=-1)
    target_candidate_tokens = F.normalize(target_candidate_tokens, dim=-1)
    composed_features = F.normalize(composed_features, dim=-1)

    batch_size, num_candidates, num_target_patches, _ = target_candidate_tokens.shape
    num_reference_regions = reference_region_tokens.size(1)
    if reference_region_tokens.size(0) != batch_size:
        raise ValueError("Reference and candidate batch sizes must match")

    similarities = torch.einsum(
        "brd,bcnd->bcrn",
        reference_region_tokens,
        target_candidate_tokens,
    )
    forward_values, forward_indices = similarities.max(dim=-1)
    reverse_values, reverse_indices = similarities.max(dim=-2)

    # A target patch is cycle-consistent when its best reference token maps
    # back to that same target patch. This is the differentiable adaptation of
    # Matcher's forward/reverse assignment; indices are latent selections.
    roundtrip_target_indices = torch.gather(
        forward_indices,
        dim=2,
        index=reverse_indices,
    )
    target_indices = torch.arange(
        num_target_patches,
        device=similarities.device,
    ).view(1, 1, -1)
    cycle_mask = roundtrip_target_indices.eq(target_indices)

    patch_composition = torch.einsum(
        "bd,bcnd->bcn",
        composed_features,
        target_candidate_tokens,
    )
    response = (
        reverse_values
        + float(text_weight) * patch_composition
        + float(cycle_weight) * cycle_mask.to(reverse_values.dtype)
    )

    grid_size = int(math.sqrt(num_target_patches))
    has_spatial_grid = grid_size * grid_size == num_target_patches
    if has_spatial_grid and spatial_kernel_size > 1:
        response = F.avg_pool2d(
            F.pad(
                response.reshape(-1, 1, grid_size, grid_size),
                (spatial_kernel_size // 2,) * 4,
                mode="replicate",
            ),
            kernel_size=spatial_kernel_size,
            stride=1,
        ).reshape(batch_size, num_candidates, num_target_patches)

    selected_mask = torch.zeros_like(response, dtype=torch.bool)
    if has_spatial_grid:
        yy, xx = torch.meshgrid(
            torch.arange(grid_size, device=response.device),
            torch.arange(grid_size, device=response.device),
            indexing="ij",
        )
        patch_coords = torch.stack([yy.flatten(), xx.flatten()], dim=-1).to(response.dtype)

    for query_index in range(batch_size):
        region_size = int(region_sizes[query_index].item())
        region_size = min(max(region_size, 1), num_target_patches)
        region_rank = response[query_index]

        if has_spatial_grid and region_size < num_target_patches:
            peak_indices = region_rank.argmax(dim=-1)
            peak_coords = patch_coords[peak_indices]
            distance_sq = (
                patch_coords.unsqueeze(0) - peak_coords.unsqueeze(1)
            ).pow(2).sum(dim=-1)
            distance_sq = distance_sq / max(float((grid_size - 1) ** 2), 1.0)
            region_rank = region_rank - float(spatial_weight) * distance_sq

        selected_indices = region_rank.topk(region_size, dim=-1).indices
        selected_mask[query_index].scatter_(1, selected_indices, True)

    min_value = torch.finfo(response.dtype).min
    region_logits = (response / float(region_temperature)).masked_fill(
        ~selected_mask,
        min_value,
    )
    region_weights = F.softmax(region_logits, dim=-1)
    region_features = torch.einsum(
        "bcn,bcnd->bcd",
        region_weights,
        target_candidate_tokens,
    )
    region_features = F.normalize(region_features, dim=-1)

    # Bidirectional Chamfer provides coverage (reference -> region) and purity
    # (region -> reference); cycle-consistent matches add reliability.
    region_similarities = similarities.masked_fill(
        ~selected_mask.unsqueeze(2),
        min_value,
    )
    coverage = region_similarities.max(dim=-1).values.mean(dim=-1)
    purity = (region_weights * reverse_values).sum(dim=-1)
    cycle_values = (reverse_values * cycle_mask.to(reverse_values.dtype)).sum(dim=-1)
    cycle_count = cycle_mask.sum(dim=-1).clamp_min(1).to(reverse_values.dtype)
    cycle_similarity = cycle_values / cycle_count
    identity_scores = (
        coverage + purity + float(cycle_weight) * cycle_similarity
    ) / (2.0 + float(cycle_weight))

    composition_scores = torch.einsum(
        "bd,bcd->bc",
        composed_features,
        region_features,
    )

    background_mask = ~selected_mask
    background_count = background_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    background_features = (
        target_candidate_tokens
        * background_mask.unsqueeze(-1).to(target_candidate_tokens.dtype)
    ).sum(dim=-2) / background_count.to(target_candidate_tokens.dtype)
    background_features = F.normalize(background_features, dim=-1)
    background_scores = torch.einsum(
        "bd,bcd->bc",
        composed_features,
        background_features,
    )

    eps = torch.finfo(region_weights.dtype).eps
    region_entropy = -(
        region_weights.clamp_min(eps) * region_weights.clamp_min(eps).log()
    ).sum(dim=-1)
    normalizer = math.log(max(num_target_patches, 2))
    region_entropy = region_entropy / normalizer

    outputs = {
        "score_id": identity_scores,
        "score_comp": composition_scores,
        "score_background": background_scores,
        "region_features": region_features,
        "cycle_ratio": cycle_mask.to(reverse_values.dtype).sum(dim=-1)
        / float(max(num_reference_regions, 1)),
        "entropy": region_entropy,
    }
    if return_heatmap:
        outputs["target_attention"] = region_weights
    return outputs
