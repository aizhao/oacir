import math
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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


def compute_anchor_target_maxsim(
    reference_patch_tokens,
    target_patch_tokens,
    anchor_mask,
    chunk_size: int = 256,
    return_heatmap: bool = False,
    region_topk: int = 16,
    region_topk_max: int = 96,
    region_area_scale: float = 1.5,
    spatial_kernel_size: int = 3,
    spatial_weight: float = 0.15,
):
    """
    Compute anchor-to-target latent instance matching.

    Args:
        reference_patch_tokens: [Bq, Nr, D]
        target_patch_tokens:    [Bt, Nt, D]
        anchor_mask:            [Bq, Nr], bool
    Returns:
        sim_matrix: [Bq, Bt]
    """
    if spatial_kernel_size < 1 or spatial_kernel_size % 2 == 0:
        raise ValueError("spatial_kernel_size must be a positive odd integer")
    if region_topk < 0:
        raise ValueError("region_topk must be non-negative")
    if region_topk_max < region_topk:
        raise ValueError("region_topk_max must be greater than or equal to region_topk")
    if region_area_scale <= 0:
        raise ValueError("region_area_scale must be positive")

    reference_patch_tokens = F.normalize(reference_patch_tokens.float(), dim=-1)
    target_patch_tokens = F.normalize(target_patch_tokens.float(), dim=-1)

    Bq, Nr, D = reference_patch_tokens.shape
    Bt, Nt, _ = target_patch_tokens.shape

    rows = []
    use_explicit_region = region_topk > 0
    need_spatial_map = use_explicit_region or spatial_kernel_size > 1 or return_heatmap
    keep_heatmaps = return_heatmap
    heatmaps = [] if return_heatmap else None

    if need_spatial_map:
        grid_size = int(math.sqrt(Nt))
        if grid_size * grid_size != Nt:
            raise ValueError(f"Cannot build a spatial region from {Nt} target patches")
        yy, xx = torch.meshgrid(
            torch.arange(grid_size, device=target_patch_tokens.device),
            torch.arange(grid_size, device=target_patch_tokens.device),
            indexing="ij",
        )
        patch_coords = torch.stack([yy.flatten(), xx.flatten()], dim=-1).float()

    for qi in range(Bq):
        cur_anchor = reference_patch_tokens[qi][anchor_mask[qi]]

        if cur_anchor.numel() == 0:
            cur_anchor = reference_patch_tokens[qi]

        adaptive_topk = Nt
        if use_explicit_region:
            adaptive_topk = int(round(cur_anchor.size(0) * float(region_area_scale)))
            adaptive_topk = max(int(region_topk), adaptive_topk)
            adaptive_topk = min(int(region_topk_max), adaptive_topk)
            adaptive_topk = min(Nt, adaptive_topk)

        cur_scores = []
        cur_heatmaps = []

        for start in range(0, Bt, chunk_size):
            end = min(start + chunk_size, Bt)
            cur_targets = target_patch_tokens[start:end]  # [C, Nt, D]

            # sim: [C, Na, Nt]
            sim = torch.einsum("ad,cnd->can", cur_anchor, cur_targets)

            # First discover a target-side response map, then constrain MaxSim
            # to one spatially coherent latent region around its strongest peak.
            target_heat = sim.max(dim=1).values  # [C, Nt]
            region_heat = target_heat

            if spatial_kernel_size > 1:
                region_heat_2d = region_heat.reshape(-1, 1, grid_size, grid_size)
                spatial_padding = spatial_kernel_size // 2
                region_heat_2d = F.pad(
                    region_heat_2d,
                    (spatial_padding,) * 4,
                    mode="replicate",
                )
                region_heat = F.avg_pool2d(
                    region_heat_2d,
                    kernel_size=spatial_kernel_size,
                    stride=1,
                ).flatten(1)

            if use_explicit_region and adaptive_topk < Nt:
                peak_indices = region_heat.argmax(dim=-1)
                peak_coords = patch_coords[peak_indices]
                distance_sq = (
                    patch_coords.unsqueeze(0) - peak_coords.unsqueeze(1)
                ).pow(2).sum(dim=-1)
                distance_sq = distance_sq / max(float((grid_size - 1) ** 2), 1.0)

                region_rank = region_heat - float(spatial_weight) * distance_sq
                topk = adaptive_topk
                selected_indices = region_rank.topk(topk, dim=-1).indices
                region_mask = torch.zeros_like(region_heat, dtype=torch.bool)
                region_mask.scatter_(1, selected_indices, True)

                region_sim = sim.masked_fill(
                    ~region_mask.unsqueeze(1),
                    torch.finfo(sim.dtype).min,
                )
                anchor_to_target = region_sim.max(dim=-1).values
            else:
                anchor_to_target = sim.max(dim=-1).values

            # average over anchor tokens: [C]
            score = anchor_to_target.mean(dim=-1)
            cur_scores.append(score)

            if keep_heatmaps:
                cur_heatmaps.append(region_heat)

        row = torch.cat(cur_scores, dim=0)
        rows.append(row)

        if return_heatmap:
            heatmaps.append(torch.cat(cur_heatmaps, dim=0))

    sim_matrix = torch.stack(rows, dim=0)  # [Bq, Bt]

    if return_heatmap:
        heatmaps = torch.stack(heatmaps, dim=0)  # [Bq, Bt, Nt]
        return sim_matrix, heatmaps

    return sim_matrix


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
        "background_features": background_features,
        "cycle_ratio": cycle_mask.to(reverse_values.dtype).sum(dim=-1)
        / float(max(num_reference_regions, 1)),
        "entropy": region_entropy,
    }
    if return_heatmap:
        outputs["target_attention"] = region_weights
    return outputs


def _log_sinkhorn(log_scores, log_mu, log_nu, iters: int):
    """Differentiable log-domain Sinkhorn normalization."""
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)

    for _ in range(int(iters)):
        u = log_mu - torch.logsumexp(log_scores + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(log_scores + u.unsqueeze(2), dim=1)

    return log_scores + u.unsqueeze(2) + v.unsqueeze(1)


def compute_anchor_target_ot(
    reference_patch_tokens,
    target_patch_tokens,
    anchor_mask,
    text_features,
    ref_dustbin_logits,
    target_dustbin_logits,
    ref_vote_logits,
    chunk_size: int = 256,
    return_heatmap: bool = False,
    sinkhorn_iters: int = 20,
    ot_temperature: float = 0.07,
    vote_temperature: float = 0.07,
    text_weight: float = 0.1,
    bbox_gamma: float = 1.0,
):
    """
    Text-conditioned, bbox-biased differentiable OT matching.

    This implements the full ELViS-style module used by the latent branch:
      1. local visual similarity plus text-to-target-patch prior,
      2. reference/target dustbins,
      3. log-domain Sinkhorn transport,
      4. differentiable soft voting with learned reference-token weights.

    Args:
        reference_patch_tokens: [Bq, Nr, D]
        target_patch_tokens:    [Bt, Nt, D]
        anchor_mask:            [Bq, Nr], bool; reference bbox prior
        text_features:          [Bq, D]
        ref_dustbin_logits:     [Bq, Nr]
        target_dustbin_logits:  [Bt, Nt]
        ref_vote_logits:        [Bq, Nr]
    Returns:
        sim_matrix: [Bq, Bt]
    """
    if sinkhorn_iters <= 0:
        raise ValueError("sinkhorn_iters must be positive")
    if ot_temperature <= 0:
        raise ValueError("ot_temperature must be positive")
    if vote_temperature <= 0:
        raise ValueError("vote_temperature must be positive")

    reference_patch_tokens = F.normalize(reference_patch_tokens.float(), dim=-1)
    target_patch_tokens = F.normalize(target_patch_tokens.float(), dim=-1)
    text_features = F.normalize(text_features.float(), dim=-1)

    ref_dustbin_logits = ref_dustbin_logits.float()
    target_dustbin_logits = target_dustbin_logits.float()
    ref_vote_logits = ref_vote_logits.float()
    anchor_mask = anchor_mask.bool()

    Bq, Nr, _ = reference_patch_tokens.shape
    Bt, Nt, _ = target_patch_tokens.shape
    device = reference_patch_tokens.device
    text_weight = torch.as_tensor(
        text_weight,
        dtype=reference_patch_tokens.dtype,
        device=device,
    )
    bbox_gamma = torch.as_tensor(
        bbox_gamma,
        dtype=reference_patch_tokens.dtype,
        device=device,
    )

    rows = []
    heatmaps = [] if return_heatmap else None
    log_mu = torch.full(
        (1, Nr + 1),
        -math.log(Nr + 1),
        dtype=torch.float32,
        device=device,
    )
    log_nu = torch.full(
        (1, Nt + 1),
        -math.log(Nt + 1),
        dtype=torch.float32,
        device=device,
    )

    for qi in range(Bq):
        ref_tokens = reference_patch_tokens[qi]
        cur_anchor = anchor_mask[qi].float()
        if cur_anchor.sum() <= 0:
            cur_anchor = torch.ones_like(cur_anchor)

        ref_dustbin = ref_dustbin_logits[qi] - bbox_gamma * cur_anchor
        ref_vote = ref_vote_logits[qi] + bbox_gamma * cur_anchor
        ref_weights = F.softmax(ref_vote / float(vote_temperature), dim=-1)

        cur_scores = []
        cur_heatmaps = []

        for start in range(0, Bt, chunk_size):
            end = min(start + chunk_size, Bt)
            cur_targets = target_patch_tokens[start:end]
            cur_target_dustbin = target_dustbin_logits[start:end]

            def score_chunk(
                chunk_targets,
                chunk_target_dustbin,
                chunk_ref_tokens,
                chunk_text_feature,
                chunk_ref_dustbin,
                chunk_ref_weights,
                chunk_text_weight,
            ):
                visual_sim = torch.einsum("rd,cnd->crn", chunk_ref_tokens, chunk_targets)
                text_sim = torch.einsum("d,cnd->cn", chunk_text_feature, chunk_targets)
                sim = visual_sim + chunk_text_weight * text_sim.unsqueeze(1)

                chunk_size_actual = chunk_targets.size(0)
                extended = sim.new_zeros(chunk_size_actual, Nr + 1, Nt + 1)
                extended[:, :Nr, :Nt] = sim
                extended[:, :Nr, Nt] = chunk_ref_dustbin.unsqueeze(0)
                extended[:, Nr, :Nt] = chunk_target_dustbin

                log_transport = _log_sinkhorn(
                    extended / float(ot_temperature),
                    log_mu.expand(chunk_size_actual, -1),
                    log_nu.expand(chunk_size_actual, -1),
                    sinkhorn_iters,
                )
                transport = log_transport[:, :Nr, :Nt].exp()

                # Soft voting keeps the module differentiable while preserving
                # the original visual/text similarity scale for the final score.
                vote_logits = transport * sim
                vote_probs = F.softmax(vote_logits / float(vote_temperature), dim=-1)
                best_match = (vote_probs * sim).sum(dim=-1)
                score = (best_match * chunk_ref_weights.unsqueeze(0)).sum(dim=-1)

                if return_heatmap:
                    heatmap = (
                        transport * chunk_ref_weights.view(1, Nr, 1)
                    ).sum(dim=1)
                    return score, heatmap

                return score

            if torch.is_grad_enabled() and not return_heatmap:
                score = checkpoint(
                    score_chunk,
                    cur_targets,
                    cur_target_dustbin,
                    ref_tokens,
                    text_features[qi],
                    ref_dustbin,
                    ref_weights,
                    text_weight,
                    use_reentrant=False,
                )
            else:
                chunk_output = score_chunk(
                    cur_targets,
                    cur_target_dustbin,
                    ref_tokens,
                    text_features[qi],
                    ref_dustbin,
                    ref_weights,
                    text_weight,
                )
                if return_heatmap:
                    score, chunk_heatmap = chunk_output
                else:
                    score = chunk_output

            cur_scores.append(score)

            if return_heatmap:
                cur_heatmaps.append(chunk_heatmap)

        rows.append(torch.cat(cur_scores, dim=0))
        if return_heatmap:
            heatmaps.append(torch.cat(cur_heatmaps, dim=0))

    sim_matrix = torch.stack(rows, dim=0)

    if return_heatmap:
        heatmaps = torch.stack(heatmaps, dim=0)
        return sim_matrix, heatmaps

    return sim_matrix
