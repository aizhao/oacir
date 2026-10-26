"""Independent split-mask ablation: selectively edit local slots with text.

Install this ONE new file in ``lavis/models/blip2_models/``. Requires the
existing latent and split_mask modules; neither is modified. Registration:
``oacir_latent_slot_film``.

The existing CORE region slots are reused (normally 16, NOT target-patch
top-k). A per-query gate chooses up to four slots and applies an additive,
FiLM-inspired text residual. Identity matching uses the original unedited
slots and the existing text-free M_id. Stage 1, candidate selection, losses,
target-patch counts, and score coefficients are inherited from split_mask.
No pose estimator, semantic part labels, DDL, or new identity loss is added.

The residual output starts at zero, preserving baseline stage-2 scores in
eval mode when the shared baseline weights match. This does NOT freeze the
shared encoder during training or guarantee improved identity recall.

No train.py / __init__.py change is needed if launched as:
    python lavis/models/blip2_models/blip2_qformer_oacir_latent_slot_film.py \
        --dataset Union --validation-domains Fashion Car Landmark [train args]
The script registers itself, then runs the existing train.py. For evaluation,
use the same entry with ``--entry evaluate.py`` and its normal arguments.

Inspired by CA-ReID, sec. 3.5. Unlike that paper, these are learned latent
regions, not pose-supervised body parts; sparse gating is our adaptation.
"""

from pathlib import Path
import runpy
import sys


def _repo_root():
    """Find the installed repository without changing any repository files."""
    source = Path(__file__).resolve()
    candidates = [source.parents[3]] if len(source.parents) > 3 else []
    candidates.append(Path.cwd().resolve())
    for root in candidates:
        if (root / "train.py").is_file() and (root / "lavis").is_dir():
            return root
    raise RuntimeError("Run inside the OACIR repo or install under lavis/models/blip2_models/")


if __name__ == "__main__":
    sys.path.insert(0, str(_repo_root()))

import torch
from torch import nn
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2_qformer_oacir_latent_split_mask import (
    Blip2QformerOacirLatentSplitMask,
)


class SelectiveSlotFiLM(nn.Module):
    """Sparse local residual, not canonical multiplicative gamma*x + beta FiLM.

    Each query is processed independently. Top-k is over reference SLOTS,
    not target patches or batch samples. A zero permission mask guarantees
    no edit, but permissions are not inferred from semantic part names here.
    """

    def __init__(self, embed_dim, hidden_dim=128, active_slots=4):
        super().__init__()
        if embed_dim < 1 or hidden_dim < 1 or active_slots < 1:
            raise ValueError("Dimensions and active_slots must be positive")
        self.embed_dim = int(embed_dim)
        self.active_slots = int(active_slots)
        self.enabled = True
        self.gate = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.residual = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim, bias=False),
        )
        nn.init.zeros_(self.residual[-1].weight)

    def forward(self, slots, text_features, permission_mask=None):
        if slots.ndim != 3 or slots.shape[1] < 1 or slots.shape[-1] != self.embed_dim:
            raise ValueError("slots must be nonempty [B,R,embed_dim]")
        if text_features.shape != (slots.shape[0], self.embed_dim):
            raise ValueError("text_features must have shape [B,embed_dim]")
        if permission_mask is not None and permission_mask.shape != slots.shape[:2]:
            raise ValueError("permission_mask must have shape [B,R]")
        if not self.enabled:
            return slots, slots.new_zeros(slots.shape[:2])

        text = F.normalize(text_features.to(dtype=slots.dtype), dim=-1)
        text = text[:, None, :].expand(-1, slots.shape[1], -1)
        features = torch.cat([F.normalize(slots, dim=-1), text], dim=-1)
        logits = self.gate(features).squeeze(-1)
        allowed = torch.ones_like(logits, dtype=torch.bool)
        if permission_mask is not None:
            allowed = permission_mask.to(device=slots.device, dtype=torch.bool)
        selected = torch.zeros_like(allowed)
        indices = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min).topk(
            min(self.active_slots, slots.shape[1]), dim=1
        ).indices
        selected.scatter_(1, indices, True)
        selected = selected & allowed
        gates = (logits.sigmoid() * selected.to(logits.dtype)).to(slots.dtype)
        delta = self.residual(features).to(slots.dtype)
        return slots + gates.unsqueeze(-1) * delta, gates


@registry.register_model("oacir_latent_slot_film")
class Blip2QformerOacirLatentSlotFiLM(Blip2QformerOacirLatentSplitMask):
    """Keep split-mask identity; text edits composition slots only."""

    requires_identity_labels = False  # Inherit the baseline CE, not multi-positive.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_slot_film = SelectiveSlotFiLM(self.instance_proj.out_features)

    def _compute_core_reference_features(self, reference_image_embeds_raw,
                                         reference_bbox, text_features):
        # Same projection, bbox prior and reference encoder as the baseline.
        patches = self._project_raw_patch_tokens(reference_image_embeds_raw)
        anchor_mask = self._build_anchor_mask(
            reference_bbox=reference_bbox,
            num_tokens=reference_image_embeds_raw.size(1),
            batch_size=reference_image_embeds_raw.size(0),
            device=patches.device,
        )
        identity_regions, reference_feature, attention = self.core_reference_encoder(
            patches, anchor_mask, bbox_floor=self.core_matcher_bbox_floor
        )
        # Recover unnormalized pooled slots so zero residual exactly preserves
        # the baseline's mean-before-normalize aggregation, not mean(unit slots).
        raw_slots = torch.einsum("brn,bnd->brd", attention, patches)
        edited_slots, edit_gates = self.local_slot_film(raw_slots, text_features)
        if self.local_slot_film.enabled:
            composition_regions = F.normalize(edited_slots, dim=-1)
            composition_reference = F.normalize(edited_slots.mean(dim=1), dim=-1)
        else:
            composition_regions = identity_regions
            composition_reference = reference_feature
        # Reuse the original global fusion ONCE; do not add another global gate.
        composed, scalar = self.core_adaptive_fusion(composition_reference, text_features)

        count = patches.size(1)
        minimum = min(max(int(self.region_topk), 1), count)
        maximum = min(max(int(self.region_topk_max), minimum), count)
        sizes = (anchor_mask.sum(-1).float() * float(self.region_area_scale))
        sizes = sizes.round().long().clamp(min=minimum, max=maximum)
        return {
            "region_tokens": composition_regions,
            "identity_region_tokens": identity_regions,
            "composed_feature": composed,
            "reference_attention": attention,
            "fusion_scalar": scalar,
            "region_sizes": sizes,
            # Inspectable via this method; inherited matcher return keys stay intact.
            "slot_edit_gates": edit_gates,
            "unedited_reference_feature": reference_feature,
        }


def _entry_arguments(arguments):
    """Consume only our launcher option; forward ordinary train/eval flags."""
    arguments = list(arguments)
    entry = "train.py"
    if "--entry" in arguments:
        position = arguments.index("--entry")
        if position + 1 == len(arguments):
            raise ValueError("--entry needs train.py or evaluate.py")
        entry = arguments[position + 1]
        del arguments[position:position + 2]
    if entry not in {"train.py", "evaluate.py"}:
        raise ValueError("--entry must be train.py or evaluate.py")
    expected = "oacir_latent_slot_film"
    names = []
    for position, argument in enumerate(arguments):
        if argument == "--blip-model-name":
            if position + 1 == len(arguments):
                raise ValueError("--blip-model-name needs a value")
            names.append(arguments[position + 1])
        elif argument.startswith("--blip-model-name="):
            names.append(argument.split("=", 1)[1])
    if names and any(name != expected for name in names):
        raise ValueError("This independent entry only launches " + expected)
    if not names:
        arguments += ["--blip-model-name", expected]
    return entry, arguments


if __name__ == "__main__":
    root = _repo_root()
    entry, arguments = _entry_arguments(sys.argv[1:])
    # Avoid duplicate registry registration if an existing entry lazily imports
    # our canonical module name. This affects this process only, never old files.
    sys.modules["lavis.models.blip2_models.blip2_qformer_oacir_latent_slot_film"] = sys.modules[__name__]
    sys.argv = [str(root / entry)] + arguments
    runpy.run_path(str(root / entry), run_name="__main__")
