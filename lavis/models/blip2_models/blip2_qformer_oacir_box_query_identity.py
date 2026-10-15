"""Box-query composition with a dedicated CORE identity encoder."""

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2_qformer_oacir_box_query import (
    Blip2QformerOacirBoxQuery,
)
from lavis.models.blip2_models.blip2_qformer_oacir_latent import (
    CoreReferenceRegionEncoder,
)


@registry.register_model("oacir_box_query_identity")
class Blip2QformerOacirBoxQueryIdentity(Blip2QformerOacirBoxQuery):
    """Decouple identity preservation from box-query composition discovery.

    A parallel, validated CORE reference encoder can be routed independently
    into composition, target-region discovery, and the identity score.  The
    default route preserves the trained box-query composition/discovery path.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        embed_dim = self.instance_proj.out_features
        self.identity_reference_encoder = CoreReferenceRegionEncoder(
            embed_dim=embed_dim,
            num_regions=16,
        )
        self.identity_route = "id_only"

    def set_identity_route(self, route):
        valid_routes = {
            "id_only",
            "discovery_id",
            "latent_comp_id",
            "latent_all",
        }
        if route not in valid_routes:
            raise ValueError(
                f"Unsupported identity route '{route}', expected one of "
                f"{sorted(valid_routes)}"
            )
        self.identity_route = route

    def _compute_core_reference_features(
        self,
        reference_image_embeds_raw,
        reference_bbox,
        text_features,
    ):
        outputs = super()._compute_core_reference_features(
            reference_image_embeds_raw,
            reference_bbox,
            text_features,
        )
        reference_tokens = self._project_raw_patch_tokens(
            reference_image_embeds_raw
        )
        anchor_mask = self._build_anchor_mask(
            reference_bbox=reference_bbox,
            num_tokens=reference_image_embeds_raw.size(1),
            batch_size=reference_image_embeds_raw.size(0),
            device=reference_image_embeds_raw.device,
        )
        identity_tokens, identity_feature, identity_attention = (
            self.identity_reference_encoder(
                reference_tokens,
                anchor_mask,
                bbox_floor=self.core_matcher_bbox_floor,
            )
        )
        outputs["identity_region_tokens"] = identity_tokens
        outputs["identity_reference_feature"] = identity_feature
        outputs["identity_reference_attention"] = identity_attention

        if self.identity_route in {"latent_comp_id", "latent_all"}:
            # Route the latent pooled feature through the checkpoint's trained
            # adaptive fusion while retaining the box-query output dictionary.
            latent_composed_feature, latent_fusion_scalar = (
                self.core_adaptive_fusion(identity_feature, text_features)
            )
            outputs["reference_feature"] = identity_feature
            outputs["composed_feature"] = latent_composed_feature
            outputs["fusion_scalar"] = latent_fusion_scalar

        if self.identity_route in {"discovery_id", "latent_all"}:
            # Let the latent CORE tokens drive target-patch discovery in
            # addition to the final identity score.
            outputs["region_tokens"] = identity_tokens
            outputs["reference_attention"] = identity_attention
        return outputs
