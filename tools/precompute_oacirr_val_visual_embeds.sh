#!/bin/bash
set -euo pipefail

DATA_ROOT="./Datasets/OACIRR"
MODEL_NAME="oacir_latent"
VIT_BACKBONE="pretrain"
TRANSFORM="targetpad"
TARGET_RATIO=1.25
BATCH_SIZE=128
NUM_WORKERS=6
CACHE_DIR="./cache"

mkdir -p "$CACHE_DIR"

for VARIANT in Fashion Car Product Landmark; do
    VARIANT_LOWER="${VARIANT,,}"
    OUTPUT="${CACHE_DIR}/oacirr_${VARIANT_LOWER}_val_vitg_targetpad125_raw_fp16.pt"

    echo "Precomputing OACIRR ${VARIANT} val raw visual embeddings..."
    python tools/precompute_oacirr_train_visual_embeds.py \
        --dataset "$VARIANT" \
        --split val \
        --data-root "$DATA_ROOT" \
        --blip-model-name "$MODEL_NAME" \
        --vit-backbone "$VIT_BACKBONE" \
        --transform "$TRANSFORM" \
        --target-ratio "$TARGET_RATIO" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --output "$OUTPUT"
done

echo "All OACIRR validation caches are ready in ${CACHE_DIR}."
