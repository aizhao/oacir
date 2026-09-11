#!/bin/bash

# ==============================================================================
# Evaluation Script for OACIRR and Standard CIR Datasets
# ==============================================================================

# -------------------------- Core Settings --------------------------
# Dataset to evaluate on. Choices: ['Fashion', 'Car', 'Product', 'Landmark', 'CIRR', 'FashionIQ']
DATASET="Car"

# Root directory of the dataset
DATA_ROOT="./Datasets/OACIRR"

# Model registry name. Choices: ['oacir_baseline', 'oacir_adafocal', 'oacir_adafocal_vector']
# MODEL_NAME="oacir_adafocal"

MODEL_NAME="oacir_latent"

USE_LATENT_MATCHING="--use-latent-matching"
LAMBDA_INS=0.05
TEMP_INS=0.07
LATENT_CHUNK_SIZE=256
LATENT_MATCHER="bicm"
OT_SINKHORN_ITERS=20
OT_TEMPERATURE=0.07
OT_VOTE_TEMPERATURE=0.07
OT_TEXT_WEIGHT=0.1
OT_BBOX_GAMMA=1.0
BICM_TEMP=0.07
BICM_LAMBDA_ID=0.5
BICM_LAMBDA_ENT=0.01
BICM_MASK_FLOOR=0.1
BICM_FUSION_WEIGHT=0.1
LATENT_GALLERY_CHUNK_SIZE=1024
VAL_FEATURE_CACHE="./cache/oacirr_{variant_lower}_val_vitg_targetpad125_raw_fp16.pt"
VAL_FEATURE_BATCH_SIZE=32
REGION_TOPK=16
REGION_TOPK_MAX=96
REGION_AREA_SCALE=1.5
REGION_SPATIAL_KERNEL=3
REGION_SPATIAL_WEIGHT=0.15
REGION_TEMPERATURE=0.07
# Path to the fine-tuned model weight to be evaluated (REQUIRED)
# e.g., "./checkpoints/OACIR_Union_finetune_.../saved_models/adafocal_finetune_best.pt"
MODEL_WEIGHT="/home/caoyu/mnt/zhaoai/OACIR/checkpoints/OACIR_Union_finetune_oacir_latent_2026-07-10_16:55:44/saved_models/adafocal_finetune_best.pt"

# ViT backbone. Choices: ['pretrain' (ViT-G), 'pretrain_vitL' (ViT-L)]
VIT_BACKBONE="pretrain"


# ------------------ Image Preprocessing Settings -------------------
# Transform type. Choices: ['squarepad', 'targetpad']
TRANSFORM="targetpad"
TARGET_RATIO=1.25


# -------------------- Visual Baseline Settings ---------------------
# To evaluate the Visual Anchor Baseline (Drawing Bbox on image), set BBOX_WIDTH > 0 (e.g., 3)
BBOX_WIDTH=0
BBOX_COLOR="red"

# To evaluate the ROI-Crop Baseline (Cropping the Bbox region), uncomment the line below:
# BBOX_CROP="--bounding-box-crop"


# -------------------------- Boolean Flags --------------------------
# AdaFocal mechanism (Keep uncommented if evaluating AdaFocal)
HIGHLIGHT_INFERENCE="--highlight-inference"

# Save the detailed retrieval results as JSON for visualization
# SAVE_RESULTS="--save-results"

# Hardware/Memory options
SAVE_MEMORY="--save-memory"

# Text Prompt
# TEXT_ENTITY="--text-entity"


# ==============================================================================
# Execute Python Script
# ==============================================================================

echo "Starting Evaluation on ${DATASET} using model ${MODEL_NAME}..."

python evaluate.py \
    --dataset ${DATASET} \
    --data-root ${DATA_ROOT} \
    --blip-model-name ${MODEL_NAME} \
    --blip-model-weight ${MODEL_WEIGHT} \
    --vit-backbone ${VIT_BACKBONE} \
    --transform ${TRANSFORM} \
    --target-ratio ${TARGET_RATIO} \
    --bounding-box-width ${BBOX_WIDTH} \
    --bounding-box-color ${BBOX_COLOR} \
    ${BBOX_CROP} \
    ${HIGHLIGHT_INFERENCE} \
    ${SAVE_RESULTS} \
    ${SAVE_MEMORY} \
    ${TEXT_ENTITY} \
    ${USE_LATENT_MATCHING} \
    --lambda-ins "$LAMBDA_INS" \
    --temp-ins "$TEMP_INS" \
    --latent-chunk-size "$LATENT_CHUNK_SIZE" \
    --latent-matcher "$LATENT_MATCHER" \
    --ot-sinkhorn-iters "$OT_SINKHORN_ITERS" \
    --ot-temperature "$OT_TEMPERATURE" \
    --ot-vote-temperature "$OT_VOTE_TEMPERATURE" \
    --ot-text-weight "$OT_TEXT_WEIGHT" \
    --ot-bbox-gamma "$OT_BBOX_GAMMA" \
    --bicm-temp "$BICM_TEMP" \
    --bicm-lambda-id "$BICM_LAMBDA_ID" \
    --bicm-lambda-ent "$BICM_LAMBDA_ENT" \
    --bicm-mask-floor "$BICM_MASK_FLOOR" \
    --bicm-fusion-weight "$BICM_FUSION_WEIGHT" \
    --latent-gallery-chunk-size "$LATENT_GALLERY_CHUNK_SIZE" \
    --val-feature-cache "$VAL_FEATURE_CACHE" \
    --val-feature-batch-size "$VAL_FEATURE_BATCH_SIZE" \
    --region-topk "$REGION_TOPK" \
    --region-topk-max "$REGION_TOPK_MAX" \
    --region-area-scale "$REGION_AREA_SCALE" \
    --region-spatial-kernel "$REGION_SPATIAL_KERNEL" \
    --region-spatial-weight "$REGION_SPATIAL_WEIGHT" \
    --region-temperature "$REGION_TEMPERATURE"
