#!/bin/bash

# ==============================================================================
# Training Script for OACIRR and Standard CIR Datasets
# ==============================================================================

# Avoid hanging on HuggingFace/LAVIS network checks during model initialization.
# The server has already run this model before, so cached tokenizer/checkpoint files
# should be used directly. If a cache is missing, training will fail fast with a
# clear missing-file error instead of blocking silently.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

# -------------------------- Core Settings --------------------------
# Dataset to train on. Choices: ['Fashion', 'Car', 'Product', 'Landmark', 'Union', 'CIRR', 'FashionIQ']
DATASET="Union"

# Root directory of the dataset
DATA_ROOT="./Datasets/OACIRR"

# Model registry name. Choices: ['oacir_baseline', 'oacir_adafocal', 'oacir_adafocal_vector']
# MODEL_NAME="oacir_adafocal"
MODEL_NAME="oacir_latent"
# Path to the pre-trained model weight for initialization (OPTIONAL)
# Leave empty "" to use the default BLIP-2 pretrained Q-Former from LAVIS.
# Provide a path only if you want to resume training from a local checkpoint.
MODEL_WEIGHT=""

# ViT backbone. Choices: ['pretrain' (ViT-G), 'pretrain_vitL' (ViT-L)]
VIT_BACKBONE="pretrain"

# Directory to save checkpoints
SAVE_DIR="./checkpoints"
TRAIN_FEATURE_CACHE="./cache/oacirr_union_train_vitg_targetpad125_raw_fp16.pt"
VAL_FEATURE_CACHE="./cache/oacirr_{variant_lower}_val_vitg_targetpad125_raw_fp16.pt"
VAL_FEATURE_BATCH_SIZE=32
LAMBDA_INS=0.05
TEMP_INS=0.07
LOSS_INS=0.0
LOSS_OT=0.0
LOSS_FINAL=0.5
LOSS_ID=0.1
LOSS_EDIT=0.1
LOSS_ENT=0.0
LATENT_CHUNK_SIZE=64
LATENT_MATCHER="bicm"
OT_SINKHORN_ITERS=10
OT_TEMPERATURE=0.07
OT_VOTE_TEMPERATURE=0.07
OT_TEXT_WEIGHT=0.1
OT_BBOX_GAMMA=1.0
OT_TOPK=50
BICM_TEMP=0.07
BICM_LAMBDA_ID=0.5
BICM_LAMBDA_ENT=0.01
BICM_MASK_FLOOR=0.1
BICM_FUSION_WEIGHT=0.1
REGION_TOPK=16
REGION_TOPK_MAX=96
REGION_AREA_SCALE=1.5
REGION_SPATIAL_KERNEL=3
REGION_SPATIAL_WEIGHT=0.15
REGION_TEMPERATURE=0.07
SPATIAL_VARIANCE_MARGIN=0.08
LOSS_SPATIAL=0.0
TYPED_COMP_GAMMA=0.5
TYPED_COMPANION_FRACTION=0.25
# ---------------------- Training Hyperparameters -------------------
EPOCHS=50
LR=1e-5
BATCH_SIZE=128
NUM_WORKERS=6
VAL_FREQ=1
LOSS_ALIGN=1.0
LOSS_COMP=0.0
SEED=2026


# ------------------ Image Preprocessing Settings -------------------
# Transform type. Choices: ['squarepad', 'targetpad']
TRANSFORM="targetpad"
TARGET_RATIO=1.25


# -------------------- Visual Baseline Settings ---------------------
# To train the Visual Anchor Baseline (Drawing Bbox on image), set BBOX_WIDTH > 0 (e.g., 3)
BBOX_WIDTH=0
BBOX_COLOR="red"

# To train the ROI-Crop Baseline (Cropping the Bbox region), uncomment the line below:
# BBOX_CROP="--bounding-box-crop"


# -------------------------- Boolean Flags --------------------------
# Reference-box supervision. oacir_latent uses this as instance-anchor input, not AdaFocal attention bias.
HIGHLIGHT_TRAINING="--highlight-training"
HIGHLIGHT_INFERENCE="--highlight-inference"

# Save strategies
SAVE_TRAINING="--save-training"
SAVE_BEST="--save-best"

# Hardware / Memory options
SAVE_MEMORY="--save-memory"

# Text Prompt
# TEXT_ENTITY="--text-entity"
TYPED_CONTRASTIVE=""


# ==============================================================================
# Execute Python Script
# ==============================================================================

echo "Starting Training for ${DATASET} using model ${MODEL_NAME}..."

python train.py \
    --dataset ${DATASET} \
    --data-root ${DATA_ROOT} \
    --blip-model-name ${MODEL_NAME} \
    --vit-backbone ${VIT_BACKBONE} \
    --save-dir ${SAVE_DIR} \
    --num-epochs ${EPOCHS} \
    --learning-rate ${LR} \
    --batch-size ${BATCH_SIZE} \
    --num-workers ${NUM_WORKERS} \
    --validation-frequency ${VAL_FREQ} \
    --loss-align ${LOSS_ALIGN} \
    --loss-comp ${LOSS_COMP} \
    --transform ${TRANSFORM} \
    --target-ratio ${TARGET_RATIO} \
    --bounding-box-width ${BBOX_WIDTH} \
    --bounding-box-color ${BBOX_COLOR} \
    --seed ${SEED} \
    --lambda-ins "$LAMBDA_INS" \
    --temp-ins "$TEMP_INS" \
    --loss-ins "$LOSS_INS" \
    --loss-ot "$LOSS_OT" \
    --loss-final "$LOSS_FINAL" \
    --loss-id "$LOSS_ID" \
    --loss-edit "$LOSS_EDIT" \
    --loss-ent "$LOSS_ENT" \
    --latent-chunk-size "$LATENT_CHUNK_SIZE" \
    --latent-matcher "$LATENT_MATCHER" \
    --ot-sinkhorn-iters "$OT_SINKHORN_ITERS" \
    --ot-temperature "$OT_TEMPERATURE" \
    --ot-vote-temperature "$OT_VOTE_TEMPERATURE" \
    --ot-text-weight "$OT_TEXT_WEIGHT" \
    --ot-bbox-gamma "$OT_BBOX_GAMMA" \
    --ot-topk "$OT_TOPK" \
    --bicm-temp "$BICM_TEMP" \
    --bicm-lambda-id "$BICM_LAMBDA_ID" \
    --bicm-lambda-ent "$BICM_LAMBDA_ENT" \
    --bicm-mask-floor "$BICM_MASK_FLOOR" \
    --bicm-fusion-weight "$BICM_FUSION_WEIGHT" \
    --train-feature-cache "$TRAIN_FEATURE_CACHE" \
    --val-feature-cache "$VAL_FEATURE_CACHE" \
    --val-feature-batch-size "$VAL_FEATURE_BATCH_SIZE" \
    --region-topk "$REGION_TOPK" \
    --region-topk-max "$REGION_TOPK_MAX" \
    --region-area-scale "$REGION_AREA_SCALE" \
    --region-spatial-kernel "$REGION_SPATIAL_KERNEL" \
    --region-spatial-weight "$REGION_SPATIAL_WEIGHT" \
    --region-temperature "$REGION_TEMPERATURE" \
    --spatial-variance-margin "$SPATIAL_VARIANCE_MARGIN" \
    --loss-spatial "$LOSS_SPATIAL" \
    --typed-comp-gamma "$TYPED_COMP_GAMMA" \
    --typed-companion-fraction "$TYPED_COMPANION_FRACTION" \
    ${BBOX_CROP} \
    ${HIGHLIGHT_TRAINING} \
    ${HIGHLIGHT_INFERENCE} \
    ${SAVE_TRAINING} \
    ${SAVE_BEST} \
    ${SAVE_MEMORY} \
    ${TEXT_ENTITY} \
    ${TYPED_CONTRASTIVE}
