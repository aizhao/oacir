#!/usr/bin/env bash
set -euo pipefail

gpu_id=$(
  nvidia-smi \
    --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits \
  | awk -F',' '
      {
        gsub(/ /, "", $1)
        gsub(/ /, "", $2)
        gsub(/ /, "", $3)
        score = $2 + 100 * $3
        if (NR == 1 || score < best) {
          best = score
          gpu = $1
        }
      }
      END { print gpu }
    '
)

echo "Selected physical GPU: ${gpu_id}"
nvidia-smi \
  --id="${gpu_id}" \
  --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader

cd /home/caoyu/mnt/zhaoai/OACIR

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

/home/caoyu/mnt/zhaoai/conda_envs/oacir/bin/python -u train.py \
  --dataset Union \
  --data-root ./Datasets/OACIRR \
  --blip-model-name oacir_box_query_identity \
  --blip-model-weight /home/caoyu/mnt/zhaoai/OACIR/checkpoints/OACIR_Union_finetune_oacir_box_query_2026-10-10_16:49:02/saved_models/adafocal_finetune_best.pt \
  --identity-model-weight /home/caoyu/mnt/zhaoai/OACIR/checkpoints/OACIR_Union_finetune_oacir_latent_2026-09-29_19:20:30/saved_models/adafocal_finetune_best.pt \
  --vit-backbone pretrain \
  --save-dir ./checkpoints \
  --num-epochs 10 \
  --learning-rate 3e-6 \
  --batch-size 256 \
  --num-workers 6 \
  --validation-frequency 1 \
  --loss-align 1.0 \
  --loss-core-matcher 1.0 \
  --loss-core-id 1.0 \
  --loss-core-comp 0.5 \
  --loss-core-region 0.0 \
  --core-matcher-temp 0.07 \
  --core-matcher-lambda-id 1.25 \
  --core-matcher-fusion-weight 0.40 \
  --core-matcher-topk 50 \
  --core-matcher-cycle-weight 0.25 \
  --core-matcher-text-weight 0.25 \
  --core-matcher-bbox-floor 0.05 \
  --core-matcher-query-chunk-size 2 \
  --region-topk 16 \
  --region-topk-max 96 \
  --region-area-scale 1.5 \
  --region-spatial-kernel 3 \
  --region-spatial-weight 0.15 \
  --region-temperature 0.07 \
  --transform targetpad \
  --target-ratio 1.25 \
  --bounding-box-width 0 \
  --bounding-box-color red \
  --seed 2026 \
  --train-feature-cache ./cache/oacirr_union_train_vitg_targetpad125_raw_fp16.pt \
  --val-feature-cache './cache/oacirr_{variant_lower}_val_vitg_targetpad125_raw_fp16.pt' \
  --val-feature-batch-size 128 \
  --latent-gallery-chunk-size 2048 \
  --val-query-batch-size 64 \
  --highlight-training \
  --highlight-inference \
  --save-training \
  --save-best \
  --save-memory
