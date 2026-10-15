#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {Fashion|Car|Product|Landmark}" >&2
  exit 2
fi

dataset=$1
case "${dataset}" in
  Fashion|Car|Product|Landmark) ;;
  *) echo "Unsupported dataset: ${dataset}" >&2; exit 2 ;;
esac

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

echo "Dataset: ${dataset}"
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

output_dir="./checkpoints/OACIR_Union_finetune_oacir_box_query_2026-10-10_16:49:02/score_search_${dataset,,}"

val_feature_batch_size=32
eval_batch_size=16
gallery_chunk_size=1024
query_chunk_size=2
if [[ "${dataset}" == "Product" ]]; then
  val_feature_batch_size=128
  eval_batch_size=64
  gallery_chunk_size=2048
  query_chunk_size=8
fi

/home/caoyu/mnt/zhaoai/conda_envs/oacir/bin/python -u \
  tools/search_core_score_grid.py \
  --dataset "${dataset}" \
  --checkpoint /home/caoyu/mnt/zhaoai/OACIR/checkpoints/OACIR_Union_finetune_oacir_box_query_2026-10-10_16:49:02/saved_models/adafocal_finetune_best.pt \
  --data-root ./Datasets/OACIRR \
  --val-feature-cache './cache/oacirr_{variant_lower}_val_vitg_targetpad125_raw_fp16.pt' \
  --output-dir "${output_dir}" \
  --blip-model-name oacir_box_query \
  --vit-backbone pretrain \
  --target-ratio 1.25 \
  --val-feature-batch-size "${val_feature_batch_size}" \
  --eval-batch-size "${eval_batch_size}" \
  --gallery-chunk-size "${gallery_chunk_size}" \
  --num-workers 4 \
  --core-matcher-temp 0.07 \
  --core-matcher-topk 50 \
  --core-matcher-cycle-weight 0.25 \
  --core-matcher-text-weight 0.25 \
  --core-matcher-bbox-floor 0.05 \
  --core-matcher-query-chunk-size "${query_chunk_size}" \
  --region-topk 16 \
  --region-topk-max 96 \
  --region-area-scale 1.5 \
  --region-spatial-kernel 3 \
  --region-spatial-weight 0.15 \
  --region-temperature 0.07 \
  --fusion-values 0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.60 \
  --lambda-values 0.00 0.25 0.50 0.75 1.00 1.25 1.50 2.00 2.50 3.00 \
  --fine-seeds 5 \
  --top-results 40
