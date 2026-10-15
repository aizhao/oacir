#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {Fashion|Car|Product|Landmark} GPU_ID" >&2
  exit 2
fi

dataset=$1
gpu_id=$2
case "${dataset}" in
  Fashion|Car|Product|Landmark) ;;
  *) echo "Unsupported dataset: ${dataset}" >&2; exit 2 ;;
esac

cd /home/caoyu/mnt/zhaoai/OACIR

run_dir="./checkpoints/OACIR_Union_finetune_oacir_box_query_identity_2026-10-12_17:21:32"
output_dir="${run_dir}/score_search_e2_pareto_${dataset,,}"
mkdir -p "${output_dir}/tmp"

echo "Dataset: ${dataset}"
echo "Selected physical GPU: ${gpu_id}"
nvidia-smi \
  --id="${gpu_id}" \
  --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export TMPDIR="$(pwd)/${output_dir#./}/tmp"

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
  --checkpoint "${run_dir}/saved_models/adafocal_finetune_best_identity.pt" \
  --data-root ./Datasets/OACIRR \
  --val-feature-cache './cache/oacirr_{variant_lower}_val_vitg_targetpad125_raw_fp16.pt' \
  --output-dir "${output_dir}" \
  --blip-model-name oacir_box_query_identity \
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
  --fusion-values \
    0.000 0.025 0.050 0.075 0.100 0.125 0.150 0.175 0.200 0.225 0.250 \
    0.275 0.300 0.325 0.350 0.375 0.400 0.425 0.450 0.475 0.500 0.525 \
    0.550 0.575 0.600 0.625 0.650 0.675 0.700 0.725 0.750 0.775 0.800 \
  --lambda-values \
    0.000 0.125 0.250 0.375 0.500 0.625 0.750 0.875 1.000 1.125 1.250 \
    1.375 1.500 1.625 1.750 1.875 2.000 2.125 2.250 2.375 2.500 2.625 \
    2.750 2.875 3.000 \
  --no-fine-search \
  --top-results 20
