#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 {Fashion|Car|Product|Landmark} GPU_ID {latent_comp_id|latent_all}" >&2
  exit 2
fi

dataset=$1
gpu_id=$2
route=$3
case "${dataset}" in
  Fashion|Car|Product|Landmark) ;;
  *) echo "Unsupported dataset: ${dataset}" >&2; exit 2 ;;
esac
case "${route}" in
  latent_comp_id) label=c1_latent_comp_id ;;
  latent_all) label=c2_latent_all ;;
  *) echo "Unsupported route: ${route}" >&2; exit 2 ;;
esac

cd /home/caoyu/mnt/zhaoai/OACIR

identity_run="./checkpoints/OACIR_Union_finetune_oacir_box_query_identity_2026-10-12_17:21:32"
box_checkpoint="./checkpoints/OACIR_Union_finetune_oacir_box_query_2026-10-10_16:49:02/saved_models/adafocal_finetune_best.pt"
latent_checkpoint="./checkpoints/OACIR_Union_finetune_oacir_latent_2026-09-29_19:20:30/saved_models/adafocal_finetune_best.pt"
output_dir="${identity_run}/ablation_${label}_${dataset,,}"
mkdir -p "${output_dir}/tmp"

echo "Dataset: ${dataset}"
echo "Identity route: ${route}"
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
  --checkpoint "${box_checkpoint}" \
  --skip-default-pretrained \
  --identity-checkpoint "${latent_checkpoint}" \
  --identity-route "${route}" \
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
  --fusion-values 0.4 \
  --lambda-values 1.0 \
  --no-fine-search \
  --top-results 1
