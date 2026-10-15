#!/usr/bin/env bash
set -euo pipefail

gpu_id=${1:-0}
cd /home/caoyu/mnt/zhaoai/OACIR

identity_run="./checkpoints/OACIR_Union_finetune_oacir_box_query_identity_2026-10-12_17:21:32"

for route in latent_comp_id latent_all; do
  case "${route}" in
    latent_comp_id) label=c1_latent_comp_id ;;
    latent_all) label=c2_latent_all ;;
  esac

  for dataset in Product Fashion Car Landmark; do
    log_file="${identity_run}/ablation_${label}_${dataset,,}.log"
    echo "[$(date '+%F %T')] START ${label} ${dataset} GPU=${gpu_id}"
    bash tools/run_identity_routing_ablation.sh \
      "${dataset}" "${gpu_id}" "${route}" 2>&1 | tee "${log_file}"
    echo "[$(date '+%F %T')] DONE ${label} ${dataset}"
  done
done

echo "[$(date '+%F %T')] C1/C2 queue complete"
