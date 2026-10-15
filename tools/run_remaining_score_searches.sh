#!/usr/bin/env bash
set -euo pipefail

cd /home/caoyu/mnt/zhaoai/OACIR

run_root=./checkpoints/OACIR_Union_finetune_oacir_box_query_2026-10-10_16:49:02
for dataset in Car Landmark Product; do
  output_dir="${run_root}/score_search_${dataset,,}"
  mkdir -p "${output_dir}"
  bash tools/run_domain_score_search.sh "${dataset}" \
    2>&1 | tee "${output_dir}/run.log"
done
