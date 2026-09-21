#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash tools/search_core_matcher_fusion_lambda_car.sh CHECKPOINT [OUTPUT_DIR]"
  exit 2
fi

CHECKPOINT=$1
OUTPUT_DIR=${2:-./core_matcher_fusion_lambda_car_search}

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT"
  exit 2
fi

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/done"

FUSION_WEIGHTS=(0.10 0.20 0.30 0.40 0.50)
LAMBDA_IDS=(0.00 0.25 0.50 0.75 1.00)
TEXT_WEIGHT=0.50
CYCLE_WEIGHT=0.00

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

run_one() {
  local fusion=$1
  local lambda_id=$2
  local tag="fusion_${fusion}_lambda_${lambda_id}_car"
  local log_path="$OUTPUT_DIR/logs/${tag}.log"
  local done_path="$OUTPUT_DIR/done/${tag}.done"

  if [[ -f "$done_path" ]]; then
    echo "[skip] fusion=$fusion lambda_id=$lambda_id domain=Car"
    return 0
  fi

  echo "[run] fusion=$fusion lambda_id=$lambda_id domain=Car"
  if python evaluate.py \
    --dataset Car \
    --data-root ./Datasets/OACIRR \
    --blip-model-name oacir_latent \
    --blip-model-weight "$CHECKPOINT" \
    --vit-backbone pretrain \
    --transform targetpad \
    --target-ratio 1.25 \
    --use-latent-matching \
    --highlight-inference \
    --latent-matcher core_matcher \
    --core-matcher-temp 0.07 \
    --core-matcher-lambda-id "$lambda_id" \
    --core-matcher-fusion-weight "$fusion" \
    --core-matcher-topk 50 \
    --core-matcher-cycle-weight "$CYCLE_WEIGHT" \
    --core-matcher-text-weight "$TEXT_WEIGHT" \
    --core-matcher-bbox-floor 0.05 \
    --core-matcher-query-chunk-size 2 \
    --region-topk 16 \
    --region-topk-max 96 \
    --region-area-scale 1.5 \
    --region-spatial-kernel 3 \
    --region-spatial-weight 0.15 \
    --region-temperature 0.07 \
    --latent-chunk-size 64 \
    --latent-gallery-chunk-size 1024 \
    --eval-batch-size 16 \
    --val-feature-batch-size 32 \
    --val-feature-cache './cache/oacirr_{variant_lower}_val_vitg_targetpad125_raw_fp16.pt' \
    --save-memory 2>&1 | tee "$log_path"; then
    touch "$done_path"
  else
    echo "[failed] fusion=$fusion lambda_id=$lambda_id domain=Car"
    return 1
  fi
}

for fusion in "${FUSION_WEIGHTS[@]}"; do
  for lambda_id in "${LAMBDA_IDS[@]}"; do
    run_one "$fusion" "$lambda_id"
  done
done

python - "$OUTPUT_DIR" "$TEXT_WEIGHT" "$CYCLE_WEIGHT" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
text_weight = float(sys.argv[2])
cycle_weight = float(sys.argv[3])
pattern = re.compile(
    r"fusion_(?P<fusion>[0-9.]+)_lambda_(?P<lambda>[0-9.]+)_car\.log$"
)
metric_names = ["R_ID@1", "R_ID@3", "R_ID@5", "R@1", "R@5", "R@10", "R@50"]
rows = []

for log_path in sorted((output_dir / "logs").glob("*.log")):
    match = pattern.match(log_path.name)
    if not match:
        continue
    text = log_path.read_text(errors="replace")
    result_objects = []
    for candidate in re.findall(r"\{[^{}]+\}", text, flags=re.DOTALL):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if "R@1" in value and "R_ID@1" in value:
            result_objects.append(value)
    if not result_objects:
        print(f"Warning: no result JSON found in {log_path}", file=sys.stderr)
        continue

    metrics = result_objects[-1]
    row = {
        "fusion": float(match.group("fusion")),
        "lambda_id": float(match.group("lambda")),
        "text_weight": text_weight,
        "cycle_weight": cycle_weight,
        **{key: metrics[key] for key in metric_names},
    }
    row["early_retrieval_score"] = (row["R@1"] + row["R@5"]) / 2.0
    row["official_arithmetic"] = sum(
        row[key] for key in ("R@1", "R@5", "R@10", "R@50")
    ) / 4.0
    rows.append(row)

rows.sort(
    key=lambda row: (row["early_retrieval_score"], row["official_arithmetic"]),
    reverse=True,
)
result_path = output_dir / "summary.csv"
fieldnames = [
    "fusion", "lambda_id", "text_weight", "cycle_weight",
    "early_retrieval_score", "official_arithmetic", *metric_names,
]
with result_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Car fusion/lambda summary: {result_path}")
if rows:
    best = rows[0]
    print(
        "Best by Car mean(R@1, R@5): "
        f"fusion={best['fusion']:.2f}, lambda_id={best['lambda_id']:.2f}, "
        f"R@1={best['R@1']:.4f}, R@5={best['R@5']:.4f}, "
        f"early_score={best['early_retrieval_score']:.4f}, "
        f"official={best['official_arithmetic']:.4f}"
    )
PY
