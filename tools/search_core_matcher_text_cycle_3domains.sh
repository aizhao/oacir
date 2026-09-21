#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash tools/search_core_matcher_text_cycle_3domains.sh CHECKPOINT [OUTPUT_DIR]"
  exit 2
fi

CHECKPOINT=$1
OUTPUT_DIR=${2:-./core_matcher_text_cycle_3domain_search}

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT"
  exit 2
fi

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/done"

DOMAINS=(Fashion Car Landmark)
TEXT_WEIGHTS=(0.25 0.50 0.75 1.00)
CYCLE_WEIGHTS=(0.00 0.10 0.25)
FUSION_WEIGHT=0.40
LAMBDA_ID=1.00

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

run_one() {
  local text_weight=$1
  local cycle_weight=$2
  local domain=$3
  local tag="text_${text_weight}_cycle_${cycle_weight}_${domain,,}"
  local log_path="$OUTPUT_DIR/logs/${tag}.log"
  local done_path="$OUTPUT_DIR/done/${tag}.done"

  if [[ -f "$done_path" ]]; then
    echo "[skip] text=$text_weight cycle=$cycle_weight domain=$domain"
    return 0
  fi

  echo "[run] fusion=$FUSION_WEIGHT lambda_id=$LAMBDA_ID text=$text_weight cycle=$cycle_weight domain=$domain"
  if python evaluate.py \
    --dataset "$domain" \
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
    --core-matcher-lambda-id "$LAMBDA_ID" \
    --core-matcher-fusion-weight "$FUSION_WEIGHT" \
    --core-matcher-topk 50 \
    --core-matcher-cycle-weight "$cycle_weight" \
    --core-matcher-text-weight "$text_weight" \
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
    echo "[failed] text=$text_weight cycle=$cycle_weight domain=$domain"
    return 1
  fi
}

for text_weight in "${TEXT_WEIGHTS[@]}"; do
  for cycle_weight in "${CYCLE_WEIGHTS[@]}"; do
    for domain in "${DOMAINS[@]}"; do
      run_one "$text_weight" "$cycle_weight" "$domain"
    done
  done
done

python - "$OUTPUT_DIR" "$FUSION_WEIGHT" "$LAMBDA_ID" <<'PY'
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

output_dir = Path(sys.argv[1])
fusion_weight = float(sys.argv[2])
lambda_id = float(sys.argv[3])
pattern = re.compile(
    r"text_(?P<text>[0-9.]+)_cycle_(?P<cycle>[0-9.]+)_"
    r"(?P<domain>fashion|car|landmark)\.log$"
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
    rows.append({
        "fusion": fusion_weight,
        "lambda_id": lambda_id,
        "text_weight": float(match.group("text")),
        "cycle_weight": float(match.group("cycle")),
        "domain": match.group("domain").title(),
        **{key: metrics[key] for key in metric_names},
    })

all_path = output_dir / "all_results.csv"
with all_path.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "fusion", "lambda_id", "text_weight", "cycle_weight", "domain",
            *metric_names,
        ],
    )
    writer.writeheader()
    writer.writerows(sorted(
        rows,
        key=lambda row: (row["text_weight"], row["cycle_weight"], row["domain"]),
    ))

grouped = defaultdict(list)
for row in rows:
    grouped[(row["text_weight"], row["cycle_weight"])].append(row)

summary = []
for (text_weight, cycle_weight), group in grouped.items():
    if len(group) != 3:
        continue
    averaged = {
        f"mean_{metric}": sum(row[metric] for row in group) / len(group)
        for metric in metric_names
    }
    summary.append({
        "fusion": fusion_weight,
        "lambda_id": lambda_id,
        "text_weight": text_weight,
        "cycle_weight": cycle_weight,
        **averaged,
        "early_retrieval_score": (
            averaged["mean_R@1"] + averaged["mean_R@5"]
        ) / 2.0,
        "official_arithmetic": sum(
            averaged[f"mean_{metric}"]
            for metric in ("R@1", "R@5", "R@10", "R@50")
        ) / 4.0,
    })

summary.sort(
    key=lambda row: (row["early_retrieval_score"], row["official_arithmetic"]),
    reverse=True,
)
summary_path = output_dir / "summary.csv"
fieldnames = [
    "fusion", "lambda_id", "text_weight", "cycle_weight",
    "early_retrieval_score", "official_arithmetic",
    *[f"mean_{metric}" for metric in metric_names],
]
with summary_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary)

print(f"Detailed results: {all_path}")
print(f"Three-domain summary: {summary_path}")
if summary:
    best = summary[0]
    print(
        "Best by three-domain mean(R@1, R@5): "
        f"text_weight={best['text_weight']:.2f}, "
        f"cycle_weight={best['cycle_weight']:.2f}, "
        f"early_score={best['early_retrieval_score']:.4f}, "
        f"official={best['official_arithmetic']:.4f}"
    )
PY
