#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/hpo_places365_wd_boundary_8d_10seeds_30k.yaml}"
NUM_SHARDS="${2:-24}"
OUTPUT_DIR="results/hpo_places365_wd_boundary_8d_10seeds_30k"
LOG_DIR="$OUTPUT_DIR/logs"

mkdir -p "$LOG_DIR"

echo "Preparing or reusing the existing Places365 feature cache..."
nice -n 19 python -u prepare_features.py --config "$CONFIG" \
  2>&1 | tee -a "$LOG_DIR/prepare_features.log"

for ((shard_index = 0; shard_index < NUM_SHARDS; shard_index++)); do
  shard_log=$(printf "%s/shard_%04d_of_%04d.log" "$LOG_DIR" "$shard_index" "$NUM_SHARDS")
  echo "Running WD-boundary shard $shard_index/$NUM_SHARDS; log: $shard_log"
  nice -n 19 python -u run_hpo.py \
    --config "$CONFIG" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$shard_index" \
    2>&1 | tee -a "$shard_log"
done

nice -n 19 python -u run_hpo.py \
  --config "$CONFIG" \
  --aggregate-only \
  2>&1 | tee -a "$LOG_DIR/aggregate.log"
