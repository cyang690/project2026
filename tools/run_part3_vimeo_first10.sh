#!/usr/bin/env bash
set -euo pipefail

cd /data/cyang690/vsr_project_2026
mkdir -p logs

clips=$(find results/vimeo_basicvsrpp/images -mindepth 1 -maxdepth 1 -type d \
  | sed 's#.*/##' \
  | sort \
  | head -10)

{
  echo "Vimeo first 10 Part 3 run started at $(date '+%F %T')"
  for seq in $clips; do
    echo "Running Part 3 on Vimeo clip: $seq"
    envs/vsr_env/bin/python part3/temporal_hybrid_vsr.py \
      --mode run \
      --dataset vimeo \
      --seq "$seq" \
      --max-frames 7 \
      --temporal-strength 0.70 \
      --max-detail-alpha 0.28
  done
  echo "Vimeo first 10 Part 3 run finished at $(date '+%F %T')"
} > logs/part3_vimeo_first10.log 2>&1
