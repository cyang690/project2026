#!/usr/bin/env bash
set -euo pipefail

cd /data/cyang690/vsr_project_2026
mkdir -p logs
rm -rf results/part3_temporal_hybrid/wild/wild_realesrgan/frames
rm -f results/part3_temporal_hybrid/wild/wild_realesrgan/metrics.json
rm -f results/part3_temporal_hybrid/wild/wild_realesrgan/frame_stats.csv
rm -f results/part3_temporal_hybrid/wild/wild_realesrgan/wild_realesrgan_part3_temporal_hybrid.mp4

nohup envs/vsr_env/bin/python part3/temporal_hybrid_vsr.py \
  --mode run \
  --dataset wild \
  --max-frames 0 \
  --temporal-strength 0.72 \
  --max-detail-alpha 0.22 \
  --flow-max-side 480 \
  > logs/part3_wild_full.log 2>&1 < /dev/null &

echo "$!" > logs/part3_wild_full.pid
echo "started wild full pid $(cat logs/part3_wild_full.pid)"
