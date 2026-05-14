#!/usr/bin/env bash
set -euo pipefail

cd /data/cyang690/vsr_project_2026
mkdir -p logs

nohup tools/run_part3_vimeo_first10.sh \
  > logs/part3_vimeo_first10_launcher.log 2>&1 < /dev/null &

echo "$!" > logs/part3_vimeo_first10.pid
echo "started vimeo first10 pid $(cat logs/part3_vimeo_first10.pid)"
