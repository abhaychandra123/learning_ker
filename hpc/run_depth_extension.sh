#!/bin/bash
# Server side: run trained 16-layer ConvLISTA checkpoints at 8..64 layers (no retraining).
set -euo pipefail
python_bin="${QDM_PYTHON:?set QDM_PYTHON}"
project_root="${QDM_PROJECT_ROOT:?set QDM_PROJECT_ROOT}"
output="${QDM_OUTPUT:-$project_root/runs/depth_extension_v1}"
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
test ! -e "$output"
date
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
"$python_bin" hpc/check_cuda.py --allow-direct
echo "=== combined: depths 8-64 ==="
QDM_PROJECT_ROOT="$project_root" "$python_bin" -u scripts/depth_extension.py \
    --model combined --device cuda --output "$output/combined"
echo "=== synthetic: depths 8-64 ==="
QDM_PROJECT_ROOT="$project_root" "$python_bin" -u scripts/depth_extension.py \
    --model synthetic --device cuda --run "${QDM_SYNTHETIC_RUN:-$project_root/runs/synthetic_data_hypothesis_fast_v2/augmented_1080}" \
    --expected-checkpoint-sha256 bd8d704b5c1eda121f0dc2559c6805cd037e15fe68f167e29064bc04b7016eee \
    --output "$output/synthetic"
date
echo "Done. Results: $output/{combined,synthetic}/summary.json and metrics.csv"
