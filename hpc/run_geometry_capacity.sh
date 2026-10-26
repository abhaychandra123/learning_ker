#!/bin/bash
# Four sequential arms in a separate experiment, using an immutable source copy.
set -euo pipefail
code_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
project_root="${QDM_PROJECT_ROOT:-$code_root}"
cd "$project_root"
: "${QDM_PYTHON:?Set the Python interpreter}"
experiment="$project_root/runs/geometry_capacity_v1"
mkdir "$experiment"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLBACKEND=Agg CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="${TMPDIR:-/tmp}/qdm-geometry-mpl-$(id -u)"
mkdir -p "$MPLCONFIGDIR"
"$QDM_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
for sharing in tied untied; do
    for layers in 16 32; do
        arm="${sharing}_${layers}"
        "$QDM_PYTHON" -u "$code_root/scripts/train_geometry_capacity.py" \
            --dataset "$project_root/runs/synthetic_v1" \
            --actual-run "$project_root/runs/combined_full_200" \
            --output "$experiment/$arm" --geometry L1_C1 \
            --sharing "$sharing" --layers "$layers" --epochs 200 \
            --kernel-size 9 --lr 1e-4 --lambda-ista 1e-4 --seed 0 \
            --device cuda --cache auto --eval-every 20 --autotune \
            2>&1 | tee "$experiment/$arm.log"
        "$QDM_PYTHON" - "$experiment/$arm/status.json" <<'PY'
import json,sys
if json.load(open(sys.argv[1]))['status']!='complete':
    raise SystemExit('Arm paused/incomplete; subsequent arms were not started.')
PY
    done
done
"$QDM_PYTHON" "$code_root/scripts/summarize_geometry_capacity.py" --experiment "$experiment"
