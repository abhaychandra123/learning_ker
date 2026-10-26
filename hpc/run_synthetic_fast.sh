#!/bin/bash
# Launched by switch_synthetic_run.py after preserving the legacy checkpoint.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
: "${QDM_PYTHON:?Set GPU Python}"
: "${QDM_EXPERIMENT:?Set fresh optimized experiment root}"
export OMP_NUM_THREADS="${QDM_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/qdm-fast-mpl-$(id -u)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$MPLCONFIGDIR"
common=(--dataset runs/synthetic_v1 --actual-run runs/combined_full_200
        --device cuda --epochs 200 --layers 16 --kernel-size 9 --lr 1e-4
        --lambda-ista 1e-4 --seed 0 --save-predictions --eval-every 5 --cache auto --fft real)
"$QDM_PYTHON" -u scripts/train_synthetic_fast.py "${common[@]}" --subset full \
    --output "$QDM_EXPERIMENT/augmented_1080" --autotune \
    --migrate-from "$QDM_EXPERIMENT/legacy_last.pt" 2>&1 | tee "$QDM_EXPERIMENT/augmented.log"
# A requested epoch-boundary pause must not start the next arm.
"$QDM_PYTHON" - "$QDM_EXPERIMENT/augmented_1080/status.json" <<'PY'
import json,sys
if json.load(open(sys.argv[1]))['status']!='complete':
    raise SystemExit('Augmented arm paused or incomplete; control not started.')
PY
read -r batch checkpoint prefix < <("$QDM_PYTHON" - "$QDM_EXPERIMENT/augmented_1080/setup.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))['contract']['config']
print(c['batch_size'],'on' if c['checkpoint_layers'] else 'off',c['legacy_prefix_epochs'])
PY
)
"$QDM_PYTHON" -u scripts/train_synthetic_fast.py "${common[@]}" --subset matched-original \
    --output "$QDM_EXPERIMENT/control_51" --batch-size "$batch" --checkpoint-mode "$checkpoint" \
    --legacy-prefix-epochs "$prefix" 2>&1 | tee "$QDM_EXPERIMENT/control.log"
"$QDM_PYTHON" scripts/compare_synthetic_experiment.py \
    --control "$QDM_EXPERIMENT/control_51" --augmented "$QDM_EXPERIMENT/augmented_1080" \
    --output "$QDM_EXPERIMENT/comparison"
