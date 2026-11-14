#!/bin/bash
# MAGIC-UNet on the professor's server: generate crops once, then train and score.
# Re-running the script resumes training from runs/magic_unet/$QDM_RUN/latest.pt (default run2).
# Run 1 (folder run) had layout-level noise and is invalid; data made before the fix get their
# noise levels rewritten once (magic_data.py --resigma) before training.
# QDM_DATA_ARGS and QDM_TRAIN_ARGS come last on their command lines, so they override the defaults
# (10 um runs: QDM_DATA_ARGS="--bin 7 --standoffs 4 --crops-per-standoff 8 --z-mean 10 --z-std 2").
# QDM_WAIT_FOR=<file>: after the data are ready, wait for that file before training (GPU queue).
# On exit, success or not, the script writes runs/magic_unet/$QDM_RUN.done for a queued run to see.
set -euo pipefail
code_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
project_root="${QDM_PROJECT_ROOT:-$code_root}"
cd "$project_root"
: "${QDM_PYTHON:?Set the Python interpreter}"
experiment="$project_root/runs/magic_unet"
data_dir="$experiment/${QDM_DATA:-data}"          # run 3: QDM_DATA=data_any
read -r -a data_args <<< "${QDM_DATA_ARGS:-}"     # run 3: --z-uniform 1 10
read -r -a train_args <<< "${QDM_TRAIN_ARGS:-}"   # run 3: --z-range 1 10
run="${QDM_RUN:-run2}"
mkdir -p "$experiment"
trap 'echo "exit $? $(date)" > "$experiment/$run.done"' EXIT
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg
"$QDM_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'

if [ ! -f "$data_dir/manifest.json" ]; then
    rm -rf "$data_dir"
    "$QDM_PYTHON" -u "$code_root/scripts/magic_data.py" --output "$data_dir" \
        --train-crops 153600 --val-crops 4096 --crop 64 --bin 2 --workers "${QDM_WORKERS:-12}" \
        "${data_args[@]}" 2>&1 | tee "$data_dir.log"
fi

if ! grep -q '"noise"' "$data_dir/manifest.json"; then
    "$QDM_PYTHON" -u "$code_root/scripts/magic_data.py" --resigma "$data_dir" \
        2>&1 | tee -a "$data_dir.log"
fi

if [ -n "${QDM_WAIT_FOR:-}" ]; then
    echo "data ready $(date); waiting for $QDM_WAIT_FOR" | tee -a "$experiment/$run.log"
    while [ ! -e "$QDM_WAIT_FOR" ]; do sleep 60; done
fi

"$QDM_PYTHON" -u "$code_root/scripts/train_magic_unet.py" --data "$data_dir" \
    --output "$experiment/$run" --epochs 50 --batch 32 --lr 1e-4 --workers 6 --device cuda \
    --sim-dataset "$project_root/runs/simulated_3_to_05_v3" \
    --comparison-run "$project_root/runs/combined_full_200" --data3 "$project_root/data/data3" \
    "${train_args[@]}" 2>&1 | tee -a "$experiment/$run.log"
