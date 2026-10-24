#!/bin/bash
# Run on a dedicated GPU server or inside an existing GPU allocation.
# This does not request scheduler resources or regenerate the dataset.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
: "${QDM_PYTHON:?Set QDM_PYTHON to the GPU environment Python executable}"
QDM_DATASET="${QDM_DATASET:-$project_root/runs/synthetic_v1}"
QDM_ACTUAL_RUN="${QDM_ACTUAL_RUN:-$project_root/runs/combined_full_200}"
QDM_EXPERIMENT="${QDM_EXPERIMENT:-$project_root/runs/synthetic_data_hypothesis_v1}"
QDM_EPOCHS="${QDM_EPOCHS:-200}"
export OMP_NUM_THREADS="${QDM_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/qdm-synthetic-training-mpl-$(id -u)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$MPLCONFIGDIR"
[[ ! -e "$QDM_EXPERIMENT" ]] || {
    echo "Experiment exists: $QDM_EXPERIMENT. Use the README's individual resume commands." >&2
    exit 2
}
"$QDM_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
"$QDM_PYTHON" -m unittest discover -s tests -p 'test_synthetic_data.py' -v
"$QDM_PYTHON" -m unittest discover -s tests -p 'test_synthetic_training.py' -v
"$QDM_PYTHON" scripts/synthetic_data.py inspect --dataset "$QDM_DATASET"
mkdir -p "$QDM_EXPERIMENT"
"$QDM_PYTHON" -m pip freeze > "$QDM_EXPERIMENT/environment.txt"
common=(--dataset "$QDM_DATASET" --actual-run "$QDM_ACTUAL_RUN" --device cuda
        --layers 16 --kernel-size 9 --lr 1e-4 --lambda-ista 1e-4 --seed 0
        --checkpoint-layers --save-predictions --epochs "$QDM_EPOCHS")
# Run requested large-data arm first; do not run competing jobs on one GPU.
"$QDM_PYTHON" -u scripts/train_synthetic.py "${common[@]}" --subset full \
    --output "$QDM_EXPERIMENT/augmented_1080" 2>&1 | tee "$QDM_EXPERIMENT/augmented.log"
"$QDM_PYTHON" -u scripts/train_synthetic.py "${common[@]}" --subset matched-original \
    --output "$QDM_EXPERIMENT/control_51" 2>&1 | tee "$QDM_EXPERIMENT/control.log"
"$QDM_PYTHON" scripts/compare_synthetic_experiment.py \
    --control "$QDM_EXPERIMENT/control_51" --augmented "$QDM_EXPERIMENT/augmented_1080" \
    --output "$QDM_EXPERIMENT/comparison"
