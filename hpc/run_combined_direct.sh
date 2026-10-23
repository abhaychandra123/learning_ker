#!/bin/bash
# Direct GPU-server equivalent of combined.sbatch. No scheduler commands.
# Set QDM_PYTHON, QDM_DATASET and QDM_RUN; additional CLI options pass through.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
: "${QDM_PYTHON:?Set QDM_PYTHON to the absolute Linux environment Python path}"
: "${QDM_DATASET:?Set QDM_DATASET to an absolute prepared-dataset directory}"
: "${QDM_RUN:?Set QDM_RUN to a fresh absolute run directory, or use --resume}"
[[ "$QDM_PYTHON" = /* && "$QDM_DATASET" = /* && "$QDM_RUN" = /* ]] || {
    echo 'QDM_PYTHON, QDM_DATASET and QDM_RUN must be absolute paths.' >&2; exit 2;
}
[[ -x "$QDM_PYTHON" ]] || { echo 'QDM_PYTHON is not executable.' >&2; exit 2; }
export OMP_NUM_THREADS="${QDM_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/qdm-mpl-$(id -u)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$MPLCONFIGDIR"

hostname
nvidia-smi
"$QDM_PYTHON" -u hpc/check_cuda.py --allow-direct
mkdir -p logs
"$QDM_PYTHON" -m pip freeze > "logs/environment-direct-$(date +%Y%m%d-%H%M%S)-$$.txt"

if [[ ! -f "$QDM_DATASET/manifest.json" ]]; then
    if [[ -e "$QDM_DATASET" ]]; then
        echo 'Dataset folder exists without a completed manifest. Use a fresh path.' >&2
        exit 2
    fi
    "$QDM_PYTHON" -u scripts/combined_data.py \
        --raw-root "${QDM_RAW_ROOT:-$project_root/data}" \
        --output "$QDM_DATASET" --pairing "${QDM_PAIRING:-anchor}"
fi

exec "$QDM_PYTHON" -u scripts/compare_combined.py \
    --manifest "$QDM_DATASET/manifest.json" --output "$QDM_RUN" \
    --device cuda --layers 16 --epochs 200 --batch-size 1 \
    --checkpoint-layers --save-reconstructions "$@"
