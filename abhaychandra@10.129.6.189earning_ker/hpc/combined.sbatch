#!/bin/bash
#SBATCH --job-name=qdm-combined
#SBATCH --account=ajitvr
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=qdm-combined-%j.out
#SBATCH --error=qdm-combined-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
: "${QDM_PYTHON:?Set QDM_PYTHON to the absolute Linux environment Python path}"
: "${QDM_DATASET:?Set QDM_DATASET to an absolute prepared-dataset directory}"
: "${QDM_RUN:?Set QDM_RUN to a fresh absolute run directory, or an existing run with --resume}"
[[ "$QDM_PYTHON" = /* && "$QDM_DATASET" = /* && "$QDM_RUN" = /* ]] || {
    echo 'QDM_PYTHON, QDM_DATASET and QDM_RUN must be absolute paths.' >&2; exit 2;
}
[[ -x "$QDM_PYTHON" ]] || { echo 'QDM_PYTHON is not executable.' >&2; exit 2; }
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/qdm-mpl-${USER}-${SLURM_JOB_ID}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$MPLCONFIGDIR"

hostname
nvidia-smi
"$QDM_PYTHON" -u hpc/check_cuda.py
"$QDM_PYTHON" -m pip freeze > "environment-${SLURM_JOB_ID}.txt"

# Existing datasets are hash-validated by the experiment, never reconverted.
if [[ ! -f "$QDM_DATASET/manifest.json" ]]; then
    if [[ -e "$QDM_DATASET" ]]; then
        echo 'Dataset folder exists without a completed manifest. Use a fresh path.' >&2
        exit 2
    fi
    "$QDM_PYTHON" -u scripts/combined_data.py \
        --raw-root "${QDM_RAW_ROOT:-$SLURM_SUBMIT_DIR/data}" \
        --output "$QDM_DATASET" --pairing "${QDM_PAIRING:-anchor}"
fi

"$QDM_PYTHON" -u scripts/compare_combined.py \
    --manifest "$QDM_DATASET/manifest.json" --output "$QDM_RUN" \
    --device cuda --layers 16 --epochs 200 --batch-size 1 \
    --checkpoint-layers --save-reconstructions "$@"
