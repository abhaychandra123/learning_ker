#!/bin/bash
# CPU preparation only. No scheduler, GPU training or changes to existing runs.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
: "${QDM_PYTHON:?Set QDM_PYTHON to the environment Python executable}"
export OMP_NUM_THREADS="${QDM_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/qdm-synthetic-mpl-$(id -u)"
mkdir -p "$MPLCONFIGDIR"
exec "$QDM_PYTHON" -u scripts/synthetic_data.py prepare "$@"
