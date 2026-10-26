#!/bin/bash
# Mac entry point. Copies NEW files only; legacy source hashes remain valid.
# Switches the identified legacy run to v2 from its latest completed epoch.
set -euo pipefail
target="${1:?Usage: bash hpc/optimize_synthetic_remote.sh USER@GPU_SERVER}"
[[ "$target" != -* && "$target" =~ ^[a-zA-Z0-9_.@:-]+$ ]] || exit 2
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
ssh "$target" 'test ! -e "$HOME/learning_ker/runs/synthetic_data_hypothesis_fast_v2" || {
    echo "Optimized experiment already exists; inspect/resume it before copying any new code." >&2
    exit 2
}'
rsync -av --relative scripts/synthetic_fast.py scripts/train_synthetic_fast.py \
    scripts/switch_synthetic_run.py scripts/audit_synthetic_roi.py \
    tests/test_synthetic_fast.py hpc/run_synthetic_fast.sh README_SYNTHETIC_OPTIMIZATION.md \
    "$target:~/learning_ker/"
ssh "$target" 'set -eu
cd "$HOME/learning_ker"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/qdm-fast-test-mpl"
export OMP_NUM_THREADS=4
mkdir -p "$MPLCONFIGDIR"
"$HOME/qdm-env/bin/python" -m unittest discover -s tests -p "test_synthetic_fast.py" -v
"$HOME/qdm-env/bin/python" -u scripts/switch_synthetic_run.py'
