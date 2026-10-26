#!/bin/bash
# Mac entry point. Copies sources into a NEW directory; no old source/run edits.
set -euo pipefail
target="${1:-abhaychandra@10.129.6.189}"
[[ "$target" != -* && "$target" =~ ^[a-zA-Z0-9_.@:-]+$ ]] || exit 2
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
ssh "$target" 'set -eu
test -x "$HOME/qdm-env/bin/python"
test -f "$HOME/learning_ker/runs/synthetic_v1/manifest.json"
test -f "$HOME/learning_ker/runs/combined_full_200/setup.json"
test ! -e "$HOME/learning_ker/runs/geometry_capacity_v1"
test ! -e "$HOME/learning_ker/experiments/geometry_capacity_v1"
mkdir -p "$HOME/learning_ker/experiments" "$HOME/learning_ker/logs"
mkdir "$HOME/learning_ker/experiments/geometry_capacity_v1"'
rsync -av --exclude '__pycache__' --exclude '*.pyc' --relative scripts/ \
    tests/test_geometry_capacity.py tests/test_synthetic_training.py tests/test_synthetic_data.py \
    hpc/run_geometry_capacity.sh README_GEOMETRY_CAPACITY.md \
    "$target:~/learning_ker/experiments/geometry_capacity_v1/"
ssh "$target" 'set -eu
cd "$HOME/learning_ker/experiments/geometry_capacity_v1"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/qdm-geometry-test-mpl"
mkdir -p "$MPLCONFIGDIR"
"$HOME/qdm-env/bin/python" -m unittest discover -s tests -p "test_geometry_capacity.py" -v
nohup env QDM_PYTHON="$HOME/qdm-env/bin/python" QDM_PROJECT_ROOT="$HOME/learning_ker" \
    bash hpc/run_geometry_capacity.sh > "$HOME/learning_ker/logs/geometry_capacity_v1.launch.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$HOME/learning_ker/logs/geometry_capacity_v1.pid"
echo "Launcher PID: $pid. Confirm GPU tuning and updates in the log."
echo "Log: ~/learning_ker/logs/geometry_capacity_v1.launch.log"'
