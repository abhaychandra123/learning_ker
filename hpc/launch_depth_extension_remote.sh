#!/bin/bash
# Mac entry point. Copies sources into a NEW directory; no old source/run edits.
set -euo pipefail
target="${1:-abhaychandra@10.129.6.189}"
[[ "$target" != -* && "$target" =~ ^[a-zA-Z0-9_.@:-]+$ ]] || exit 2
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
ssh "$target" 'set -eu
test -x "$HOME/qdm-env/bin/python"
test -f "$HOME/learning_ker/runs/combined_full_200/best.pt"
test ! -e "$HOME/learning_ker/runs/depth_extension_v1"
test ! -e "$HOME/learning_ker/experiments/depth_extension_v1"
mkdir -p "$HOME/learning_ker/experiments" "$HOME/learning_ker/logs"
mkdir -p "$HOME/learning_ker/experiments/depth_extension_v1/inputs/augmented_1080"'
rsync -av --exclude '__pycache__' --exclude '*.pyc' --relative scripts/ \
    hpc/check_cuda.py hpc/run_depth_extension.sh \
    "$target:~/learning_ker/experiments/depth_extension_v1/"
# The synthetic arm's checkpoint is no longer in the server's runs/; carry the local copy.
rsync -av runs/synthetic_data_hypothesis_fast_v2/augmented_1080/final.pt \
    runs/synthetic_data_hypothesis_fast_v2/augmented_1080/setup.json \
    "$target:~/learning_ker/experiments/depth_extension_v1/inputs/augmented_1080/"
ssh "$target" 'set -eu
cd "$HOME/learning_ker/experiments/depth_extension_v1"
nohup env QDM_PYTHON="$HOME/qdm-env/bin/python" QDM_PROJECT_ROOT="$HOME/learning_ker" \
    QDM_SYNTHETIC_RUN="$HOME/learning_ker/experiments/depth_extension_v1/inputs/augmented_1080" \
    bash hpc/run_depth_extension.sh > "$HOME/learning_ker/logs/depth_extension_v1.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$HOME/learning_ker/logs/depth_extension_v1.pid"
echo "Launcher PID: $pid"
echo "Log: ~/learning_ker/logs/depth_extension_v1.log"'
