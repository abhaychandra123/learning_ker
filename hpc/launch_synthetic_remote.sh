#!/bin/bash
# Run from the Mac: bash hpc/launch_synthetic_remote.sh USER@GPU_SERVER
# Assumes ~/learning_ker and ~/qdm-env already exist on the dedicated GPU server.
set -euo pipefail
target="${1:?Usage: bash hpc/launch_synthetic_remote.sh USER@GPU_SERVER}"
[[ "$target" != -* && "$target" =~ ^[a-zA-Z0-9_.@:-]+$ ]] || {
    echo 'Supply a plain SSH user@hostname destination.' >&2; exit 2;
}
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
ssh "$target" 'set -eu
test -d "$HOME/learning_ker/scripts" || { echo "Missing remote project" >&2; exit 2; }
test -x "$HOME/qdm-env/bin/python" || { echo "Missing remote Python environment" >&2; exit 2; }
test -f "$HOME/learning_ker/runs/combined_full_200/setup.json" || { echo "Missing actual COMSOL run" >&2; exit 2; }
test ! -e "$HOME/learning_ker/runs/synthetic_data_hypothesis_v1" || { echo "Experiment exists; use README resume commands" >&2; exit 2; }
mkdir -p "$HOME/learning_ker/logs"
test ! -e "$HOME/learning_ker/logs/synthetic_data_hypothesis_v1.pid" || { echo "Previous launch recorded; inspect logs/process before launching again" >&2; exit 2; }
if [ -e "$HOME/learning_ker/runs/synthetic_v1" ] && [ ! -f "$HOME/learning_ker/runs/synthetic_v1/manifest.json" ]; then
    echo "Remote synthetic directory is incomplete; inspect before replacing" >&2; exit 2
fi'
rsync -av --relative \
    scripts/train_synthetic.py scripts/synthetic_experiment.py \
    scripts/compare_synthetic_experiment.py scripts/synthetic_data.py scripts/propagator.py \
    tests/test_synthetic_training.py tests/test_synthetic_data.py \
    hpc/run_synthetic_experiment.sh README_SYNTHETIC_EXPERIMENT.md \
    "$target:~/learning_ker/"
# Copy only if absent. Never merge over an existing remote dataset.
if ssh "$target" 'test -d "$HOME/learning_ker/runs/synthetic_v1"'; then
    echo 'Using existing remote synthetic dataset; loader will verify it.'
else
    result=$?
    [[ "$result" -eq 1 ]] || exit "$result"
    rsync -av runs/synthetic_v1/ "$target:~/learning_ker/runs/synthetic_v1/"
fi
ssh "$target" 'set -eu
cd "$HOME/learning_ker"
"$HOME/qdm-env/bin/python" scripts/synthetic_data.py inspect --dataset runs/synthetic_v1
nohup env QDM_PYTHON="$HOME/qdm-env/bin/python" bash hpc/run_synthetic_experiment.sh \
    > logs/synthetic_data_hypothesis_v1.launch.log 2>&1 < /dev/null &
pid=$!
echo "$pid" > logs/synthetic_data_hypothesis_v1.pid
echo "Launcher PID: $pid (GPU tests/training status must be confirmed in log)"
echo "Log: ~/learning_ker/logs/synthetic_data_hypothesis_v1.launch.log"'
