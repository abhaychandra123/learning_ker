#!/bin/bash
# Fetch the completed augmented checkpoint, then perform fresh inference on this Mac.
set -euo pipefail
target="${1:-abhaychandra@10.129.6.189}"
[[ "$target" != -* && "$target" =~ ^[a-zA-Z0-9_.@:-]+$ ]] || exit 2
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_bin="${QDM_PYTHON:-/tmp/qdm-synthetic-torch-env/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    echo 'Set QDM_PYTHON to a local Python with torch, numpy and matplotlib.' >&2
    exit 2
fi
"$python_bin" -c 'import torch, numpy, matplotlib'
run='runs/synthetic_data_hypothesis_fast_v2/augmented_1080'
output='runs/reconstruction_synthetic_L1_C1_10um'
if [[ -e "$output" ]]; then
    echo "Output already exists: $output. Inspect status.json; use the Python entry point with a fresh output to rerun." >&2
    exit 2
fi
if [[ ! -f "$run/final.pt" || ! -f "$run/setup.json" ]]; then
    mkdir -p "$run"
    rsync -av "$target:~/learning_ker/$run/final.pt" \
        "$target:~/learning_ker/$run/setup.json" "$run/"
fi
export MPLCONFIGDIR="${TMPDIR:-/tmp}/qdm-synthetic-inference-mpl"
mkdir -p "$MPLCONFIGDIR"
"$python_bin" -u scripts/reconstruct_synthetic.py --run "$run" \
    --geometry L1_C1 --standoff 10 --target-standoff 0.5 --device cpu \
    --expected-checkpoint-sha256 bd8d704b5c1eda121f0dc2559c6805cd037e15fe68f167e29064bc04b7016eee \
    --include-saved-baselines --output "$output"
if [[ "$(uname -s)" == Darwin ]]; then
    open "$output/index.html"
fi
