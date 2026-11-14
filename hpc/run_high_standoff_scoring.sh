#!/bin/bash
# Scoring for the 10 -> 0.5 um runs (HIGH_STANDOFF_PROTOCOL.md). Inference and FISTA only, no training.
#   1. FISTA padding test on simulation (starts at once);
#   2. wait for run10a and run10b to finish (their .done markers);
#   3. MAGIC-UNet height read-out and data-consistency correction (m1_rescore.py) for every finished run;
#   4. FISTA at 300 / 1,000 / 3,000 iterations with the chosen padding, COMSOL and full simulated images.
# Each step is skipped if its output exists, so the script can be re-run. A failed step does not stop the next.
set -uo pipefail
code_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
project_root="${QDM_PROJECT_ROOT:-$code_root}"
cd "$project_root"
: "${QDM_PYTHON:?Set the Python interpreter}"
out="$project_root/runs/high_standoff"
magic="$project_root/runs/magic_unet"
mkdir -p "$out"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg
fista=(--frozen "$project_root/experiments/fista_depths" --z-input 10 --z-delta 9.5 --skip-gate --workers 8)

if [ ! -f "$out/fista_pad_test/pad_test.json" ]; then
    rm -rf "$out/fista_pad_test"
    "$QDM_PYTHON" -u "$code_root/scripts/m2_fista_depth.py" --output "$out/fista_pad_test" \
        --pad-test 128 256 384 512 "${fista[@]}" 2>&1 | tee "$out/fista_pad_test.log"
fi

for marker in run10a.done run10b.done; do
    echo "waiting for $marker $(date)"
    while [ ! -e "$magic/$marker" ]; do sleep 60; done
done

runs=()
for run in run10a run10b run2 run3; do
    [ -f "$magic/$run/results.json" ] && runs+=("$run")
done
if [ ! -f "$out/m1/comsol.csv" ]; then
    rm -rf "$out/m1"
    "$QDM_PYTHON" -u "$code_root/scripts/m1_rescore.py" --runs "${runs[@]}" --output "$out/m1" \
        --standoffs 6 8 10 12 14 --data3-heights 10 --comsol-z-inputs 10 --save-z-input 10 --height-max 20 \
        --protocol "$code_root/HIGH_STANDOFF_PROTOCOL.md" 2>&1 | tee "$out/m1.log"
fi

if [ ! -f "$out/fista/results.json" ] || ! grep -q sim_full_mean "$out/fista/results.json"; then
    pad=$("$QDM_PYTHON" -c "import json; print(json.load(open('$out/fista_pad_test/pad_test.json'))['chosen'])") || exit 1
    rm -rf "$out/fista"
    "$QDM_PYTHON" -u "$code_root/scripts/m2_fista_depth.py" --output "$out/fista" --pad "$pad" --ladder \
        --sim-layouts 50 --sim-layouts-deep 20 --full-batch 3 "${fista[@]}" 2>&1 | tee "$out/fista.log"
fi
echo "done $(date)"
