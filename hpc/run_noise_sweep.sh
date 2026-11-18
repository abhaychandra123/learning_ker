#!/bin/bash
# Noise sweep for the 20 Nov evaluation (scripts/noise_sweep.py). Inference only, no training.
#   1. MAGIC-UNet run B at 10 -> 0.5 um and run 2 at 3 -> 0.5 um, 20 layouts, noise 0-100 % of each component range
#      (67 % is about the paper's sigma_n = 1);
#   2. FISTA at its reference depth, 5 layouts, 3 um (925 iterations) then 10 um (3,000 iterations), noise 0-50 %;
#      above 2.5 % lambda x 1, 3, 10 are all scored.
# Each step starts at level 'native' (each layout's own 1-2.5 % noise), which is a reproduction gate against the
# saved scores. Each step is skipped if its scores.csv exists and the step finished ("done" in its log).
set -uo pipefail
cd "${QDM_PROJECT_ROOT:-$HOME/learning_ker}"
py="${QDM_PYTHON:-$HOME/qdm-env/bin/python}"
s=experiments/noise_sweep/scripts
out=runs/noise_sweep
mkdir -p "$out"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg
batch="${QDM_FISTA_BATCH:-1}"

step() {   # name, then arguments
    local name=$1; shift
    if [ -f "$out/$name.log" ] && grep -q '\] done' "$out/$name.log"; then echo "skip $name"; return; fi
    rm -rf "${out:?}/$name"
    echo "start $name $(date)"
    "$py" -u "$s/noise_sweep.py" "$@" --output "$out/$name" > "$out/$name.log" 2>&1
    local rc=$?
    echo "end $name $(date) (exit $rc)"
}

step magic_10um magic --run run10b --z-input 10 --layouts 20 --levels 0 1 2 5 10 25 50 67 100 \
    --mu-curve runs/high_standoff/m1/mu_curve.json --m1-scores runs/high_standoff/m1/sim_scores.csv --height-max 20
step magic_3um magic --run run2 --z-input 3 --layouts 20 --levels 0 1 2 5 10 25 50 67 100 \
    --mu-curve runs/m1_m2/m1/mu_curve.json --m1-scores runs/m1_m2/m1/sim_scores.csv --height-max 12
# Re-solved COMSOL L1_C1 (tolerance 1e-8), every method again with SSIM and RMSE added (same code as amendment A7;
# the original-export pass stays only as the reproduction gate). Output: runs/noise_sweep/resolved_ssim.
if ! grep -q "resolved ssim done" "$out/resolved_ssim.log" 2>/dev/null; then
    rm -rf "$out/resolved_ssim"
    echo "start resolved_ssim $(date)"
    r=$s/score_resolved_comsol.py
    (export QDM_RESOLVED_OUT=$out/resolved_ssim
     "$py" -u $r magic
     "$py" -u $r fista
     "$py" -u $r net experiments/csc_lista_v4/scripts runs/csc_lista_strict/layers_64 csc 3 0.03162277660168379 csc3
     "$py" -u $r net experiments/fista_net_code_start/scripts runs/fista_net/run_2 fistanet 3 none fistanet3
     "$py" -u $r net experiments/high_standoff_methods/csc_lista/scripts runs/high_standoff/csc_lista_10um csc 10 0.1 csc10
     "$py" -u $r net experiments/high_standoff_methods/fista_net/scripts runs/high_standoff/fista_net_10um fistanet 10 none fistanet10
     echo "resolved ssim done $(date)") > "$out/resolved_ssim.log" 2>&1
    echo "end resolved_ssim $(date)"
fi
step fista_3um fista --z-input 3 --layouts 5 --levels 0 1 2 5 10 25 50 --images-per-batch "$batch" \
    --fista-results runs/high_standoff/fista_3um_dense/results.json --fista-sim none
step fista_10um fista --z-input 10 --layouts 5 --levels 0 1 2 5 10 25 50 --images-per-batch "$batch" \
    --fista-results runs/high_standoff/fista/results.json --fista-sim runs/high_standoff/fista/sim_full.csv
echo "all done $(date)"
