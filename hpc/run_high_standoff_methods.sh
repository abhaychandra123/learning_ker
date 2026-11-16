#!/bin/bash
# Queue for amendment A6 of HIGH_STANDOFF_PROTOCOL.md, one job at a time on the GPU:
#   1. CSC-LISTA strict, 64 layers, at 10 -> 0.5 um (the 3 um run's arguments, z_delta 9.5);
#   2. FISTA validation curve at 10 um extended to 10,000 iterations (curve only);
#   3. FISTA-Net at 10 -> 0.5 um (run 2's arguments, z_delta 9.5).
# Each step is skipped if its output folder has results.json; a failed step does not stop the next.
set -uo pipefail
cd "${QDM_PROJECT_ROOT:-$HOME/learning_ker}"
py="${QDM_PYTHON:-$HOME/qdm-env/bin/python}"
code=experiments/high_standoff_methods
out=runs/high_standoff
export QDM_Z_INPUT=10 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg
step() { name=$1; shift; if [ -f "$out/$name/results.json" ]; then echo "$name: done already"; return; fi
         rm -rf "$out/$name"; echo "$name: start $(date)"; "$@" > "logs/high_standoff_$name.log" 2>&1 \
             || echo "$name failed" >> "logs/high_standoff_$name.log"; echo "$name: end $(date)"; }

step csc_lista_10um "$py" -u $code/csc_lista/scripts/train_conv_lista.py --dataset runs/simulated_3_to_05_v3 \
    --dictionary runs/csc_lista_v2/dictionary_strong/dictionary_0.3.pt --filter-width 155 --layers 64 --steps 1000 \
    --eval-every 100 --workers 8 --skip-fista --loss mse --theta-floor 0 --batch 8 --z-delta 9.5 \
    --output $out/csc_lista_10um

step fista_curve_10k "$py" -u experiments/magic_unet_10um/scripts/m2_fista_depth.py \
    --output $out/fista_curve_10k --pad 128 --curve-every 25 --snapshots 3000 10000 --scales 0.001 0.0003 0.0001 \
    --curve-only --frozen experiments/fista_depths --z-input 10 --z-delta 9.5 --skip-gate --workers 8

step fista_net_10um "$py" -u $code/fista_net/scripts/train_fista_net.py --dataset runs/simulated_3_to_05_v3 \
    --batch 24 --epochs 360 --evals 90 --start code --z-delta 9.5 --output $out/fista_net_10um
echo "queue done $(date)"
