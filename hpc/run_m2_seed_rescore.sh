#!/bin/bash
# Amendment A3 (M1_M2_PROTOCOL.md): once both seeds have finished training and their standard evaluation,
# re-score them exactly as M1 (own gate, own mu(h)), with simulated standoffs 2-4 um only.
# Usage on the server, from ~/learning_ker:  bash experiments/m1_m2/hpc/run_m2_seed_rescore.sh
set -euo pipefail
cd "$HOME/learning_ker"
for s in 1 2; do
    until [ -f "runs/m1_m2/magic_seed$s/results.json" ]; do sleep 120; done
done
OMP_NUM_THREADS=8 "$HOME/qdm-env/bin/python" -u experiments/m1_m2/scripts/m1_rescore.py \
    --magic-root runs/m1_m2 --runs magic_seed1 magic_seed2 --standoffs 2 2.5 3 3.5 4 \
    --output runs/m1_m2/m1_seeds
