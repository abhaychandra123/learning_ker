#!/bin/bash
# M2e (M1_M2_PROTOCOL.md): two more seeds of MAGIC-UNet run 2 on the professor's server.
# Run 2's frozen code and exact arguments, except --seed and --output. Does NOT generate or delete data
# (unlike hpc/run_magic_unet.sh, which removes the data folder when its manifest is missing).
# Usage on the server, from ~/learning_ker:  bash experiments/m1_m2/hpc/run_m2_seeds.sh SEED
set -euo pipefail
seed="${1:?seed}"
root="$HOME/learning_ker"
py="$HOME/qdm-env/bin/python"
data="$root/runs/magic_unet/data"
out="$root/runs/m1_m2/magic_seed$seed"
grep -q '"noise"' "$data/manifest.json" || { echo "data lacks the per-crop noise fix"; exit 1; }
[ -e "$out" ] && { echo "$out exists; refusing"; exit 1; }
mkdir -p "$root/runs/m1_m2"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg
cd "$root"
"$py" -u experiments/magic_unet_run2/scripts/train_magic_unet.py \
    --data "$data" --output "$out" --epochs 50 --batch 32 --lr 1e-4 --noise-corr-um 0.0 \
    --workers 6 --sim-dataset "$root/runs/simulated_3_to_05_v3" --val-layouts 50 --stride-layouts 10 \
    --strides 16 32 48 --tile 64 --infer-batch 32 \
    --comparison-run "$root/runs/combined_full_200" --data3 "$root/data/data3" \
    --z-mean 3.0 --z-std 0.6 --seed "$seed" --device cuda
