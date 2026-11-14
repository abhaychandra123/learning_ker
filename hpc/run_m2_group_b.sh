#!/bin/bash
# M2c (M1_M2_PROTOCOL.md): save COMSOL predictions of every Group B model with its frozen code.
# Usage on the server, from ~/learning_ker:  bash experiments/m1_m2/hpc/run_m2_group_b.sh
set -uo pipefail
cd "$HOME/learning_ker"
py="$HOME/qdm-env/bin/python"
script=experiments/m1_m2/scripts/m2_predict_group_b.py
out=runs/m1_m2/group_b
mkdir -p "$out"
export OMP_NUM_THREADS=4
run() {  # name frozen run [extra]
    echo "=== $1"
    "$py" -u "$script" --name "$1" --frozen "experiments/$2" --run "runs/$3" --output "$out/$1" "${@:4}" \
        || echo "!!! $1 failed"
}
run csc_strict_64      csc_lista_v4         csc_lista_strict/layers_64 --wiener
run csc_strict_32      csc_lista_v4         csc_lista_strict/layers_32
run csc_stream_64      csc_lista_stream     csc_lista_stream/layers_64
run csc_relative_64    csc_lista_relative   csc_lista_relative/layers_64_stored
run csc_relative_64_stream csc_lista_relative csc_lista_relative/layers_64_stream
run lfista_32          csc_lista_lfista     csc_lista_lfista/layers_32
run csc_plain_16       csc_lista_v2         csc_lista_v2/full_plain
run fista_net_run2     fista_net_code_start fista_net/run_2
echo "all done"
