#!/bin/bash
# CSC-LISTA and FISTA-Net at 10 -> 0.5 um (HIGH_STANDOFF_PROTOCOL.md, amendment A6). Run once on the server.
# Copies the frozen code of the two 3 um runs that are in the scoreboard and adds one thing: with QDM_Z_INPUT set,
# the simulated layouts are rendered at that input height and COMSOL is read at that height. Unset, nothing changes.
#   CSC-LISTA strict, 64 layers (33.8% at 3 um): experiments/csc_lista_v4
#   FISTA-Net run 2 (33.3% at 3 um):             experiments/fista_net_code_start
set -euo pipefail
cd "${QDM_PROJECT_ROOT:-$HOME/learning_ker}"
root=experiments/high_standoff_methods
[ -e "$root" ] && { echo "$root exists"; exit 1; }
mkdir -p "$root"
for pair in csc_lista:csc_lista_v4 fista_net:fista_net_code_start; do
    name=${pair%%:*}; source=${pair##*:}
    mkdir -p "$root/$name"
    cp -r "experiments/$source/scripts" "experiments/$source/tests" "$root/$name/"
    find "$root/$name" -name __pycache__ -prune -exec rm -rf {} +
    "${QDM_PYTHON:-$HOME/qdm-env/bin/python}" - "$root/$name/scripts/train_conv_lista.py" <<'PY'
import sys
path = sys.argv[1]
s = open(path).read()
edits = [
    ("DATA3 = ('C1_C2', 'C1_C2_C3', 'C1_C2_C3_C4')\n",
     "DATA3 = ('C1_C2', 'C1_C2_C3', 'C1_C2_C3_C4')\n"
     "# 10 um runs: QDM_Z_INPUT renders the layouts and reads COMSOL at this input height (default 3 um, unchanged).\n"
     "Z_INPUT = float(__import__('os').environ.get('QDM_Z_INPUT', 3.0))\n"),
    ("    jobs = [(r, dataset.settings) for r in dataset.recipes(split)]\n",
     "    settings = dict(dataset.settings, z_input=Z_INPUT) if Z_INPUT != 3.0 else dataset.settings\n"
     "    jobs = [(r, settings) for r in dataset.recipes(split)]\n"),
    ("f'{geometry}_z03.0_to_z00.5.npz'", "f'{geometry}_z{Z_INPUT:04.1f}_to_z00.5.npz'"),
    ("load_csv(f'{stem}_Z3p0um_600uA_0p1PR_MF_B.csv'))",
     "load_csv(f'{stem}_Z' + f'{Z_INPUT:.1f}'.replace('.', 'p') + 'um_600uA_0p1PR_MF_B.csv'))"),
]
for old, new in edits:
    assert s.count(old) == 1, old
    s = s.replace(old, new)
open(path, 'w').write(s)
PY
done
sha256sum "$root"/*/scripts/*.py > "$root/code_sha256.txt"
echo prepared "$root"
