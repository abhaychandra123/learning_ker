"""M2c (M1_M2_PROTOCOL.md): re-run one saved Group B model with its own frozen code and save full-image
COMSOL predictions at 3 -> 0.5 um. Inference only; runs on the server GPU.

The model is rebuilt from the run's args.json with the frozen experiment's own modules (imported from
<frozen>/scripts), its best.pt is loaded, and the five COMSOL configurations are predicted with the frozen
train_*.predict, exactly as when it was scored. Every per-component relative L2 must match the run's saved
comsol.csv within 1e-4 (reproduction gate) or nothing is saved.

  python m2_predict_group_b.py --frozen experiments/csc_lista_v4 --run runs/csc_lista_strict/layers_64 \
      --name csc_strict_64 --output runs/m1_m2/group_b/csc_strict_64
"""
import argparse
import csv
import importlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--frozen', required=True, help='experiments/<name>, holding scripts/')
    p.add_argument('--run', required=True, help='run folder with args.json, best.pt, comsol.csv')
    p.add_argument('--name', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--wiener', action='store_true', help='also save the Tikhonov baseline of this run')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(a.frozen).resolve() / 'scripts'))
    import torch
    legacy = (Path(a.frozen) / 'scripts' / 'train_conv_lista.py').exists()
    tr = importlib.import_module('train_conv_lista' if legacy else 'train_csc_lista')
    mod = importlib.import_module('conv_lista' if legacy else 'csc_lista')
    from simulate_fields import SimulatedDataset

    run = Path(a.run)
    args = json.loads((run / 'args.json').read_text())
    device = torch.device(a.device)
    scale = SimulatedDataset(args['dataset']).scale
    state = torch.load(run / 'best.pt', map_location=device, weights_only=True)
    if 'start' in args:                       # FISTA-Net
        fn = importlib.import_module('fista_net')
        model = fn.FISTANet(tr.DX, args['z_delta'], args['layers'], start=args['start']).to(device)
        model.load_state_dict(state['state'])
        model.eval()
        method = lambda b: model(b)[0]
        label = 'fista_net'
    else:                                     # CSC-LISTA family
        saved = torch.load(args['dictionary'], weights_only=True)
        cls = getattr(mod, 'CSCLISTA', None) or getattr(mod, 'ConvLISTA')
        kwargs = dict(chunk=args['chunk'])
        if 'momentum' in inspect.signature(cls.__init__).parameters:
            kwargs['momentum'] = args.get('momentum', False)
        canvas = args['crop'] + 2 * args['pad']
        model = cls(saved['g'].to(device), tr.DX, args['z_delta'], canvas, args['pad'], saved['lam'],
                    args['layers'], args['filter_width'], **kwargs).to(device)
        model.load_state_dict(state['state'])
        model.eval()
        method = model
        label = None
    methods = {a.name: method}
    if a.wiener:
        mu = json.loads((run / 'results.json').read_text())['wiener'][1]
        methods['wiener'] = mod.Dataless(tr.DX, args['z_delta'], mu)

    saved_rows = list(csv.DictReader(open(run / 'comsol.csv')))
    model_label = label or next(r['method'] for r in saved_rows
                                if r['method'] not in ('wiener', 'ista16', 'fista300'))
    reference = {}
    for r in saved_rows:
        key = a.name if r['method'] == model_label else r['method']
        reference[(key, r['config'], r['component'])] = float(r['relative_l2'])

    gate, preds = [], {name: {} for name in methods}
    for config, sharp, blurred in tr.comsol_pairs(args['comparison_run'], args['data3']):
        target = torch.tensor(sharp / scale, dtype=torch.float32)
        observed = torch.tensor(blurred / scale, dtype=torch.float32)
        for name, m in methods.items():
            y = tr.predict(m, observed, args['pad'], device)
            errors = tr.relative_l2(y, target)
            preds[name][config] = (y.numpy() * scale).astype(np.float32)
            for c, comp in enumerate(('Bx', 'By', 'Bz')):
                ref = reference[(name, config, comp)]
                gate.append(dict(method=name, config=config, component=comp, saved=ref,
                                 rerun=float(errors[c]), diff=float(errors[c]) - ref))
        print(config, 'done', flush=True)
    with open(out / 'gate.csv', 'w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(gate[0]))
        w.writeheader()
        w.writerows(gate)
    worst = max(abs(g['diff']) for g in gate)
    passed = worst <= 1e-4
    (out / 'gate.json').write_text(json.dumps(dict(worst_abs_diff=worst, passed=passed, frozen=a.frozen,
                                                   run=a.run, legacy_module=legacy), indent=2))
    print('worst |rerun - saved|', worst, 'passed' if passed else 'FAILED', flush=True)
    if not passed:
        raise SystemExit(1)
    for name, d in preds.items():
        np.savez_compressed(out / f'{name}_predictions.npz', **d)


if __name__ == '__main__':
    main()
