"""Score the 10 -> 0.5 um CSC-LISTA or FISTA-Net run (amendment A6) like FISTA, and export its maps for the figures.

Inference only, on the server. The best checkpoint of runs/high_standoff/<kind>_10um is loaded with the run's own
code and arguments. Three outputs:

  runs/high_standoff/methods_10um/<kind>/comsol.csv     the 15 COMSOL pairs (re-run; compared with the run's own csv)
  runs/high_standoff/methods_10um/<kind>/sim_full.csv   full images of the first 20 simulated validation layouts
  runs/high_standoff/export/<kind>_part.npz, .json      Bz maps of the two figure layouts and of L1_C1 and IC_4um
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

COMPONENTS = ('Bx', 'By', 'Bz')
Z_IN = 10.0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('kind', choices=('csc_lista', 'fista_net'))
    p.add_argument('--code', default='experiments/high_standoff_methods')
    p.add_argument('--export-code', default='experiments/magic_unet_10um/export')
    p.add_argument('--results', default='runs/high_standoff')
    p.add_argument('--layouts', type=int, default=20)
    a = p.parse_args()
    os.environ['QDM_Z_INPUT'] = str(Z_IN)
    sys.path.insert(0, str(Path(a.export_code).resolve()))
    sys.path.insert(0, str((Path(a.code) / a.kind / 'scripts').resolve()))
    import torch
    import train_conv_lista as t
    from export_high_standoff import add
    from simulate_fields import SimulatedDataset
    device = torch.device('cuda')
    run_dir = Path(a.results) / f'{a.kind}_10um'
    args = json.loads((run_dir / 'args.json').read_text())
    state = torch.load(run_dir / 'best.pt', map_location=device, weights_only=True)
    if a.kind == 'csc_lista':
        import conv_lista as cl
        saved = torch.load(args['dictionary'], weights_only=True)
        model = cl.ConvLISTA(saved['g'].to(device), t.DX, args['z_delta'], args['crop'] + 2 * args['pad'],
                             args['pad'], saved['lam'], args['layers'], args['filter_width'], chunk=args['chunk'],
                             momentum=args['momentum']).to(device)
        model.load_state_dict(state['state'])
        run, own = model, 'conv_lista'
    else:
        import fista_net as fn
        model = fn.FISTANet(t.DX, args['z_delta'], args['layers'], start=args['start']).to(device)
        model.load_state_dict(state['state'])
        run, own = (lambda b: model(b)[0]), 'fista_net'
    model.eval()
    print(a.kind, 'best checkpoint: step', state.get('step'), 'validation', state.get('score'), flush=True)
    dataset = SimulatedDataset(args['dataset'])
    scale, pad = dataset.scale, args['pad']

    @torch.no_grad()
    def field(blurred):
        """[3, H, W] uT in, [3, H, W] uT out."""
        y = t.predict(run, torch.tensor(blurred / scale, dtype=torch.float32), pad, device)
        return y.cpu().numpy().astype(np.float64) * scale

    out = Path(a.results) / 'methods_10um' / a.kind
    out.mkdir(parents=True, exist_ok=True)
    arrays, meta = {}, dict(kind=a.kind, step=state.get('step'), validation=state.get('score'))

    # COMSOL, with a gate against the run's own scores
    saved_rows = [r for r in csv.DictReader(open(run_dir / 'comsol.csv')) if r.get('method', own) == own]
    ref = {(r['config'], r['component']): float(r['relative_l2']) for r in saved_rows}
    rows, worst = [], 0.0
    for name, sharp, blurred in t.comsol_pairs(args['comparison_run'], args['data3']):
        y = field(blurred)
        e = np.sqrt(((y - sharp) ** 2).sum((-2, -1)) / (sharp ** 2).sum((-2, -1)))
        for i, c in enumerate(COMPONENTS):
            rows.append(dict(config=name, component=c, relative_l2=float(e[i])))
            worst = max(worst, abs(float(e[i]) - ref[name, c]))
        if name in ('L1_C1', 'IC_4um'):
            add(arrays, meta, f'comsol_{name}_{a.kind}', y, sharp)
    print('COMSOL mean', np.mean([r['relative_l2'] for r in rows]), 'worst |rerun - saved|', worst, flush=True)
    with open(out / 'comsol.csv', 'w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # simulation: full images of the first layouts, as FISTA
    chosen = json.loads((Path(a.results) / 'export' / 'magic_part.json').read_text())['chosen']
    settings = dict(dataset.settings, z_input=Z_IN)
    rows = []
    for recipe in dataset.recipes('validation')[:a.layouts]:
        target, blurred = t.render((recipe, settings))
        y = field(blurred)
        e = np.sqrt(((y - target) ** 2).sum((-2, -1)) / (target.astype(np.float64) ** 2).sum((-2, -1)))
        rows.append(dict(layout=recipe['id'], **{c: float(e[i]) for i, c in enumerate(COMPONENTS)}))
        for tag, layout in chosen.items():
            if layout == recipe['id']:
                add(arrays, meta, f'sim_{tag}_{a.kind}', y, target, dict(layout=layout))
    with open(out / 'sim_full.csv', 'w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    mean = float(np.mean([[r[c] for c in COMPONENTS] for r in rows]))
    print('simulation, full images:', len(rows), 'layouts, mean', mean, flush=True)
    (out / 'summary.json').write_text(json.dumps(dict(
        kind=a.kind, step=state.get('step'), validation=state.get('score'), sim_full_layouts=len(rows),
        sim_full_mean_relative_l2=mean, comsol_gate_worst_abs_diff=worst), indent=2))
    export = Path(a.results) / 'export'
    np.savez_compressed(export / f'{a.kind}_part.npz', **arrays)
    (export / f'{a.kind}_part.json').write_text(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
