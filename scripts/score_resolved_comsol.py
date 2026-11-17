"""Amendment A7 of HIGH_STANDOFF_PROTOCOL.md: every method on the re-solved COMSOL L1_C1 (tolerance 1e-8).

Server side, inference only. Each method is run on the ORIGINAL L1_C1 pair first (reproduction gate against its saved
scores) and then on the RE-SOLVED pair (runs/comsol_resolved/comparison). Nothing is chosen on these data.

  python score_resolved_comsol.py magic
  python score_resolved_comsol.py fista
  python score_resolved_comsol.py net <code dir> <run dir> <csc|fistanet> <z_input> <wiener mu or none> <tag>

Outputs in runs/comsol_resolved/: scores_<tag>.csv (one row per method, z, data, component), gate_<tag>.json, and
export/<tag>_part.npz/.json with Bz maps for the figures (keys res<z>_<data>_<method>, as export_high_standoff.py).
"""
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path.cwd()
OUT = ROOT / 'runs/comsol_resolved'
DATA = {'original': ROOT / 'runs/combined_full_200/comparison', 'resolved': OUT / 'comparison'}
COMPONENTS = ('Bx', 'By', 'Bz')
sys.path.insert(0, str(ROOT / 'experiments/magic_unet_10um/export'))


def pair(data, z_in):
    with np.load(DATA[data] / f'L1_C1_z{z_in:04.1f}_to_z00.5.npz') as z:
        return z['sharp'].astype(np.float64), z['blurred'].astype(np.float64)


def scores(pred, truth):
    err = np.asarray(pred, np.float64) - truth
    l2 = np.sqrt((err ** 2).sum((-2, -1)) / (truth ** 2).sum((-2, -1)))
    nrmse = np.sqrt((err ** 2).mean((-2, -1))) / (truth.max((-2, -1)) - truth.min((-2, -1)))
    return l2, nrmse


class Recorder:
    def __init__(self, tag):
        from export_high_standoff import add
        self.tag, self.add, self.rows, self.gate, self.arrays, self.meta = tag, add, [], [], {}, {}

    def record(self, method, z_in, data, field, truth, saved=None, tol=1e-4, h_star=None):
        l2, nrmse = scores(field, truth)
        for c, comp in enumerate(COMPONENTS):
            self.rows.append(dict(method=method, z_input=z_in, data=data, component=comp, relative_l2=float(l2[c]),
                                  nrmse=float(nrmse[c]), h_star='' if h_star is None else round(float(h_star), 3)))
        if saved is not None:
            diff = max(abs(float(l2[c]) - saved[comp]) for c, comp in enumerate(COMPONENTS))
            self.gate.append(dict(method=method, z_input=z_in, worst_abs_diff=diff, tolerance=tol, passed=diff <= tol))
            print(f'gate {method} z={z_in}: worst |rerun - saved| = {diff:.2e} ({"pass" if diff <= tol else "FAIL"})',
                  flush=True)
        key = f'res{z_in:g}_{data}_{method}'
        self.add(self.arrays, self.meta, key, np.asarray(field, np.float64), truth)
        if f'res{z_in:g}_{data}_target' not in self.arrays:
            self.add(self.arrays, self.meta, f'res{z_in:g}_{data}_target', truth, truth)
        print(f'{method:22s} z={z_in:4.1f} {data:9s} L2 {np.round(100 * l2, 1)} mean {100 * l2.mean():.1f}%'
              + ('' if h_star is None else f'  h* {h_star:.2f}'), flush=True)

    def write(self):
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'export').mkdir(exist_ok=True)
        with open(OUT / f'scores_{self.tag}.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            w.writeheader()
            w.writerows(self.rows)
        (OUT / f'gate_{self.tag}.json').write_text(json.dumps(self.gate, indent=2))
        np.savez_compressed(OUT / 'export' / f'{self.tag}_part.npz', **self.arrays)
        (OUT / 'export' / f'{self.tag}_part.json').write_text(json.dumps(self.meta, indent=2))
        if not all(g['passed'] for g in self.gate):
            print('WARNING: a reproduction gate failed; see', OUT / f'gate_{self.tag}.json', flush=True)


def saved_rows(path, keep):
    """{component: relative_l2} from a saved csv, for rows where keep(row) is true."""
    out = {}
    for r in csv.DictReader(open(path)):
        if keep(r):
            out[r['component']] = float(r.get('relative_l2') or r.get('raw'))
    return out


def magic_part():
    sys.path.insert(0, str(ROOT / 'experiments/magic_unet_10um/scripts'))
    import torch
    import m12_common as mc
    from m1_rescore import Corrector, Sheet, current, load_model, read_height
    device = torch.device('cuda')
    rec = Recorder('magic')
    plan = [(10.0, ('run10b', 'run10a'), ROOT / 'runs/high_standoff/m1', 20.0),
            (3.0, ('run2',), ROOT / 'runs/m1_m2/m1', mc.EDGE[1])]
    for z_in, runs, m1, h_max in plan:
        curves = json.loads((m1 / 'mu_curve.json').read_text())
        for run in runs:
            folder = ROOT / 'runs/magic_unet' / run
            model, beta, factor = load_model(folder / 'final.pt', device)
            stride = json.loads((folder / 'results.json').read_text())['stride']
            for data in ('original', 'resolved'):
                truth, b = pair(data, z_in)
                j = current(model, beta, factor, stride, b, device)
                sheet = Sheet(j, device)
                pred = sheet.field64(0.5).cpu().numpy()
                h, _, _ = read_height(sheet, b, device, h_max)
                mu = mc.mu_at(h, curves[run]['standoffs'], curves[run]['mu'])
                fixed = Corrector(pred, b, h - 0.5, device)(mu).cpu().numpy()
                for variant, field in (('raw', pred), ('dc_est', fixed)):
                    saved = None
                    if data == 'original':
                        saved = saved_rows(m1 / 'comsol.csv', lambda r: r['run'] == run and r['variant'] == variant
                                           and r['config'] == 'L1_C1' and float(r['z_input']) == z_in
                                           and float(r['z_target']) == 0.5)
                    rec.record(f'{run}_{variant}', z_in, data, field, truth, saved, h_star=h)
    rec.write()


def fista_part():
    sys.path.insert(0, str(ROOT / 'experiments/fista_depths/scripts'))
    import torch
    import conv_lista as csc
    from simulate_fields import SimulatedDataset
    from train_conv_lista import padded
    device = torch.device('cuda')
    scale = SimulatedDataset('runs/simulated_3_to_05_v3').scale
    stored = torch.load('runs/csc_lista_v2/dictionary_strong/dictionary_0.3.pt', weights_only=True)
    g, lam0 = stored['g'].to(device), stored['lam']
    rec = Recorder('fista')
    plan = [(10.0, ROOT / 'runs/high_standoff/fista', (3000,)),
            (3.0, ROOT / 'runs/high_standoff/fista_3um_dense', (925, 1000))]
    for z_in, folder, depths in plan:
        res = json.loads((folder / 'results.json').read_text())
        pad, dz = res['pad'], res['z_delta']
        for depth_group in sorted({res['choice'][str(d)] for d in depths}):
            group = [d for d in depths if res['choice'][str(d)] == depth_group]
            lam = lam0 * depth_group
            for data in ('original', 'resolved'):
                truth, blurred = pair(data, z_in)
                b = padded(torch.tensor(blurred / scale, dtype=torch.float32, device=device), pad)
                fields = {}
                with torch.no_grad():     # the loop of m2_fista_depth.fista_snapshots
                    n = b.shape[-1]
                    physics = csc.Operator(g, csc.transfer(n, 0.1, dz, b.dtype, b.device), 8)
                    synth = csc.Operator(g, chunk=8)
                    mask = csc.window_mask(n, pad, b.device)
                    step = 1 / physics.bound(n)
                    x = torch.zeros(b.shape[0], g.shape[0], n, n, dtype=b.dtype, device=b.device)
                    y, t = x, 1.0
                    for s in range(1, max(group) + 1):
                        update = csc.shrink(y - step * physics.adjoint(mask * (physics.forward(y) - b)), lam * step)
                        t_next = (1 + (1 + 4 * t * t) ** .5) / 2
                        y = update + (t - 1) / t_next * (update - x)
                        t, x = t_next, update
                        if s in group:
                            fields[s] = synth.forward(x)[..., pad:-pad, pad:-pad].cpu().numpy().astype(np.float64) * scale
                for d in group:
                    saved = None
                    if data == 'original':
                        saved = saved_rows(folder / 'comsol.csv', lambda r: r['method'] == f'fista{d}'
                                           and r['config'] == 'L1_C1')
                    rec.record(f'fista{d}', z_in, data, fields[d], truth, saved, tol=2e-4)
    rec.write()


def net_part(code, run_dir, kind, z_in, wiener_mu, tag):
    sys.path.insert(0, str((ROOT / code).resolve()))
    import torch
    import train_conv_lista as t
    from simulate_fields import SimulatedDataset
    device = torch.device('cuda')
    run_dir = ROOT / run_dir
    args = json.loads((run_dir / 'args.json').read_text())
    state = torch.load(run_dir / 'best.pt', map_location=device, weights_only=True)
    if kind == 'csc':
        import conv_lista as cl
        saved = torch.load(args['dictionary'], weights_only=True)
        model = cl.ConvLISTA(saved['g'].to(device), t.DX, args['z_delta'], args['crop'] + 2 * args['pad'], args['pad'],
                             saved['lam'], args['layers'], args['filter_width'], chunk=args['chunk'],
                             momentum=args['momentum']).to(device)
        model.load_state_dict(state['state'])
        methods, own = {'csc_lista': model}, 'conv_lista'
        if wiener_mu != 'none':
            methods['tikhonov'] = cl.Dataless(t.DX, args['z_delta'], float(wiener_mu))
    else:
        import fista_net as fn
        model = fn.FISTANet(t.DX, args['z_delta'], args['layers'], start=args['start']).to(device)
        model.load_state_dict(state['state'])
        methods, own = {'fista_net': lambda b: model(b)[0]}, 'fista_net'
    model.eval()
    scale = SimulatedDataset(args['dataset']).scale
    saved_method = {'csc_lista': own, 'fista_net': own, 'tikhonov': 'wiener'}
    rec = Recorder(tag)
    for data in ('original', 'resolved'):
        truth, blurred = pair(data, z_in)
        for name, run in methods.items():
            with torch.no_grad():
                y = t.predict(run, torch.tensor(blurred / scale, dtype=torch.float32), args['pad'], device)
            field = y.cpu().numpy().astype(np.float64) * scale
            saved = None
            if data == 'original':     # older runs saved CSC-LISTA as 'csc_lista', newer ones as 'conv_lista'
                saved = saved_rows(run_dir / 'comsol.csv', lambda r: r['config'] == 'L1_C1'
                                   and r['method'] in (saved_method[name], name))
            rec.record(name, z_in, data, field, truth, saved)
    rec.write()


if __name__ == '__main__':
    part = sys.argv[1]
    if part == 'magic':
        magic_part()
    elif part == 'fista':
        fista_part()
    else:
        net_part(sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]), sys.argv[6], sys.argv[7])
