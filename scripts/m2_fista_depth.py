"""M2d (M1_M2_PROTOCOL.md): FISTA on the CSC-LISTA dictionary beyond 300 iterations.

Same dictionary, operator, padding and validation crops as runs/fista_depths_lambda (frozen code of
experiments/fista_depths, imported from --frozen). One FISTA run per lambda to the largest depth, with the
iterate saved at every snapshot depth; FISTA's iterate at step s does not depend on the total number of steps,
so this equals separate runs. The step-300 snapshot must reproduce the stored fista300 trials (gate).
Lambda is chosen per depth on the first 16 validation crops only; COMSOL is scored afterwards.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--frozen', default='experiments/fista_depths')
    p.add_argument('--dataset', default='runs/simulated_3_to_05_v3')
    p.add_argument('--dictionary', default='runs/csc_lista_v2/dictionary_strong/dictionary_0.3.pt')
    p.add_argument('--reference', default='runs/fista_depths_lambda/results.json')
    p.add_argument('--comparison-run', default='runs/combined_full_200')
    p.add_argument('--data3', default='data/data3')
    p.add_argument('--output', required=True)
    p.add_argument('--snapshots', type=int, nargs='+', default=[300, 1000, 3000])
    p.add_argument('--scales', type=float, nargs='+', default=[2.29e-06, 0.001, 0.01, 0.03, 0.1])
    p.add_argument('--val-crops', type=int, default=64)
    p.add_argument('--crop', type=int, default=512)
    p.add_argument('--pad', type=int, default=128)
    p.add_argument('--chunk', type=int, default=8)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--z-delta', type=float, default=2.5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'args.json').write_text(json.dumps(vars(a), indent=2))
    sys.path.insert(0, str(Path(a.frozen).resolve() / 'scripts'))
    import torch
    import conv_lista as csc
    from simulate_fields import SimulatedDataset
    from train_conv_lista import comsol_pairs, padded, paired_crops, relative_l2, simulated

    started = time.time()
    log = lambda *x: print(f'[{time.time() - started:7.0f}s]', *x, flush=True)
    device = torch.device(a.device)
    snaps = sorted(a.snapshots)

    def fista_snapshots(b, g, lam):
        """conv_lista.code_ista (fast=True) with the field saved at every snapshot depth."""
        h = csc.transfer(b.shape[-1], 0.1, a.z_delta, b.dtype, b.device)
        physics, synth = csc.Operator(g, h, a.chunk), csc.Operator(g, chunk=a.chunk)
        n = b.shape[-1]
        mask = csc.window_mask(n, a.pad, b.device)
        step = 1 / physics.bound(n)
        x = torch.zeros(b.shape[0], g.shape[0], n, n, dtype=b.dtype, device=b.device)
        y, t, fields = x, 1.0, {}
        for s in range(1, snaps[-1] + 1):
            residual = mask * (physics.forward(y) - b)
            update = csc.shrink(y - step * physics.adjoint(residual), lam * step)
            t_next = (1 + (1 + 4 * t * t) ** .5) / 2
            y = update + (t - 1) / t_next * (update - x)
            t = t_next
            x = update
            if s in snaps:
                fields[s] = synth.forward(x)
        return fields

    @torch.no_grad()
    def predict(inputs, g, lam, batch):
        """{depth: [N, H, W]} cropped outputs, as train_conv_lista.predict for each depth."""
        out = {s: [] for s in snaps}
        for part in inputs.split(batch):
            f = fista_snapshots(padded(part.to(device), a.pad), g, lam)
            for s in snaps:
                out[s].append(f[s][..., a.pad:-a.pad, a.pad:-a.pad].cpu())
        return {s: torch.cat(v) for s, v in out.items()}

    dataset = SimulatedDataset(a.dataset)
    val = paired_crops(simulated(dataset, 'validation', a.workers), a.val_crops, a.crop, a.seed + 1)
    saved = torch.load(a.dictionary, weights_only=True)
    g, lam0 = saved['g'].to(device), saved['lam']
    log('dictionary lambda', lam0)

    trials = {}                               # scale -> {depth: score on 16 crops}

    def trial(scale):
        pred = predict(val[1][:16], g, lam0 * scale, 4)
        trials[scale] = {s: float(relative_l2(pred[s], val[0][:16]).mean()) for s in snaps}
        log(f'scale {scale:g} (lambda {lam0 * scale:g}):', trials[scale])

    for scale in a.scales:
        trial(scale)
    # gate: step-300 snapshot against the stored fista300 trials
    ref = {round(lam, 12): score for score, lam in
           json.loads(Path(a.reference).read_text())['choices']['fista300']['trials']}
    gate = [dict(scale=s, lam=lam0 * s, stored=ref[round(lam0 * s, 12)], rerun=trials[s][300])
            for s in a.scales if round(lam0 * s, 12) in ref]
    worst = max(abs(x['rerun'] - x['stored']) for x in gate)
    (out / 'gate.json').write_text(json.dumps(dict(rows=gate, worst_abs_diff=worst, passed=worst <= 1e-4),
                                              indent=2))
    log('gate worst', worst)
    if worst > 1e-4:
        raise SystemExit('gate failed')
    # choice per depth, with the protocol's single extension at a grid edge
    choice = {}
    for s in snaps:
        best = min(trials, key=lambda sc: trials[sc][s])
        extra = {0.1: 0.3, 0.001: 0.0003}.get(best)
        if extra is not None and extra not in trials:
            trial(extra)
            best = min(trials, key=lambda sc: trials[sc][s])
        choice[s] = best
    log('chosen scales per depth', choice)

    rows, validation, comsol = [], {}, []
    preds = {}
    for scale in sorted(set(choice.values())):
        depths = [s for s in snaps if choice[s] == scale]
        pv = predict(val[1], g, lam0 * scale, 4)
        for s in depths:
            validation[s] = float(relative_l2(pv[s], val[0]).mean())
        for config, sharp, blurred in comsol_pairs(a.comparison_run, a.data3):
            target = torch.tensor(sharp / dataset.scale, dtype=torch.float32)
            y = predict(torch.tensor(blurred / dataset.scale, dtype=torch.float32), g, lam0 * scale, 1)
            for s in depths:
                e = relative_l2(y[s], target)
                preds[f'fista{s}_{config}'] = (y[s].numpy() * dataset.scale).astype(np.float32)
                for c, comp in enumerate(('Bx', 'By', 'Bz')):
                    comsol.append(dict(method=f'fista{s}', lam=lam0 * scale, config=config, component=comp,
                                       relative_l2=float(e[c])))
            log(f'scale {scale:g} {config} done')
    with open(out / 'comsol.csv', 'w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(comsol[0]))
        w.writeheader()
        w.writerows(comsol)
    summary = {f'fista{s}': float(np.mean([r['relative_l2'] for r in comsol if r['method'] == f'fista{s}']))
               for s in snaps}
    (out / 'results.json').write_text(json.dumps(dict(
        trials={str(k): v for k, v in trials.items()}, choice={str(k): v for k, v in choice.items()},
        validation={f'fista{k}': v for k, v in validation.items()}, comsol_mean_relative_l2=summary),
        indent=2))
    np.savez_compressed(out / 'predictions.npz', **preds)
    log('COMSOL means', summary)


if __name__ == '__main__':
    main()
