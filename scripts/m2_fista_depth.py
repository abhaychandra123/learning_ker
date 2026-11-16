"""M2d (M1_M2_PROTOCOL.md): FISTA on the CSC-LISTA dictionary beyond 300 iterations.

Same dictionary, operator, padding and validation crops as runs/fista_depths_lambda (frozen code of
experiments/fista_depths, imported from --frozen). One FISTA run per lambda to the largest depth, with the
iterate saved at every snapshot depth; FISTA's iterate at step s does not depend on the total number of steps,
so this equals separate runs. The step-300 snapshot must reproduce the stored fista300 trials (gate).
Lambda is chosen per depth on the first 16 validation crops only; COMSOL is scored afterwards.

High-standoff use (HIGH_STANDOFF_PROTOCOL.md): --z-input 10 --z-delta 9.5 renders the same validation layouts
at a 10 um input and scores COMSOL pairs at 10 -> 0.5 um; --skip-gate (no stored 10 um trials exist);
--ladder extends the lambda grid while the best value sits at an end; --pad-test picks the padding on simulation;
--sim-layouts scores full simulated images at the reference depth (lowest 64-crop validation error).
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
    p.add_argument('--z-input', type=float, help='Input height (um) for simulation and COMSOL; default 3 -> 0.5')
    p.add_argument('--skip-gate', action='store_true', help='No comparison with stored fista300 trials')
    p.add_argument('--ladder', action='store_true', help='Keep extending lambda while the best is at a grid end')
    p.add_argument('--pad-test', type=int, nargs='+', metavar='PAD',
                   help='Only compare these paddings (FISTA-300, every scale, 16 crops), write pad_test.json')
    p.add_argument('--pad-tolerance', type=float, default=0.005)
    p.add_argument('--pad-test-depth', type=int, default=300, help='Iterations of the padding test')
    p.add_argument('--sim-layouts', type=int, default=0, help='Full simulated images scored at the reference depth')
    p.add_argument('--sim-layouts-deep', type=int, default=0,
                   help='Number used instead when the reference depth is above 1,000 iterations')
    p.add_argument('--full-batch', type=int, default=1, help='Components per batch for full images')
    p.add_argument('--curve-every', type=int, default=0,
                   help='Record the validation error every N iterations and take the reference depth from\n'
                        'that curve; --snapshots are then the depths always reported (multiples of N)')
    p.add_argument('--sim-deep-above', type=int, default=1000,
                   help='Use --sim-layouts-deep when the reference depth is above this')
    p.add_argument('--curve-only', action='store_true',
                   help='With --curve-every: stop after the validation curve (no COMSOL, no full images)')
    p.add_argument('--keep-depths', type=int, nargs='+',
                   help='Save COMSOL predictions only at these depths and the reference depth (default: all)')
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
    snaps = [a.pad_test_depth] if a.pad_test else sorted(a.snapshots)

    def fista_snapshots(b, g, lam, upto=None, score=None):
        """conv_lista.code_ista (fast=True) with the field saved at every snapshot depth up to `upto`.

        With `score` (the cropped targets) nothing is kept except the summed relative L2 of the batch at every
        --curve-every-th iteration: {depth: sum over the batch}.
        """
        h = csc.transfer(b.shape[-1], 0.1, a.z_delta, b.dtype, b.device)
        physics, synth = csc.Operator(g, h, a.chunk), csc.Operator(g, chunk=a.chunk)
        n = b.shape[-1]
        mask = csc.window_mask(n, a.pad, b.device)
        step = 1 / physics.bound(n)
        x = torch.zeros(b.shape[0], g.shape[0], n, n, dtype=b.dtype, device=b.device)
        y, t, fields = x, 1.0, {}
        for s in range(1, (upto or snaps[-1]) + 1):
            residual = mask * (physics.forward(y) - b)
            update = csc.shrink(y - step * physics.adjoint(residual), lam * step)
            t_next = (1 + (1 + 4 * t * t) ** .5) / 2
            y = update + (t - 1) / t_next * (update - x)
            t = t_next
            x = update
            if score is not None:
                if s % a.curve_every == 0:
                    field = synth.forward(x)[..., a.pad:-a.pad, a.pad:-a.pad]
                    fields[s] = float(relative_l2(field, score).sum())
            elif s in snaps:
                fields[s] = synth.forward(x)
        return fields

    @torch.no_grad()
    def predict(inputs, g, lam, batch, upto=None):
        """{depth: [N, H, W]} cropped outputs, as train_conv_lista.predict for each depth."""
        keep = [s for s in snaps if s <= (upto or snaps[-1])]
        out = {s: [] for s in keep}
        for part in inputs.split(batch):
            f = fista_snapshots(padded(part.to(device), a.pad), g, lam, upto)
            for s in keep:
                out[s].append(f[s][..., a.pad:-a.pad, a.pad:-a.pad].cpu())
        return {s: torch.cat(v) for s, v in out.items()}

    def comsol_inputs():
        """(config, sharp, blurred) in uT: the frozen 3 -> 0.5 um list, or the five configurations at --z-input."""
        if a.z_input is None:
            yield from comsol_pairs(a.comparison_run, a.data3)
            return
        import pandas as pd

        def table(path):
            t = pd.read_csv(path, comment='%', header=None, usecols=[2, 3, 4]).to_numpy()
            return t.T.reshape(3, 1400, 1400) * 1e6
        for geometry in ('L1_C1', 'IC_4um'):
            with np.load(Path(a.comparison_run) / 'comparison' / f'{geometry}_z{a.z_input:04.1f}_to_z00.5.npz') as z:
                yield geometry, z['sharp'].astype(np.float64), z['blurred'].astype(np.float64)
        tag = f'{a.z_input:.1f}'.replace('.', 'p')
        for config in ('C1_C2', 'C1_C2_C3', 'C1_C2_C3_C4'):
            stem = Path(a.data3) / f'Coil_{config}' / f'L1_{config}'
            yield (config, table(f'{stem}_Z0p5um_600uA_0p1PR_MF_B.csv'),
                   table(f'{stem}_Z{tag}um_600uA_0p1PR_MF_B.csv'))

    dataset = SimulatedDataset(a.dataset)
    if a.z_input is not None:       # same validation layouts, noise seeds and crop positions, other input height
        dataset.settings = dict(dataset.settings, z_input=float(a.z_input))
    full = simulated(dataset, 'validation', a.workers)
    val = paired_crops(full, a.val_crops, a.crop, a.seed + 1)
    saved = torch.load(a.dictionary, weights_only=True)
    g, lam0 = saved['g'].to(device), saved['lam']
    log('dictionary lambda', lam0)

    if a.pad_test:
        table = {}
        for pad in a.pad_test:
            a.pad = pad
            table[pad] = {}
            for scale in a.scales:
                pred = predict(val[1][:16], g, lam0 * scale, 4)[a.pad_test_depth]
                table[pad][scale] = float(relative_l2(pred, val[0][:16]).mean())
            log(f'pad {pad}:', table[pad])
        best = {pad: min(v.values()) for pad, v in table.items()}
        chosen = min(pad for pad in best if best[pad] <= min(best.values()) + a.pad_tolerance)
        (out / 'pad_test.json').write_text(json.dumps(dict(
            trials={str(k): {str(sc): e for sc, e in v.items()} for k, v in table.items()},
            best={str(k): v for k, v in best.items()}, tolerance=a.pad_tolerance, chosen=chosen), indent=2))
        log('best per pad', best, '-> chosen', chosen)
        return

    floor = min(a.scales)                     # 2.29e-6: effectively no penalty, not a grid end to extend from

    def extension(best, tried):
        """Next scale to try when `best` sits at an end of the tried grid, else None (--ladder)."""
        inner = sorted(k for k in tried if k != floor)
        if best == inner[-1]:
            return next((x for x in (0.3, 1.0, 3.0) if x > inner[-1]), None)
        if best == inner[0] or best == floor:
            return next((x for x in (0.0003, 0.0001, 0.00003, 0.00001) if x < inner[0]), None)
        return None

    def run_gate(trials):
        """Step-300 errors on the 16 crops against the stored fista300 trials (3 -> 0.5 um only)."""
        if a.skip_gate:
            return
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

    notes = {}
    if a.curve_every:
        # Validation error recorded every --curve-every iterations. Lambda is chosen per depth on the first 16
        # crops; the reference depth is the minimum of the 64-crop curve. COMSOL is scored only after that.
        if any(s % a.curve_every for s in a.snapshots):
            raise SystemExit('--snapshots must be multiples of --curve-every')
        report = sorted(a.snapshots)
        grid = list(range(report[0], report[-1] + 1, a.curve_every))

        @torch.no_grad()
        def curve(inputs, targets, lam):
            total = {}
            for part, truth in zip(inputs.split(4), targets.split(4)):
                errors = fista_snapshots(padded(part.to(device), a.pad), g, lam, score=truth.to(device))
                for s, e in errors.items():
                    total[s] = total.get(s, 0.0) + e
            return {s: total[s] / len(inputs) for s in grid}

        trials = {}                           # scale -> {depth: error on 16 crops}

        def trial(scale):
            trials[scale] = curve(val[1][:16], val[0][:16], lam0 * scale)
            best = min(grid, key=trials[scale].get)
            log(f'scale {scale:g} (lambda {lam0 * scale:g}): best {trials[scale][best]:.4f} at {best};',
                {s: round(trials[scale][s], 4) for s in report})

        for scale in a.scales:
            trial(scale)
        run_gate(trials)
        grew = True
        while grew:                           # extend lambda until no depth has its best at a grid end
            grew = False
            for s in grid:
                best = min(trials, key=lambda sc: trials[sc][s])
                more = extension(best, trials) if a.ladder else {0.1: 0.3, 0.001: 0.0003}.get(best)
                if more is not None and more not in trials:
                    trial(more)
                    grew = True
                    break
        choice = {s: min(trials, key=lambda sc: trials[sc][s]) for s in grid}
        curves = {}
        for scale in sorted(set(choice.values())):
            curves[scale] = curve(val[1], val[0], lam0 * scale)
            log(f'64-crop curve for scale {scale:g} done')
        validation = {s: curves[choice[s]][s] for s in grid}
        reference = min(grid, key=validation.get)
        near = [s for s in grid if validation[s] <= validation[reference] + 0.001]
        notes = dict(curve_every=a.curve_every, reference_scale=choice[reference],
                     reference_validation=validation[reference],
                     within_0p1_point=[near[0], near[-1]],
                     validation_at_report={str(s): validation[s] for s in report})
        log(f'reference depth {reference} (scale {choice[reference]:g}), validation {validation[reference]:.4f};',
            f'within 0.1 point from {near[0]} to {near[-1]};', notes['validation_at_report'])
        if a.curve_only:
            (out / 'results.json').write_text(json.dumps(dict(
                trials={str(k): {str(d): e for d, e in v.items()} for k, v in trials.items()},
                choice={str(k): v for k, v in choice.items()},
                curves={str(k): {str(d): e for d, e in v.items()} for k, v in curves.items()},
                validation={f'fista{k}': v for k, v in validation.items()}, reference_depth=reference,
                reference_at_last_snapshot=bool(reference == report[-1]), z_input=a.z_input,
                z_delta=a.z_delta, pad=a.pad, **notes), indent=2))
            return
        snaps[:] = sorted(set(report) | {reference})
        comsol, preds = [], {}
        for scale in sorted({choice[s] for s in snaps}):
            depths = [s for s in snaps if choice[s] == scale]
            for config, sharp, blurred in comsol_inputs():
                target = torch.tensor(sharp / dataset.scale, dtype=torch.float32)
                y = predict(torch.tensor(blurred / dataset.scale, dtype=torch.float32), g, lam0 * scale,
                            a.full_batch, upto=depths[-1])
                for s in depths:
                    e = relative_l2(y[s], target)
                    preds[f'fista{s}_{config}'] = (y[s].numpy() * dataset.scale).astype(np.float32)
                    for c, comp in enumerate(('Bx', 'By', 'Bz')):
                        comsol.append(dict(method=f'fista{s}', lam=lam0 * scale, config=config, component=comp,
                                           relative_l2=float(e[c])))
                log(f'scale {scale:g} {config} done')
    else:
        trials = {}                               # scale -> {depth: score on 16 crops}

        def trial(scale):
            pred = predict(val[1][:16], g, lam0 * scale, 4)
            trials[scale] = {s: float(relative_l2(pred[s], val[0][:16]).mean()) for s in snaps}
            log(f'scale {scale:g} (lambda {lam0 * scale:g}):', trials[scale])

        for scale in a.scales:
            trial(scale)
        run_gate(trials)
        # choice per depth: the protocol's single extension at a grid edge, or the ladder
        choice = {}
        for s in snaps:
            best = min(trials, key=lambda sc: trials[sc][s])
            if a.ladder:
                while (extra := extension(best, trials)) is not None:
                    trial(extra)
                    best = min(trials, key=lambda sc: trials[sc][s])
            else:
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
            for config, sharp, blurred in comsol_inputs():
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
    results = dict(
        trials={str(k): v for k, v in trials.items()}, choice={str(k): v for k, v in choice.items()},
        validation={f'fista{k}': v for k, v in validation.items()}, comsol_mean_relative_l2=summary,
        z_input=a.z_input, z_delta=a.z_delta, pad=a.pad)
    results.update(notes)
    # reference depth: lowest 64-crop validation error among the sampled depths, chosen without COMSOL
    reference = min(validation, key=validation.get)
    results.update(reference_depth=reference, reference_at_last_snapshot=bool(reference == snaps[-1]))
    (out / 'results.json').write_text(json.dumps(results, indent=2))
    keep = set(a.keep_depths or snaps) | {reference}
    np.savez_compressed(out / 'predictions.npz',
                        **{k: v for k, v in preds.items() if int(k.split('_')[0][len('fista'):]) in keep})
    log('COMSOL means', summary, 'reference depth', reference)

    # full simulated images at the reference depth
    count = a.sim_layouts if reference <= a.sim_deep_above else (a.sim_layouts_deep or a.sim_layouts)
    if count:
        recipes = dataset.recipes('validation')[:count]
        rows = []
        for i, recipe in enumerate(recipes):
            target, blurred = full[0][3 * i:3 * i + 3], full[1][3 * i:3 * i + 3]
            y = predict(blurred, g, lam0 * choice[reference], a.full_batch, upto=reference)[reference]
            e = relative_l2(y, target)
            rows.append(dict(layout=recipe['id'], depth=reference, lam=lam0 * choice[reference],
                             **{c: float(e[k]) for k, c in enumerate(('Bx', 'By', 'Bz'))}))
            if (i + 1) % 5 == 0:
                log(f'simulated full images: {i + 1}/{len(recipes)}')
        with open(out / 'sim_full.csv', 'w', newline='') as stream:
            w = csv.DictWriter(stream, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        results.update(sim_full_layouts=len(rows),
                       sim_full_mean_relative_l2=float(np.mean([[r[c] for c in ('Bx', 'By', 'Bz')] for r in rows])))
        (out / 'results.json').write_text(json.dumps(results, indent=2))
        log('simulated full images', results['sim_full_mean_relative_l2'], 'at depth', reference)


if __name__ == '__main__':
    main()
