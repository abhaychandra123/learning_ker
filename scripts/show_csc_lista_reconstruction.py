"""Show a trained CSC-LISTA reconstruction of one simulated validation field.

Draws, per component: the 3 um input, Wiener, optional FISTA runs on the same
dictionary, CSC-LISTA, the true 0.5 um field and the error maps of the last FISTA run
and CSC-LISTA. Validation fields were never used for
weight updates, only for choosing the checkpoint.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import csc_lista as csc
from simulate_fields import COMPONENTS, SimulatedDataset, synthesise
from train_csc_lista import DX, CodedSolver, predict, relative_l2


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--run', default='runs/csc_lista_strict/layers_64')
    p.add_argument('--sample', default='validation_00003')
    p.add_argument('--output', required=True)
    p.add_argument('--window', type=float, nargs=4, metavar=('X0', 'X1', 'Y0', 'Y1'),
                   help='Region to draw, in um; scores always use the whole image')
    p.add_argument('--fista', type=float, nargs=2, action='append', default=[],
                   metavar=('ITERATIONS', 'LAMBDA'),
                   help='Add FISTA on the same dictionary; repeat for several depths')
    p.add_argument('--components', nargs='+', default=list(COMPONENTS), choices=COMPONENTS)
    p.add_argument('--name', help='Figure file name; defaults to the sample id')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    run = Path(args.run)
    config = json.loads((run / 'args.json').read_text())
    results = json.loads((run / 'results.json').read_text())
    device = torch.device(args.device)
    dataset = SimulatedDataset(config['dataset'])
    recipe = next(r for r in dataset.recipes('validation') if r['id'] == args.sample)
    item = synthesise(recipe, dataset.settings)
    scale = dataset.scale
    target = torch.tensor(item['target'] / scale, dtype=torch.float32)
    observed = torch.tensor(item['input'] / scale, dtype=torch.float32)

    saved = torch.load(config['dictionary'], weights_only=True)
    pad = config['pad']
    canvas = config['crop'] + 2 * pad
    model = csc.CSCLISTA(saved['g'].to(device), DX, config['z_delta'], canvas, pad,
                         saved['lam'], config['layers'], config['filter_width'],
                         chunk=config['chunk']).to(device)
    model.load_state_dict(torch.load(run / 'best.pt', map_location=device,
                                     weights_only=True)['state'])
    model.eval()
    wiener = csc.Dataless(DX, config['z_delta'], results['wiener'][1])

    outputs = {'Wiener': predict(wiener, observed, pad, device)}
    for steps, lam in args.fista:
        solver = CodedSolver(saved['g'].to(device), config['z_delta'], lam, int(steps), pad,
                             True, config['chunk'])
        outputs[f'FISTA-{int(steps)} (lambda {lam:g})'] = predict(solver, observed, pad, device)
    cscname = f"CSC-LISTA ({config['layers']} layers)"
    outputs[cscname] = predict(model, observed, pad, device)
    errors = {name: relative_l2(y, target).numpy() for name, y in outputs.items()}
    shown_errors = [name for name in outputs if name.startswith('FISTA')][-1:] + [cscname]

    extent = [0, target.shape[-1] * DX, 0, target.shape[-2] * DX]
    if args.window:
        xlim, ylim = args.window[:2], args.window[2:]
    else:
        rows, cols = np.nonzero((target.abs() > .05 * target.abs().max()).any(0).numpy())
        margin = 5.0
        xlim = (cols.min() * DX - margin, cols.max() * DX + margin)
        ylim = (rows.min() * DX - margin, rows.max() * DX + margin)
    picked = [COMPONENTS.index(c) for c in args.components]
    count = 2 + len(outputs) + len(shown_errors)
    if len(picked) == 1:
        grid = (2, -(-count // 2))
    else:
        grid = (len(picked), count)
    fig, axes = plt.subplots(*grid, figsize=(4.4 * grid[1], 4.4 * grid[0] + .6),
                             constrained_layout=True, squeeze=False)
    rows_of_axes = [axes.ravel()] if len(picked) == 1 else list(axes)
    for row, i in enumerate(picked):
        component = COMPONENTS[i]
        t = target[i].numpy() * scale
        limit = np.abs(t).max()
        panels = [('Input, 3 um', observed[i].numpy() * scale)]
        for name, y in outputs.items():
            panels.append((f'{name}\nrelative L2 {errors[name][i]:.1%}', y[i].numpy() * scale))
        panels.append(('Target, 0.5 um', t))
        for name in shown_errors:
            diff = (outputs[name][i].numpy() - target[i].numpy()) * scale
            panels.append((f"{name.split(' (')[0]} minus target\n(same colour scale)", diff))
        for ax in rows_of_axes[row][len(panels):]:
            ax.axis('off')
        for ax, (title, image) in zip(rows_of_axes[row], panels):
            shown = ax.imshow(image, cmap='RdBu_r', vmin=-limit, vmax=limit, extent=extent,
                              origin='lower')
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_title(f'{component}: {title}', fontsize=11)
            ax.set_xlabel('um')
        fig.colorbar(shown, ax=rows_of_axes[row], shrink=.85, label='uT')
    means = ', '.join(f"{name.split(' (')[0]} {errors[name][picked].mean():.1%}"
                      for name in outputs)
    fig.suptitle(f"{args.sample} ({recipe.get('family', '')}), simulated v3, 3 -> 0.5 um\n"
                 f"relative L2 ({'/'.join(args.components)}): {means}", fontsize=13)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f'{args.name or args.sample}.png', dpi=110)
    np.savez_compressed(output / f'{args.name or args.sample}.npz',
                        target=target.numpy() * scale, observed=observed.numpy() * scale,
                        **{name.split(' (')[0].lower().replace('-', '_'): y.numpy() * scale
                           for name, y in outputs.items()})
    print({k: [round(float(e), 4) for e in v] for k, v in errors.items()})


if __name__ == '__main__':
    main()
