"""Select L1 lambda by Morozov discrepancy using a manually chosen flat patch.

python scripts/select_lambda.py --z 3 --comp Bz --select-roi --output runs/lambda_bz
python scripts/select_lambda.py --z 3 --comp Bz --roi 800 1000 800 1000 --output runs/lambda_bz
ROI order: x_start x_stop y_start y_stop, zero-based pixels, stops exclusive.
The kernel is fixed during lambda selection; sharp truth is only used to fit
that same-sample kernel and to evaluate reconstruction, never to select lambda.
"""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector

import fields as F
from deblur import adjoint_operator, forward_operator, learn_kernel_joint
from iterative import solve_l1, console_progress


def patch_statistics(b, roi):
    x0, x1, y0, y1 = map(int, roi)
    if not (0 <= x0 < x1 <= b.shape[1] and 0 <= y0 < y1 <= b.shape[0]):
        raise ValueError('ROI must lie inside the image; stops must exceed starts')
    patch = b[y0:y1, x0:x1]
    if patch.size < 2 or not np.isfinite(patch).all():
        raise ValueError('ROI needs at least two finite pixels')
    variance = float(np.var(patch, ddof=1))
    return dict(roi_pixels=[x0, x1, y0, y1], roi_order='x_start x_stop y_start y_stop; stops exclusive',
                patch_pixels=patch.size, image_pixels=b.size, mean_uT=float(patch.mean()),
                variance_uT2=variance, sigma_uT=float(np.sqrt(variance)),
                target_squared_residual_uT2=float(b.size*variance))


def patch_figure(b, roi, title):
    info = patch_statistics(b, roi)
    x0, x1, y0, y1 = roi
    patch = b[y0:y1, x0:x1]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    im = axes[0, 0].imshow(b, origin='lower', cmap='RdBu_r')
    axes[0, 0].add_patch(Rectangle((x0-.5, y0-.5), x1-x0, y1-y0, fill=False, ec='lime', lw=2))
    axes[0, 0].set(title='Selected region in observed field', xlabel='Column (pixels)', ylabel='Row (pixels)')
    fig.colorbar(im, ax=axes[0, 0], label='µT')
    im = axes[0, 1].imshow(patch, origin='lower', cmap='RdBu_r', extent=(x0-.5,x1-.5,y0-.5,y1-.5))
    axes[0, 1].set(title='Patch with its own color scale', xlabel='Column (pixels)', ylabel='Row (pixels)')
    fig.colorbar(im, ax=axes[0, 1], label='µT')
    axes[1, 0].plot(patch.mean(axis=0)-patch.mean(), label='Column means minus patch mean')
    axes[1, 0].plot(patch.mean(axis=1)-patch.mean(), label='Row means minus patch mean')
    axes[1, 0].set(title='Look for spatial trends (no detrending applied)', xlabel='Position within patch', ylabel='µT')
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].hist((patch-patch.mean()).ravel(), bins=40, density=True)
    if info['sigma_uT'] > 0:
        v = np.linspace(-4*info['sigma_uT'], 4*info['sigma_uT'], 200)
        axes[1, 1].plot(v, np.exp(-v*v/(2*info['variance_uT2']))/(info['sigma_uT']*np.sqrt(2*np.pi)), label='Gaussian with estimated σ')
        axes[1, 1].legend()
    axes[1, 1].set(title='Mean-subtracted patch distribution', xlabel='µT', ylabel='Density')
    fig.suptitle(f"{title}\nσ²={info['variance_uT2']:.6g} µT²; σ={info['sigma_uT']:.6g} µT; m={patch.size}")
    return fig


def choose_roi(b):
    """Manual selection, followed by a diagnostic preview and explicit acceptance."""
    try:
        plt.switch_backend('MacOSX' if sys.platform == 'darwin' else 'TkAgg')
    except ImportError as exc:
        raise RuntimeError('Interactive display unavailable. Use --roi X0 X1 Y0 Y1 instead.') from exc
    while True:
        selected = []
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(b, origin='lower', cmap='RdBu_r')
        fig.colorbar(im, ax=ax, label='µT')
        ax.set(title='Drag a rectangle over a flat region; press Enter to preview', xlabel='Column (pixels)', ylabel='Row (pixels)')
        def on_select(a, z):
            xs = sorted([a.xdata, z.xdata]); ys = sorted([a.ydata, z.ydata])
            roi = [max(0, int(np.ceil(xs[0]))), min(b.shape[1], int(np.floor(xs[1]))+1),
                   max(0, int(np.ceil(ys[0]))), min(b.shape[0], int(np.floor(ys[1]))+1)]
            try:
                patch_statistics(b, roi)
            except ValueError:
                return
            selected[:] = roi
        selector = RectangleSelector(ax, on_select, button=[1], interactive=True, useblit=False)
        def on_key(event):
            if event.key == 'enter' and selected:
                plt.close(fig)
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show(block=True)
        if not selected:
            raise ValueError('No patch selected; cancelled')
        preview = patch_figure(b, selected, 'Enter: accept this patch | R: select again')
        action = []
        def confirm(event):
            if event.key in ('enter', 'r'):
                action[:] = [event.key]
                plt.close(preview)
        preview.canvas.mpl_connect('key_press_event', confirm)
        plt.show(block=True)
        if action == ['enter']:
            plt.switch_backend('Agg')
            return selected
        if action != ['r']:
            raise ValueError('Patch confirmation cancelled')


def search_discrepancy(evaluate, target, lambda_max, zero_energy, *, start=1e-4, tolerance=.05, max_trials=35):
    """Bracket and bisect a nondecreasing converged residual curve in log lambda.

    evaluate(lambda) returns a dict with energy and converged. No truth enters.
    Returns an explicit status rather than accepting an unreachable target.
    """
    if not np.isfinite(target) or target <= 0:
        raise ValueError('Noise target must be positive and finite')
    if zero_energy < target*(1-tolerance):
        return dict(status='target_above_maximum_residual', selected=None)
    if lambda_max <= 0:
        return dict(status='zero_operator_correlation', selected=None)
    low, high, candidate = None, None, min(start, lambda_max)
    best = None
    for _ in range(max_trials):
        result = evaluate(candidate)
        if not result['converged']:
            return dict(status='inner_solver_not_converged', selected=None)
        mismatch = abs(result['energy']/target-1)
        if best is None or mismatch < best['mismatch']:
            best = dict(result)
            best.update(lambda_value=candidate, mismatch=mismatch)
        if mismatch <= tolerance:
            return dict(status='matched', selected=best)
        if result['energy'] < target:
            low = candidate
        else:
            high = candidate
        if low is not None and high is not None:
            candidate = np.sqrt(low*high)
        elif low is not None:
            if candidate == lambda_max:
                break
            candidate = min(candidate*10, lambda_max)
        else:
            candidate /= 10
        if candidate == 0 or (low is not None and high is not None and high/low-1 < 1e-10):
            break
    return dict(status='target_not_matched', selected=None, closest=best)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--z', type=float, default=3.)
    parser.add_argument('--comp', choices=F.SIGNED, default='Bz')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--select-roi', action='store_true')
    group.add_argument('--roi', nargs=4, type=int, metavar=('X0','X1','Y0','Y1'))
    group.add_argument('--roi-from', type=Path, help='reuse coordinates from a saved noise.json, recomputing variance')
    parser.add_argument('--estimate-only', action='store_true', help='save patch/noise diagnostics without a lambda search')
    parser.add_argument('--solver', choices=('ista','fista'), default='fista')
    parser.add_argument('--epsilon', type=float, default=1e-4)
    parser.add_argument('--discrepancy-tolerance', type=float, default=.05, help='relative tolerance on squared residual (default 5%%)')
    parser.add_argument('--pad', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.z <= .5 or args.z not in F.z_values():
        parser.error('z must be an available primary-cache height above .5 µm')
    if not 0 < args.epsilon < 1 or not 0 < args.discrepancy_tolerance < 1 or args.pad < 0:
        parser.error('epsilon and discrepancy-tolerance must be in (0,1); pad must be nonnegative')
    if args.output.exists():
        parser.error('output must be a new directory')
    b = F.load(args.z, args.comp)
    if args.roi_from:
        previous = json.loads(args.roi_from.read_text())
        if previous['z_um'] != args.z or previous['component'] != args.comp:
            parser.error('saved ROI belongs to a different height/component')
        roi = previous['roi_pixels']
    else:
        roi = choose_roi(b) if args.select_roi else args.roi
    info = patch_statistics(b, roi)
    info.update(z_um=args.z, component=args.comp, estimator='sample variance, ddof=1; mean subtraction only',
                assumption='approximately constant underlying field; representative, independent constant-variance noise',
                source_sha256=hashlib.sha256((F.CACHE/f'{F.z_tag(args.z)}_{args.comp}.npy').read_bytes()).hexdigest())
    args.output.mkdir(parents=True)
    write_json(args.output/'noise.json', info)
    fig = patch_figure(b, roi, f'{args.comp}, z={args.z:g} µm: manually selected noise region')
    fig.savefig(args.output/'noise_patch.png', dpi=130); plt.close(fig)
    print(json.dumps(info, indent=2), flush=True)
    if args.estimate_only:
        print(f'Patch estimate saved. Reuse with --roi-from {args.output/"noise.json"}', flush=True)
        return
    target = info['target_squared_residual_uT2']
    if target <= 0:
        raise ValueError('Selected patch has zero variance; choose a representative region with measurable fluctuations')
    sharps = [F.load(.5,c) for c in F.SIGNED]
    blurreds = [F.load(args.z,c) for c in F.SIGNED]
    h = learn_kernel_joint(sharps, blurreds, pad=args.pad)
    np.save(args.output/'kernel.npy', h)
    truth = F.load(.5, args.comp)
    rows = []
    histories = {}
    lambda_max = float(np.max(abs(adjoint_operator(b,h,args.pad))))
    best_gap = np.inf

    def evaluate(lam, epsilon=None, verification=False):
        nonlocal best_gap
        eps = args.epsilon if epsilon is None else epsilon
        print(f'\nLambda trial {len(rows)+1}: λ={lam:.8g}; epsilon={eps:g}', flush=True)
        x, history = solve_l1(b,h,lam,method=args.solver,pad=args.pad,epsilon=eps,
                             progress=console_progress(args.solver.upper()))
        energy = float(np.sum((b-forward_operator(x,h,args.pad))**2))
        last = history[-1]
        rmse = float(np.sqrt(np.mean((x-truth)**2)))
        row = dict(trial=len(rows)+1, lambda_value=lam, epsilon=eps, verification=verification,
                   energy=energy, ratio=energy/target, mismatch=abs(energy/target-1),
                   iterations=last['iteration'], converged=last['converged'], stop_reason=last['stop_reason'],
                   relative_pg=last['relative_pg'], solver_seconds=last['solver_seconds'],
                   rmse_uT=rmse, nrmse_range=rmse/float(np.ptp(truth)) if np.ptp(truth)>0 else None)
        rows.append(row); histories[str(row['trial'])]=history
        with (args.output/'lambda_trials.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(row)); writer.writeheader(); writer.writerows(rows)
        write_json(args.output/'histories.json', histories)
        if lam == 1e-4 and not verification:
            np.save(args.output/'baseline_lambda_1e-4.npy', x)
        if row['converged'] and row['mismatch'] < best_gap:
            best_gap = row['mismatch']
            np.save(args.output/'closest_reconstruction.npy', x)
        if verification:
            np.save(args.output/'verified_candidate.npy', x)
        print(f"Squared residual / noise target = {row['ratio']:.6g}; target 1 ± {args.discrepancy_tolerance:g}", flush=True)
        return row

    # Preserve the existing arbitrary lambda as an explicit evaluation baseline.
    result = search_discrepancy(evaluate, target, lambda_max, float(np.sum(b*b)), tolerance=args.discrepancy_tolerance)
    selected = result.get('selected')
    if selected:
        check = evaluate(selected['lambda_value'], args.epsilon/10, verification=True)
        stable = abs(check['energy']-selected['energy'])/target <= args.discrepancy_tolerance/5
        if check['converged'] and check['mismatch'] <= args.discrepancy_tolerance and stable:
            result.update(status='selected_and_verified', selected=check)
            np.save(args.output/'selected_reconstruction.npy', np.load(args.output/'verified_candidate.npy'))
        else:
            result.update(status='tighter_solver_verification_failed', selected=None,
                          candidate=selected, verification=check,
                          next_step='Repeat with a smaller --epsilon; no lambda accepted')
    sources = [Path(__file__), Path(__file__).with_name('iterative.py'), Path(__file__).with_name('deblur.py'), Path(F.__file__), F.CACHE/'grid.json']
    sources += [F.CACHE/f'{F.z_tag(z)}_{c}.npy' for z in (.5,args.z) for c in F.SIGNED]
    write_json(args.output/'metrics.json', dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
               noise=info, selection=result, lambda_max=lambda_max, trials=rows,
               evaluation='Same-sample joint kernel fit; truth used for evaluation, not discrepancy selection',
               sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
    render_results(args.output, b, truth, h, args.pad, rows, target, result, args.discrepancy_tolerance)
    print(f"\nStatus: {result['status']}", flush=True)
    if result.get('selected'):
        print(f"Selected lambda: {result['selected']['lambda_value']:.8g}", flush=True)
    print(f'Report: {args.output/"index.html"}', flush=True)


def render_results(output, b, truth, h, pad, rows, target, result, tolerance=.05):
    gallery = [('noise_patch.png','Manually selected patch and noise estimate')]
    if rows:
        fig, axes = plt.subplots(1,3,figsize=(16,4.5),constrained_layout=True)
        trials = sorted([r for r in rows if not r['verification']], key=lambda r:r['lambda_value'])
        for ax, metric, title in zip(axes, ['ratio','rmse_uT','iterations'],
                                     ['Discrepancy: target is 1','Truth RMSE: evaluation only','Actual solver updates']):
            ax.semilogx([r['lambda_value'] for r in trials], [r[metric] for r in trials], 'o-')
            ax.set(xlabel='λ',title=title); ax.grid(alpha=.3)
        axes[0].axhline(1,color='black',ls='--')
        axes[0].axhspan(1-tolerance,1+tolerance,color='green',alpha=.15)
        axes[0].set_ylabel('Squared residual / estimated noise energy')
        axes[1].set_ylabel('µT'); axes[2].set_ylabel('Updates')
        if result.get('selected'):
            for ax in axes:
                ax.axvline(result['selected']['lambda_value'], color='green', ls=':')
        fig.suptitle(result['status']); fig.savefig(output/'lambda_search.png',dpi=130); plt.close(fig)
        gallery.append(('lambda_search.png','Lambda search; truth error does not control selection'))
        fig, ax = plt.subplots(figsize=(10,5),constrained_layout=True)
        histories = json.loads((output/'histories.json').read_text())
        for row in rows:
            hist = histories[str(row['trial'])]
            ax.semilogy([r['iteration'] for r in hist],np.maximum([r['relative_pg'] for r in hist],1e-16),
                        label=f"λ={row['lambda_value']:.3g}, ε={row['epsilon']:g}: {row['iterations']} updates; {row['stop_reason']}")
        ax.set(xlabel='Completed updates',ylabel='Relative PG',title='Inner solver convergence')
        ax.legend(fontsize=7); ax.grid(alpha=.3)
        fig.savefig(output/'convergence.png',dpi=130); plt.close(fig)
        gallery.append(('convergence.png','Convergence for every lambda trial'))
    selected_path = output/'selected_reconstruction.npy'
    path = selected_path if selected_path.exists() else output/'closest_reconstruction.npy'
    if path.exists():
        x = np.load(path)
        label = 'Selected reconstruction' if selected_path.exists() else 'Closest trial — no λ accepted'
        images=[(b,'Observed'),(truth,'Sharp reference'),(x,label),(truth-x,'Reconstruction residual'),
                (forward_operator(x,h,pad),'Reblurred estimate'),(b-forward_operator(x,h,pad),'Reblurred residual')]
        baseline = output/'baseline_lambda_1e-4.npy'
        if baseline.exists():
            old = np.load(baseline); images += [(old,'Baseline λ=1e-4'),(truth-old,'Baseline reconstruction residual')]
        fig,axes=plt.subplots(2,4,figsize=(19,9),constrained_layout=True)
        vmax=max(float(np.max(abs(truth))),1e-15)
        for ax,(img,title) in zip(axes.flat,images):
            limit=max(float(np.max(abs(img))),1e-15) if 'residual' in title.lower() else vmax
            im=ax.imshow(img,origin='lower',cmap='RdBu_r',vmin=-limit,vmax=limit)
            ax.set(title=title,xlabel='Column (pixels)',ylabel='Row (pixels)'); fig.colorbar(im,ax=ax,label='µT',shrink=.7)
        for ax in list(axes.flat)[len(images):]: ax.set_visible(False)
        if result.get('selected'):
            chosen = result['selected']
            fig.suptitle(f"λ={chosen['lambda_value']:.6g}; {chosen['iterations']} updates; "
                         f"{chosen['stop_reason']}; relative PG={chosen['relative_pg']:.3g}")
        fig.savefig(output/'reconstruction.png',dpi=130); plt.close(fig)
        gallery.append(('reconstruction.png','Reconstruction and residuals'))
    doc='<!doctype html><meta charset="utf-8"><title>Morozov lambda selection</title><style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:20px}img{width:100%}</style><h1>Patch-based Morozov lambda selection</h1>'
    doc+=f"<p>Status: {html.escape(result['status'])}. Target squared residual: {target:.6g} µT².</p>"
    if result.get('selected'): doc+=f"<p>Selected λ: {result['selected']['lambda_value']:.8g}; discrepancy ratio: {result['selected']['ratio']:.5g}.</p>"
    doc+='<p>Assumes a flat patch and representative noise variance. Same-sample kernel fit; no holdout claim.</p><p><a href="noise.json">Noise estimate</a> · <a href="metrics.json">Selection details</a>'
    if rows: doc+=' · <a href="lambda_trials.csv">All trials</a>'
    doc+='</p>'
    for name,caption in gallery: doc+=f'<h2>{caption}</h2><img src="{name}">'
    (output/'index.html').write_text(doc)


if __name__ == '__main__':
    main()
