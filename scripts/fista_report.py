"""Visual report for saved FISTA runs; never reruns the inverse solver."""
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np

import fields as F
import propagator as P
from deblur import forward_operator, learn_kernel_joint


def scores(pred, truth):
    rmse = float(np.sqrt(np.mean((pred-truth)**2)))
    span = float(np.ptp(truth))
    return dict(rmse_uT=rmse, nrmse_range=rmse/span if span else None)


def predict_holdout(sharps, plane1, pad):
    """Fit only .5/1 planes; repeat the padded .5-um operator three times.

    Crop/repad at each step, so this explicitly includes finite-window effects.
    The held-out plane is deliberately not an argument.
    """
    h = learn_kernel_joint(sharps, plane1, pad=pad)
    predictions = []
    for sharp in sharps:
        pred = sharp.copy()
        for _ in range(3):
            pred = forward_operator(pred, h, pad)
        predictions.append(pred)
    return h, predictions


def render_report(output):
    output = Path(output)
    meta = json.loads((output/'metrics.json').read_text())
    cfg = meta['config']
    z, pad = cfg['z'], cfg['pad']
    grid = F.grid()
    # Historical reconstructions must not silently be paired with changed inputs.
    for name, digest in meta.get('sha256', {}).items():
        path = Path(name)
        if path.suffix == '.npy' or path.name == 'grid.json':
            local = F.CACHE/path.name
            if hashlib.sha256(local.read_bytes()).hexdigest() != digest:
                raise ValueError(f'Cached data changed since this run: {local}')
    gallery, diagnostics = [], {}

    def save(fig, filename, caption):
        fig.savefig(output/filename, dpi=120)
        plt.close(fig)
        gallery.append((filename, caption))
        print(f'Saved {output/filename}', flush=True)

    def maps(images, titles, filename, caption, reference):
        fig, axes = plt.subplots(1, len(images), figsize=(5*len(images), 4.8),
                                 constrained_layout=True, squeeze=False)
        vmax = max(float(np.max(np.abs(reference))), 1e-15)
        for ax, img, title in zip(axes[0], images, titles):
            limit = max(float(np.max(np.abs(img))), 1e-15) if 'residual' in title.lower() else vmax
            im = ax.imshow(img, origin='lower', extent=grid.extent, cmap='RdBu_r',
                           vmin=-limit, vmax=limit)
            ax.set(title=title, xlabel='x (µm)', ylabel='y (µm)')
            fig.colorbar(im, ax=ax, label='µT', shrink=.8)
        fig.suptitle(caption)
        save(fig, filename, caption)

    def kernel_plot(h, dz, filename, caption):
        impulse = np.fft.fftshift(np.fft.ifft2(h).real)
        peak = max(float(np.max(abs(impulse))), 1e-15)
        ny, nx = h.shape
        extent = ((-(nx//2)-.5)*grid.dx, (nx-nx//2-.5)*grid.dx,
                  (-(ny//2)-.5)*grid.dy, (ny-ny//2-.5)*grid.dy)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
        for ax, zoom in zip(axes[:2], (False, True)):
            im = ax.imshow(impulse, origin='lower', extent=extent, cmap='RdBu_r',
                           norm=SymLogNorm(peak*1e-4, vmin=-peak, vmax=peak))
            if zoom:
                radius = min(5*dz, min(nx*grid.dx, ny*grid.dy)/2)
                ax.set(xlim=(-radius, radius), ylim=(-radius, radius))
            ax.set(title='Impulse response — '+('center' if zoom else 'full grid'),
                   xlabel='x offset (µm)', ylabel='y offset (µm)')
            fig.colorbar(im, ax=ax, shrink=.8)
        k = P.k_grid(h.shape, grid.dx, grid.dy)
        for transfer, label in [(abs(h), 'Learned |H|'),
                                (P.analytic_transfer(k, dz), 'Analytic |H|')]:
            kr, mean = P.radial_average(transfer, k)
            axes[2].semilogy(kr, np.maximum(mean, 1e-16), label=label)
        axes[2].set(xlabel='Spatial frequency (cycles/µm)', ylabel='Radial mean magnitude',
                    title='Transfer spectrum')
        axes[2].legend(); axes[2].grid(alpha=.3)
        fig.suptitle(caption)
        save(fig, filename, caption)

    h = np.load(output/'kernel.npy')
    kernel_plot(h, z-.5, 'learned_kernel.png', f'Joint kernel fitted on 0.5 → {z:g} µm; all three components')
    for c, history in meta['history'].items():
        sharp, blurred = F.load(.5, c), F.load(z, c)
        x = np.load(output/f'reconstruction_{c}.npy')
        forward = forward_operator(sharp, h, pad)
        reblurred = forward_operator(x, h, pad)
        last = history[-1]
        status = f"{last['iteration']} updates; {last['stop_reason']}"
        err = scores(x, sharp)
        diagnostics[c] = dict(reconstruction=err, forward=scores(forward, blurred),
                              reblurred=scores(reblurred, blurred))
        error_text = f"NRMSE {err['nrmse_range']*100:.3f}%" if err['nrmse_range'] is not None else f"RMSE {err['rmse_uT']:.3g} µT"
        maps([sharp, blurred], ['Sharp reference (0.5 µm)', f'Observed ({z:g} µm)'],
             f'inputs_{c}.png', f'{c}: input pair', sharp)
        maps([sharp, x, sharp-x], ['Sharp reference', f'FISTA: {error_text}\n{status}', 'Residual: truth − FISTA'],
             f'reconstruction_{c}.png', f'{c}: same-sample reconstruction', sharp)
        maps([blurred, forward, reblurred, blurred-reblurred],
             ['Observed', 'A(sharp reference)', 'A(FISTA estimate)', 'Residual: observed − reblurred'],
             f'forward_{c}.png', f'{c}: forward consistency | {status}', blurred)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
        for arr, label in [(sharp, 'Sharp'), (blurred, 'Observed'), (x, 'FISTA')]:
            axes[0].plot(grid.x, arr[arr.shape[0]//2], label=label)
            kr, spectrum = P.radial_spectrum(arr, grid.dx)
            axes[1].semilogy(kr, np.maximum(spectrum, 1e-16), label=label)
        axes[0].set(title='Central row profile', xlabel='x (µm)', ylabel='µT')
        axes[1].set(title='Windowed radial amplitude spectrum', xlabel='cycles/µm', ylabel='µT')
        for ax in axes:
            ax.legend(); ax.grid(alpha=.3)
        fig.suptitle(f'{c}: {status}')
        save(fig, f'profiles_spectra_{c}.png', f'{c}: profiles and spectra')

    holdout = dict(status='unavailable', reason='Requires primary-cache planes 0.5, 1 and 2 µm')
    if all(zv in F.z_values() for zv in (.5, 1., 2.)):
        print('Kernel forward holdout: fitting .5 → 1 only; predicting z=2.', flush=True)
        sharps = [F.load(.5, c) for c in F.SIGNED]
        plane1 = [F.load(1., c) for c in F.SIGNED]
        hh, predictions = predict_holdout(sharps, plane1, pad)
        np.save(output/'holdout_step_kernel.npy', hh)
        kernel_plot(hh, .5, 'holdout_kernel.png', 'Separate holdout kernel: fit 0.5 → 1 µm only')
        analytic_h = P.analytic_transfer(P.k_grid(hh.shape, grid.dx, grid.dy), 1.5)
        holdout = dict(status='complete', kind='forward height holdout; not inverse reconstruction',
                       fit_heights_um=[.5, 1.], test_height_um=2., repetitions=3,
                       boundary='zero pad/crop at each learned step; one padded analytic step',
                       metrics={})
        for c, sharp, pred in zip(F.SIGNED, sharps, predictions):
            actual = F.load(2., c)  # Used only after predictions have been constructed.
            analytic = forward_operator(sharp, analytic_h, pad)
            holdout['metrics'][c] = dict(learned=scores(pred, actual), analytic=scores(analytic, actual))
            def error_label(method):
                value = holdout['metrics'][c][method]
                return (f"NRMSE {100*value['nrmse_range']:.3f}%" if value['nrmse_range'] is not None
                        else f"RMSE {value['rmse_uT']:.3g} µT")
            np.save(output/f'holdout_prediction_{c}.npy', pred)
            np.save(output/f'holdout_analytic_{c}.npy', analytic)
            maps([actual, pred, analytic, actual-pred],
                 ['Held-out observed z=2', 'Learned prediction (3 steps)\n'+error_label('learned'),
                  'Analytic prediction\n'+error_label('analytic'), 'Residual: observed − learned'],
                 f'holdout_{c}.png', f'{c}: forward height holdout; z=2 excluded from this kernel fit', actual)
    else:
        print(holdout['reason'], flush=True)
    (output/'report_metrics.json').write_text(json.dumps(dict(reconstruction=diagnostics, holdout=holdout), indent=2, allow_nan=False))
    sources = [Path(__file__), Path(__file__).with_name('deblur.py'), Path(P.__file__), Path(F.__file__), F.CACHE/'grid.json']
    sources += [F.CACHE/f'{F.z_tag(zv)}_{c}.npy' for zv in set([.5,z]+([1.,2.] if holdout['status']=='complete' else [])) for c in F.SIGNED]
    (output/'report_provenance.json').write_text(json.dumps({'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}, indent=2))
    if (output/'convergence.png').exists():
        gallery.append(('convergence.png', 'FISTA convergence and actual update counts'))
    doc = '<!doctype html><meta charset="utf-8"><title>FISTA report</title><style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:20px}img{width:100%}section{margin:40px 0}a{color:#1764ad}</style><h1>FISTA visual report</h1>'
    doc += f'<p>Reconstruction: {z:g} → 0.5 µm, same-sample kernel fit. Separate forward holdout: fit 0.5 → 1, predict z=2; no FISTA inverse holdout is claimed.</p>'
    doc += '<p><a href="metrics.json">Solver metrics</a> · <a href="summary.csv">Step counts and summary</a> · <a href="report_metrics.json">Forward and holdout metrics</a></p>'
    if holdout['status'] != 'complete':
        doc += '<p>'+html.escape(holdout['reason'])+'</p>'
    for filename, caption in gallery:
        doc += f'<section><h2>{html.escape(caption)}</h2><a href="{filename}"><img src="{filename}" loading="lazy"></a></section>'
    (output/'index.html').write_text(doc)
    print(f'Report: {output/"index.html"}', flush=True)
