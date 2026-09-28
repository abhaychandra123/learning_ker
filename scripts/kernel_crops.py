"""Direct equal-size crop FFT kernel experiment for the IC_4um exports.

python scripts/kernel_crops.py --output runs/ic4_crop_kernel
All-crop fits, separate spatial holdouts, and a z=5 height holdout are labeled
separately. FFTs are circular on each crop; no windowing/padding/rescaling.
"""
import argparse
import csv
import hashlib
import html
import json
import platform
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from propagator import k_grid, analytic_transfer

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("Bx", "By", "Bz")
HEIGHTS = (.5, 3., 4., 5., 6., 7., 10.)
plt.rcParams.update({"font.size": 9, "figure.dpi": 100, "savefig.dpi": 120})


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_plane(path):
    """Validate the complete coordinate raster; retain physical float64 µT."""
    with Path(path).open() as stream:
        header = [next(stream).strip() for _ in range(9)]
    if not all(line.startswith("%") for line in header):
        raise ValueError(f"expected nine COMSOL header lines: {path}")
    if not any("Length unit" in line and "µm" in line for line in header):
        raise ValueError("expected coordinates in µm")
    if not all(f"mf.{c} (T)" in header[-1] for c in COMPONENTS):
        raise ValueError("expected signed components in tesla")
    a = pd.read_csv(path, skiprows=9, header=None).to_numpy(dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 6 or not np.isfinite(a).all():
        raise ValueError(f"invalid numeric data: {path}")
    changes = np.flatnonzero(np.abs(a[:, 1] - a[0, 1]) > 1e-7)
    if not len(changes):
        raise ValueError("expected a two-dimensional raster")
    nx = int(changes[0])
    ny, remainder = divmod(len(a), nx)
    if remainder or min(nx, ny) < 2:
        raise ValueError("incomplete raster")
    xs, ys = a[:nx, 0], a[::nx, 1]
    dx, dy = float(xs[1]-xs[0]), float(ys[1]-ys[0])
    if dx <= 0 or dy <= 0:
        raise ValueError("expected increasing coordinates")
    expected_x = xs[0] + np.arange(nx)*dx
    expected_y = ys[0] + np.arange(ny)*dy
    if (not np.allclose(a[:, 0].reshape(ny, nx), expected_x[None, :], rtol=0, atol=1e-6)
            or not np.allclose(a[:, 1].reshape(ny, nx), expected_y[:, None], rtol=0, atol=1e-6)):
        raise ValueError("coordinates are not a complete uniform x-fastest grid")
    grid = dict(nx=nx, ny=ny, x0=float(xs[0]), y0=float(ys[0]), dx=dx, dy=dy)
    fields = a[:, 2:5].T.reshape(3, ny, nx).copy()*1e6
    return fields, grid


def make_crops(fields, size):
    """[component,y,x] -> [row-major crop,component,y,x]."""
    if fields.ndim != 3 or size < 1:
        raise ValueError("expected 3D fields and a positive crop size")
    c, ny, nx = fields.shape
    if ny % size or nx % size:
        raise ValueError(f"crop size {size} must divide grid {ny}x{nx}")
    return fields.reshape(c, ny//size, size, nx//size, size).transpose(1, 3, 0, 2, 4).reshape(-1, c, size, size).copy()


def assemble(crops, shape):
    ny, nx = shape
    n, c, sy, sx = crops.shape
    if sy != sx or ny % sy or nx % sx or n != (ny//sy)*(nx//sx):
        raise ValueError("crop batch does not tile requested shape")
    return crops.reshape(ny//sy, nx//sx, c, sy, sx).transpose(2, 0, 3, 1, 4).reshape(c, ny, nx)


def fit_kernel(sharp, blurred, ids=None, den_rtol=1e-12):
    """Joint unregularized LS; mask denominator <= rtol * its maximum."""
    if sharp.shape != blurred.shape or sharp.ndim != 4 or not np.isfinite(sharp).all() or not np.isfinite(blurred).all():
        raise ValueError("expected equal finite crop batches")
    if not np.isfinite(den_rtol) or not 0 <= den_rtol < 1:
        raise ValueError("den_rtol must be finite and in [0,1)")
    ids = list(range(len(sharp))) if ids is None else list(ids)
    if not ids or len(set(ids)) != len(ids) or any(i < 0 or i >= len(sharp) for i in ids):
        raise ValueError("expected nonempty unique valid crop IDs")
    num = np.zeros(sharp.shape[-2:], dtype=np.complex128)
    den = np.zeros(sharp.shape[-2:], dtype=np.float64)
    for i in ids:
        s = np.fft.fft2(sharp[i].astype(np.float64))
        b = np.fft.fft2(blurred[i].astype(np.float64))
        num += np.sum(s.conj()*b, axis=0)
        den += np.sum(abs(s)**2, axis=0)
    cutoff = float(den_rtol*den.max())
    supported = den > cutoff
    h = np.divide(num, den, out=np.zeros_like(num), where=supported)
    return h, {"denominator_cutoff": cutoff, "supported_fraction": float(supported.mean()),
               "max_gain": float(abs(h).max()), "dc_real": float(h[0, 0].real),
               "dc_imag": float(h[0, 0].imag), "crop_ids": ids}


def apply_kernel(crops, transfer):
    out = np.empty_like(crops, dtype=np.float64)
    for i, crop in enumerate(crops):
        out[i] = np.fft.ifft2(np.fft.fft2(crop.astype(np.float64))*transfer).real
    return out


def split_ids(nrows, ncols):
    """One deterministic held-out crop per row; shared across all planes."""
    if min(nrows, ncols) < 2:
        raise ValueError("need at least 2x2 crops for a spatial holdout")
    test = [r*ncols + r % ncols for r in range(nrows)]
    train = sorted(set(range(nrows*ncols))-set(test))
    return train, test


def height_holdout(reference, plane3, plane4, den_rtol):
    """No z=5 argument: fit only .5->3 and 3->4, then predict .5->5."""
    anchor, anchor_info = fit_kernel(reference, plane3, den_rtol=den_rtol)
    step, step_info = fit_kernel(plane3, plane4, den_rtol=den_rtol)
    composed = anchor*step**2
    return apply_kernel(reference, composed), anchor, step, composed, {
        "fit_pairs_um": [[.5, 3.], [3., 4.]], "path_um": [.5, 3., 4., 5.],
        "heldout_height_um": 5., "anchor": anchor_info, "step": step_info,
        "composed_max_gain": float(abs(composed).max())}


def scores(prediction, actual):
    rmse = float(np.sqrt(np.mean((prediction-actual)**2)))
    span = float(np.ptp(actual))
    rms = float(np.sqrt(np.mean(actual**2)))
    return dict(rmse_uT=rmse, nrmse_range=rmse/span if span else None,
                relative_l2=rmse/rms if rms else None)


def score_rows(prediction, actual, z, evaluation, method, ids):
    rows = []
    for c, component in enumerate(COMPONENTS):
        for i in [None, *ids]:
            sel = ids if i is None else i
            rows.append(dict(z=z, evaluation=evaluation, method=method, component=component,
                             crop_id="all" if i is None else i,
                             **scores(prediction[sel, c], actual[sel, c])))
    return rows


def save_figure(fig, output, name, gallery, caption):
    fig.savefig(output/"figures"/name)
    plt.close(fig)
    gallery.append((name, caption))


def extent(grid):
    return [grid["x0"]-.5*grid["dx"], grid["x0"]+(grid["nx"]-.5)*grid["dx"],
            grid["y0"]-.5*grid["dy"], grid["y0"]+(grid["ny"]-.5)*grid["dy"]]


def plot_grid(crops, grid, size, test, output, gallery):
    planes = assemble(crops, (grid["ny"], grid["nx"]))
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    ex = extent(grid)
    cols = grid["nx"]//size
    for c, ax in enumerate(axes):
        v = max(float(np.percentile(abs(planes[c]), 99.5)), 1e-12)
        im = ax.imshow(planes[c], origin="lower", extent=ex, cmap="RdBu_r", vmin=-v, vmax=v)
        for i in range(len(crops)):
            row, col = divmod(i, cols)
            x, y = ex[0]+col*size*grid["dx"], ex[2]+row*size*grid["dy"]
            color = "#00ff9d" if i in test else "#202020"
            ax.add_patch(Rectangle((x, y), size*grid["dx"], size*grid["dy"], fill=False, edgecolor=color, lw=1.5))
            ax.text(x+size*grid["dx"]*.5, y+size*grid["dy"]*.5, str(i), ha="center", color="black",
                    bbox=dict(facecolor="white", alpha=.7, edgecolor="none", pad=1))
        ax.set(title=COMPONENTS[c], xlabel="x (µm)", ylabel="y (µm)")
        fig.colorbar(im, ax=ax, label="µT", shrink=.75)
    fig.suptitle(f"IC_4um reference z=0.5 µm | {len(crops)} aligned {size}×{size} crops\n"
                 "Green outlines: spatial holdout crops (excluded only from the separate spatial-holdout fits)")
    save_figure(fig, output, "01_crop_grid.png", gallery, "Reference plane and numbered crop locations")


def plot_atlas(sharp, blurred, z, c, output, gallery):
    n = len(sharp)
    columns = 5
    fig, axes = plt.subplots(int(np.ceil(n/columns)), columns, figsize=(15, 1.8*int(np.ceil(n/columns))+.6),
                             constrained_layout=True, squeeze=False)
    v = max(float(np.percentile(abs(np.concatenate([sharp[:, c], blurred[:, c]])), 99.5)), 1e-12)
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        ax.imshow(np.concatenate([sharp[i, c], blurred[i, c]], axis=1), origin="lower",
                  cmap="RdBu_r", vmin=-v, vmax=v)
        ax.axvline(sharp.shape[-1]-.5, color="black", lw=.7)
        ax.set_title(f"Crop {i}: reference | z={z:g}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{COMPONENTS[c]} aligned initial crop pairs: 0.5 → {z:g} µm | shared scale ±{v:.2f} µT")
    save_figure(fig, output, f"pairs_z{z:g}_{COMPONENTS[c]}.png", gallery,
                f"All {n} initial {COMPONENTS[c]} crop pairs: 0.5 → {z:g} µm")


def plot_prediction(sharp, actual, pred, analytic, grid, z, output, gallery, stem, title, ids=None):
    shape = (grid["ny"], grid["nx"])
    arrays = [assemble(x, shape) for x in (sharp, actual, pred, analytic)]
    if ids is not None:
        mask = np.zeros((len(sharp), 1, *sharp.shape[-2:]))
        mask[ids] = 1
        valid = assemble(mask, shape)[0].astype(bool)
        arrays = [np.where(valid[None], a, np.nan) for a in arrays]
    fig, axes = plt.subplots(3, 5, figsize=(20, 11), constrained_layout=True)
    chosen = list(range(len(sharp))) if ids is None else ids
    for c, component in enumerate(COMPONENTS):
        v = max(float(np.percentile(abs(actual[chosen, c]), 99.5)), 1e-12)
        residual = arrays[2][c]-arrays[1][c]
        rv = max(float(np.nanpercentile(abs(residual), 99.5)), 1e-12)
        learned_score = scores(pred[chosen, c], actual[chosen, c])["nrmse_range"]
        analytic_score = scores(analytic[chosen, c], actual[chosen, c])["nrmse_range"]
        fmt = lambda q: "undefined" if q is None else f"{100*q:.2f}%"
        titles = ["Reference z=0.5", f"Actual z={z:g}", f"Learned ({fmt(learned_score)})",
                  f"Analytic ({fmt(analytic_score)})", "Learned − actual"]
        for j, data in enumerate([arrays[0][c], arrays[1][c], arrays[2][c], arrays[3][c], residual]):
            scale = rv if j == 4 else v
            im = axes[c, j].imshow(data, origin="lower", extent=extent(grid), cmap="RdBu_r", vmin=-scale, vmax=scale)
            axes[c, j].set_title(f"{component}: {titles[j]}")
            axes[c, j].set_xlabel("x (µm)")
            if j == 0:
                axes[c, j].set_ylabel("y (µm)")
            if j in (3, 4):
                fig.colorbar(im, ax=axes[c, j], label="µT", shrink=.8)
    fig.suptitle(title+"\nPercentages = pooled range NRMSE; images clipped at 99.5th percentile; crop-wise circular FFT")
    save_figure(fig, output, stem, gallery, title)


def plot_kernels(kernels, grid, output, gallery, crop_count):
    fig, axes = plt.subplots(3, len(kernels), figsize=(23, 11), constrained_layout=True)
    for j, (z, h) in enumerate(kernels.items()):
        n = h.shape[0]
        center = np.fft.fftshift(np.fft.ifft2(h).real)
        lim = max(float(abs(center).max()), 1e-12)
        lagx = (np.arange(n)-n//2)*grid["dx"]
        lagy = (np.arange(n)-n//2)*grid["dy"]
        im = axes[0, j].imshow(center, origin="lower", extent=[lagx[0]-.5*grid["dx"], lagx[-1]+.5*grid["dx"],
                                  lagy[0]-.5*grid["dy"], lagy[-1]+.5*grid["dy"]],
                                  cmap="RdBu_r", norm=SymLogNorm(linthresh=lim*1e-3, vmin=-lim, vmax=lim))
        axes[0, j].set(title=f"0.5 → {z:g} µm: impulse", xlabel="x lag (µm)", ylabel="y lag (µm)")
        fig.colorbar(im, ax=axes[0, j], label="discrete weight", shrink=.7)
        k = k_grid(h.shape, grid["dx"], grid["dy"])
        a = analytic_transfer(k, z-.5)
        # Bin only included frequencies, without folding overflow into the last bin.
        edges = np.linspace(0, 1., 50)
        bins = np.digitize(k.ravel(), edges)-1
        means = [np.mean(abs(h).ravel()[bins == i]) if np.any(bins == i) else np.nan for i in range(len(edges)-1)]
        centers = (edges[:-1]+edges[1:])/2
        axes[1, j].semilogy(centers, np.maximum(means, 1e-12), label="Learned radial mean")
        axes[1, j].semilogy(centers, analytic_transfer(centers, z-.5), "--", label="Analytic")
        axes[1, j].set(xlabel="cycles/µm", ylabel="|H|", ylim=(1e-8, max(2, float(abs(h).max())*1.1)))
        axes[1, j].legend(fontsize=7)
        axes[1, j].grid(alpha=.2)
        analytic_impulse = np.fft.fftshift(np.fft.ifft2(a).real)
        axes[2, j].plot(lagx, center[n//2], label="Learned")
        axes[2, j].plot(lagx, analytic_impulse[n//2], "--", label="Analytic (same FFT grid)")
        axes[2, j].set(xlabel="x lag at y=0 (µm)", ylabel="discrete weight")
        axes[2, j].legend(fontsize=7)
    fig.suptitle(f"Six shared kernels fitted on all {crop_count} crops × Bx/By/Bz | unregularized joint FFT least squares\n"
                 "Impulse heatmaps use a symmetric log color scale to show the peak and weaker tails")
    save_figure(fig, output, "02_learned_kernels.png", gallery, "Learned impulse responses, transfer spectra, and center-row profiles")


def plot_metric_summary(rows, output, gallery):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for ax, c in zip(axes, COMPONENTS):
        for evaluation, method, label in [("all_crop_fit", "learned", "All-crop fitting"),
                                          ("spatial_holdout", "learned", "Held-out crops"),
                                          ("spatial_holdout", "analytic", "Analytic: held-out crops")]:
            selected = [r for r in rows if r["component"] == c and r["crop_id"] == "all"
                        and r["evaluation"] == evaluation and r["method"] == method]
            ax.plot([r["z"] for r in selected], [100*r["nrmse_range"] if r["nrmse_range"] is not None else np.nan for r in selected], "o-", label=label)
        ax.set(title=c, xlabel="target height (µm)", ylabel="range NRMSE (%)")
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.suptitle("Fitting and held-out-crop errors | distinct evaluations, not independent geometries")
    save_figure(fig, output, "03_error_summary.png", gallery, "Error versus standoff: fitting, held-out crops, and analytic baseline")


def write_report(output, gallery, rows, manifest):
    selected = [r for r in rows if r["crop_id"] == "all"]
    table = []
    for r in selected:
        val = "undefined" if r["nrmse_range"] is None else f'{100*r["nrmse_range"]:.3f}%'
        table.append(f'<tr><td>{r["evaluation"]}</td><td>{r["z"]:g}</td><td>{r["method"]}</td>'
                     f'<td>{r["component"]}</td><td>{r["rmse_uT"]:.4f}</td><td>{val}</td></tr>')
    groups = [("Overview", lambda n: n.startswith(("01_", "02_", "03_"))),
              ("Initial crop pairs", lambda n: n.startswith("pairs_")),
              ("All-crop fitted predictions", lambda n: n.startswith("fit_")),
              ("Held-out crops", lambda n: n.startswith("spatial_")),
              ("Held-out height z=5 µm", lambda n: n.startswith("height_"))]
    sections = []
    for title, include in groups:
        figures = [f'<figure><a href="figures/{name}" target="_blank"><img loading="lazy" src="figures/{name}" alt="{html.escape(caption)}"></a>'
                   f'<figcaption>{html.escape(caption)} · click for full resolution</figcaption></figure>' for name, caption in gallery if include(name)]
        sections.append(f'<details open><summary>{title}</summary>{"".join(figures)}</details>')
    doc = '''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IC_4um crop kernel experiment</title><style>
body{font:16px system-ui;background:#f2f4f7;color:#172235;margin:0 auto;max-width:1500px;padding:28px}
h1{font-size:32px}p{max-width:1050px;line-height:1.6}summary{cursor:pointer;font-size:23px;font-weight:650;padding:18px 0}
figure{background:white;margin:16px 0;padding:12px;border-radius:12px}img{width:100%;height:auto}figcaption{padding:12px;color:#45546a}
table{border-collapse:collapse;background:white;font-size:14px}td,th{padding:8px 14px;border-bottom:1px solid #dde3eb;text-align:left}
.card{background:white;padding:18px;border-radius:12px;margin:16px 0}a{color:#135abd}</style></head><body>
<h1>IC_4um · crop-based blur kernels</h1>'''
    doc += f'<div class="card"><b>{manifest["crop_count"]} crops per plane · {manifest["crop_size"]}×{manifest["crop_size"]} pixels · Bx / By / Bz · six target heights</b>'
    doc += '<p>One shared direct FFT kernel per height pair. All-crop fits use every crop. Separate spatial-holdout fits exclude the green crops from fitting. The height-holdout prediction uses only the 0.5→3 and 3→4 training pairs: H(0.5→3) × H(3→4)², predicting z=5. Its target is used only for scoring. Direct z=5 fits elsewhere in this report are separate models.</p>'
    doc += '<p>All fields are µT. Each crop uses periodic FFT boundaries; crop seams and edge errors are part of this exploratory result. Errors are pooled over pixels before normalization. Held-out crops remain regions of the same geometry.</p>'
    doc += '<a href="metrics.csv">All numeric metrics (CSV)</a> · <a href="manifest.json">Configuration and data provenance</a> · <a href="kernels/height_holdout.json">Height-holdout protocol</a></div>'
    doc += ''.join(sections)
    doc += '<details><summary>Aggregate numeric results</summary><table><tr><th>Evaluation</th><th>Target µm</th><th>Method</th><th>Component</th><th>RMSE µT</th><th>Range NRMSE</th></tr>' + ''.join(table) + '</table></details></body></html>'
    (output/"index.html").write_text(doc)


def replot(output):
    """Rebuild the entire gallery from saved arrays, without fitting or CSV reads."""
    manifest = json.loads((output/"manifest.json").read_text())
    rows = json.loads((output/"metrics.json").read_text())["rows"]
    grid, size, test = manifest["grid"], manifest["crop_size"], manifest["spatial_test_ids"]
    train = manifest["spatial_train_ids"]
    crops = {z: np.load(output/"crops"/f"z{z:g}.npy", mmap_mode="r") for z in HEIGHTS}
    gallery, kernels = [], {}
    plot_grid(crops[.5], grid, size, test, output, gallery)
    for z in HEIGHTS[1:]:
        print(f"Plotting 0.5 → {z:g} µm", flush=True)
        kernels[z] = np.load(output/"kernels"/f"all_z{z:g}.npy")
        pred, spatial, analytic = [np.load(output/"predictions"/f"{name}_z{z:g}.npy", mmap_mode="r")
                                   for name in ("all_fit", "spatial", "analytic")]
        for c in range(3):
            plot_atlas(crops[.5], crops[z], z, c, output, gallery)
        plot_prediction(crops[.5], crops[z], pred, analytic, grid, z, output, gallery, f"fit_z{z:g}.png",
                        f"All-crop fit: 0.5 → {z:g} µm | all {len(crops[.5])} crops used for fitting")
        plot_prediction(crops[.5], crops[z], spatial, analytic, grid, z, output, gallery, f"spatial_z{z:g}.png",
                        f"Held-out crops: 0.5 → {z:g} µm | {len(train)} training, {len(test)} held out", test)
    hp = np.load(output/"predictions"/"height_holdout_z5.npy", mmap_mode="r")
    ha = np.load(output/"predictions"/"height_analytic_z5.npy", mmap_mode="r")
    plot_prediction(crops[.5], crops[5.], hp, ha, grid, 5., output, gallery, "height_holdout_z5.png",
                    "Height holdout: z=5 target excluded from fitting | 0.5→3 kernel, then 3→4 kernel twice")
    plot_kernels(kernels, grid, output, gallery, len(crops[.5]))
    plot_metric_summary(rows, output, gallery)
    write_report(output, gallery, rows, manifest)
    # Keep the original fitting provenance intact when plots are refreshed.
    (output/"plot_provenance.json").write_text(json.dumps({
        "script_sha256": sha256(__file__), "matplotlib": matplotlib.__version__,
        "figure_count": len(gallery)}, indent=2))
    print(f"Complete: {output/'index.html'} ({len(gallery)} figures)", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data"/"data2")
    parser.add_argument("--crop-size", type=int, default=280)
    parser.add_argument("--den-rtol", type=float, default=1e-12)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replot", action="store_true", help="regenerate plots/report in an existing completed output directory")
    args = parser.parse_args(argv)
    if args.crop_size <= 0 or not np.isfinite(args.den_rtol) or not 0 <= args.den_rtol < 1:
        parser.error("invalid crop size or denominator threshold")
    if args.replot:
        replot(args.output)
        return
    paths = {}
    for path in args.data_dir.glob("IC_4um_Z*um_600uA_0p1PR__MF.csv"):
        match = re.search(r"_Z(\d+)p(\d+)um_", path.name)
        z = float(f"{match[1]}.{match[2]}")
        if z in paths:
            raise ValueError(f"duplicate IC_4um height {z}")
        paths[z] = path
    if set(paths) != set(HEIGHTS):
        raise ValueError(f"expected IC_4um heights {HEIGHTS}; found {sorted(paths)}")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    for sub in ("crops", "kernels", "predictions", "figures"):
        (output/sub).mkdir()
    crops, sources = {}, []
    grid = None
    for z in HEIGHTS:
        print(f"Loading and validating IC_4um z={z:g}", flush=True)
        fields, current = load_plane(paths[z])
        if grid is not None and any(not np.isclose(current[key], grid[key], rtol=0, atol=1e-6) for key in grid):
            raise ValueError(f"grid mismatch at z={z}")
        grid = current
        batch = make_crops(fields, args.crop_size)
        dest = output/"crops"/f"z{z:g}.npy"
        np.save(dest, batch)
        crops[z] = np.load(dest, mmap_mode="r")
        sources.append(dict(height_um=z, path=str(paths[z].resolve()), sha256=sha256(paths[z])))
        del fields, batch
    train, test = split_ids(grid["ny"]//args.crop_size, grid["nx"]//args.crop_size)
    n = len(crops[.5]); all_ids = list(range(n))
    boxes = [dict(id=i, y_start=(i//(grid["nx"]//args.crop_size))*args.crop_size,
                  x_start=(i%(grid["nx"]//args.crop_size))*args.crop_size,
                  height=args.crop_size, width=args.crop_size) for i in all_ids]
    manifest = dict(dataset="IC_4um", grid=grid, field_unit="uT", length_unit="um", components=COMPONENTS,
                    crop_size=args.crop_size, crop_count=n, crop_layout="[crop, component, y, x]",
                    reference_height_um=.5, target_heights_um=HEIGHTS[1:], crop_coordinates=boxes,
                    spatial_train_ids=train, spatial_test_ids=test, den_rtol=args.den_rtol,
                    boundary="circular FFT independently on each crop; no padding or window",
                    fit="unregularized joint Fourier least squares, masked tiny denominators",
                    source_files=sources, code_sha256={p.name: sha256(p) for p in [Path(__file__), ROOT/"scripts"/"propagator.py"]},
                    environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__))
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2))
    rows, diagnostics = [], {}
    freq = k_grid(crops[.5].shape[-2:], grid["dx"], grid["dy"])
    # Fit the held-out-height model before any direct z=5 kernel; no z=5 input.
    hp, anchor, step, composed, protocol = height_holdout(crops[.5], crops[3.], crops[4.], args.den_rtol)
    for name, value in [("height_anchor_0p5_to_3", anchor), ("height_step_3_to_4", step), ("height_composed_to_5", composed)]:
        np.save(output/"kernels"/f"{name}.npy", value)
    (output/"kernels"/"height_holdout.json").write_text(json.dumps(protocol, indent=2))
    np.save(output/"predictions"/"height_holdout_z5.npy", hp)
    ha = apply_kernel(crops[.5], analytic_transfer(freq, 4.5))
    np.save(output/"predictions"/"height_analytic_z5.npy", ha)
    for method, prediction in [("composed_learned", hp), ("analytic", ha)]:
        rows += score_rows(prediction, crops[5.], 5., "height_holdout", method, all_ids)
    del hp, ha
    for z in HEIGHTS[1:]:
        print(f"Fitting all-crop and spatial-holdout kernels: 0.5 → {z:g} µm", flush=True)
        sharp, blurred = crops[.5], crops[z]
        h, info = fit_kernel(sharp, blurred, den_rtol=args.den_rtol)
        held, held_info = fit_kernel(sharp, blurred, train, args.den_rtol)
        diagnostics[str(z)] = dict(all_crop_fit=info, spatial_holdout_fit=held_info)
        for name, value in [(f"all_z{z:g}", h), (f"spatial_z{z:g}", held),
                            (f"all_z{z:g}_impulse_centered", np.fft.fftshift(np.fft.ifft2(h).real))]:
            np.save(output/"kernels"/f"{name}.npy", value)
        pred, spatial = apply_kernel(sharp, h), apply_kernel(sharp, held)
        analytic = apply_kernel(sharp, analytic_transfer(freq, z-.5))
        for name, prediction in [("all_fit", pred), ("spatial", spatial), ("analytic", analytic)]:
            np.save(output/"predictions"/f"{name}_z{z:g}.npy", prediction)
        for evaluation, prediction, method, ids in [("all_crop_fit", pred, "learned", all_ids),
                                                   ("all_crop_fit", analytic, "analytic", all_ids),
                                                   ("spatial_holdout", spatial, "learned", test),
                                                   ("spatial_holdout", analytic, "analytic", test)]:
            rows += score_rows(prediction, blurred, z, evaluation, method, ids)
        del pred, spatial, analytic
    with (output/"metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output/"metrics.json").write_text(json.dumps(dict(rows=rows, fit_diagnostics=diagnostics), indent=2, allow_nan=False))
    replot(output)


if __name__ == "__main__":
    main()
