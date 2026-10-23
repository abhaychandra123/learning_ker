"""
Look at every field plane before doing anything quantitative with them.

Produces, in figures/:
    overview.png        all 4 components at 4 representative standoffs
    montage_<comp>.png  one component across all 11 standoffs   (x4)
    zoom_Bz.png         a 20 um crop, per-panel normalised - blur is visible here
    linecuts.png        one row through the coil, every standoff overlaid
    decay.png           peak and rms vs standoff, against exp(-2*pi*k*z)
    spectra.png         radial spectra + the measured/analytic transfer ratio

Colour rules: signed components get a diverging map with a neutral midpoint and
limits symmetric about zero; |B| gets a single-hue sequential map. Standoff is
an ordered variable, so the 11 curves are sampled from one sequential ramp with
a colourbar rather than 11 categorical hues.

    python scripts/visualize.py
"""

import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable

import fields as F
import propagator as P

DIVERGING = "RdBu_r"        # two hues, neutral midpoint
SEQUENTIAL = "magma"        # single-hue magnitude ramp
ORDERED = "viridis"         # for the ordered standoff variable
CLIP_PCT = 99.5             # percentile for symmetric colour limits
ZOOM_UM = 20.0

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,             # recessive grid
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 9,
})


# ------------------------------------------------------------------ helpers

def z_colors(zs):
    """One colour per standoff, sampled from a sequential ramp."""
    norm = Normalize(vmin=min(zs), vmax=max(zs))
    cmap = matplotlib.colormaps[ORDERED]
    return [cmap(norm(z)) for z in zs], ScalarMappable(norm=norm, cmap=cmap)


def show_plane(ax, field, comp, grid, vmax=None, norm_to_peak=False):
    """Draw one field plane with the colour rule appropriate to its component."""
    data = field / np.abs(field).max() if norm_to_peak else field
    if comp in F.SIGNED:
        v = vmax if vmax is not None else np.percentile(np.abs(data), CLIP_PCT)
        im = ax.imshow(data, origin="lower", extent=grid.extent,
                       cmap=DIVERGING, vmin=-v, vmax=v, interpolation="nearest")
    else:
        v = vmax if vmax is not None else np.percentile(data, CLIP_PCT)
        im = ax.imshow(data, origin="lower", extent=grid.extent,
                       cmap=SEQUENTIAL, vmin=0, vmax=v, interpolation="nearest")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.grid(False)
    return im


def interesting_row(grid):
    """The grid row with the most Bz structure - used for the line cuts."""
    bz = F.load(F.z_values()[0], "Bz")
    inset = int(30.0 / grid.dx)                 # skip the feed traces, which dominate the variance
    var = bz[:, inset:].var(axis=1)
    var[:inset] = 0.0
    return int(np.argmax(var))


def zoom_box(grid, size_um=ZOOM_UM):
    """A square window centred on the strongest feature, kept inside the grid."""
    bz = F.load(F.z_values()[0], "Bz")
    r, c = np.unravel_index(np.abs(bz).argmax(), bz.shape)
    half = size_um / 2
    cx = float(np.clip(grid.x[c], grid.x[0] + half, grid.x[-1] - half))
    cy = float(np.clip(grid.y[r], grid.y[0] + half, grid.y[-1] - half))
    return (cx - half, cx + half), (cy - half, cy + half)


# ------------------------------------------------------------------ figures

def fig_overview(grid, zs):
    picks = [zs[0], 2.0, 5.0, 10.0]
    fig, axes = plt.subplots(len(picks), len(F.COMPONENTS),
                             figsize=(13, 3.1 * len(picks)),
                             constrained_layout=True)
    for i, z in enumerate(picks):
        for j, comp in enumerate(F.COMPONENTS):
            ax = axes[i, j]
            im = show_plane(ax, F.load(z, comp), comp, grid)
            ax.set_title(f"{F.LABEL[comp]}   z = {z} um")
            fig.colorbar(im, ax=ax, fraction=0.046, label="uT")
            if j:
                ax.set_ylabel("")
            if i < len(picks) - 1:
                ax.set_xlabel("")
    fig.suptitle("Field components at four standoffs "
                 "(colour limits are per-panel: peak spans ~200x across z)",
                 fontsize=11)
    fig.savefig(F.FIGURES / "overview.png", bbox_inches="tight")
    plt.close(fig)


def fig_montage(grid, zs, comp):
    ncol = 4
    nrow = int(np.ceil(len(zs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.3 * nrow),
                             constrained_layout=True)
    for ax, z in zip(axes.ravel(), zs):
        im = show_plane(ax, F.load(z, comp), comp, grid)
        ax.set_title(f"z = {z} um")
        fig.colorbar(im, ax=ax, fraction=0.046, label="uT")
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axes.ravel()[len(zs):]:
        ax.axis("off")
    fig.suptitle(f"{F.LABEL[comp]} across all standoffs "
                 f"(per-panel colour limits)", fontsize=11)
    fig.savefig(F.FIGURES / f"montage_{comp}.png", bbox_inches="tight")
    plt.close(fig)


def fig_zoom(grid, zs):
    """Same crop, each panel normalised to its own peak: pure blur, no decay."""
    xr, yr = zoom_box(grid)
    sy, sx = grid.window(xr, yr)
    extent = (xr[0], xr[1], yr[0], yr[1])

    ncol = 4
    nrow = int(np.ceil(len(zs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.0 * nrow),
                             constrained_layout=True)
    for ax, z in zip(axes.ravel(), zs):
        crop = F.load(z, "Bz")[sy, sx]
        crop = crop / np.abs(crop).max()
        ax.imshow(crop, origin="lower", extent=extent, cmap=DIVERGING,
                  vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"z = {z} um")
        ax.grid(False)
    for ax in axes.ravel()[len(zs):]:
        ax.axis("off")
    fig.suptitle(f"$B_z$ over the same {ZOOM_UM:.0f} um window, each panel "
                 f"normalised to its own peak - this is the blur, isolated",
                 fontsize=11)
    fig.savefig(F.FIGURES / "zoom_Bz.png", bbox_inches="tight")
    plt.close(fig)


def fig_linecuts(grid, zs):
    row = interesting_row(grid)
    y_um = grid.y[row]
    colors, sm = z_colors(zs)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                             constrained_layout=True)
    for z, c in zip(zs, colors):
        cut = F.load(z, "Bz")[row]
        axes[0].plot(grid.x, cut, color=c, lw=1.2)
        axes[1].plot(grid.x, cut / np.abs(cut).max(), color=c, lw=1.2)

    axes[0].set_ylabel("$B_z$ (uT)")
    axes[0].set_title(f"$B_z$ along y = {y_um:.2f} um - absolute")
    axes[1].set_ylabel("$B_z$ / peak")
    axes[1].set_xlabel("x (um)")
    axes[1].set_title("normalised to each curve's own peak: "
                      "shape only, amplitude decay divided out")
    for ax in axes:
        cb = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.035)
        cb.set_label("standoff z (um)")
    fig.savefig(F.FIGURES / "linecuts.png", bbox_inches="tight")
    plt.close(fig)


def fig_decay(grid, zs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    cmap = matplotlib.colormaps[ORDERED]
    comp_color = {c: cmap(i / 3) for i, c in enumerate(F.COMPONENTS)}

    for comp in F.COMPONENTS:
        peak = [np.abs(F.load(z, comp)).max() for z in zs]
        rms = [np.sqrt(np.mean(F.load(z, comp).astype(np.float64) ** 2))
               for z in zs]
        axes[0].semilogy(zs, peak, "o-", ms=4, lw=1.5,
                         color=comp_color[comp], label=F.LABEL[comp])
        axes[1].semilogy(zs, rms, "o-", ms=4, lw=1.5,
                         color=comp_color[comp], label=F.LABEL[comp])

    for ax, title in zip(axes, ("peak |field|", "rms field")):
        ax.set_xlabel("standoff z (um)")
        ax.set_ylabel("uT")
        ax.set_title(title)
        ax.legend(frameon=False)
    fig.suptitle("Amplitude decay with standoff (log scale)", fontsize=11)
    fig.savefig(F.FIGURES / "decay.png", bbox_inches="tight")
    plt.close(fig)


def fig_spectra(grid, zs):
    """Radial spectra, and the measured transfer ratio against exp(-2*pi*k*dz).

    Right panel is the anchor test: if the reference plane is clean, every
    measured ratio sits on its dashed analytic curve until the mesh noise floor.
    """
    colors, sm = z_colors(zs)
    ref = zs[0]

    spectra = {}
    for z in zs:
        k, amp = P.radial_spectrum(F.load(z, "Bz"), grid.dx)
        spectra[z] = amp
    k_ax = k

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    for z, c in zip(zs, colors):
        axes[0].semilogy(k_ax, spectra[z], color=c, lw=1.2)
    axes[0].set_xlim(0, 2.0)
    axes[0].set_ylim(1e-8, None)
    axes[0].set_xlabel("spatial frequency k (cycles / um)")
    axes[0].set_ylabel("radially averaged |B_z(k)| (uT)")
    axes[0].set_title("$B_z$ spectrum at each standoff")
    fig.colorbar(sm, ax=axes[0], pad=0.01, fraction=0.04,
                 label="standoff z (um)")

    valid = spectra[ref] > 0
    for z, c in zip(zs[1:], colors[1:]):
        ratio = np.divide(spectra[z], spectra[ref],
                          out=np.full_like(spectra[z], np.nan), where=valid)
        axes[1].semilogy(k_ax, ratio, color=c, lw=1.3)
        axes[1].semilogy(k_ax, P.analytic_transfer(k_ax, z - ref),
                         color=c, lw=1.0, ls="--", alpha=0.75)
    axes[1].set_xlim(0, 1.5)
    axes[1].set_ylim(1e-6, 3)
    axes[1].set_xlabel("spatial frequency k (cycles / um)")
    axes[1].set_ylabel(f"|B_z(k, z)| / |B_z(k, {ref} um)|")
    axes[1].set_title(f"measured transfer (solid) vs exp(-2$\\pi$k$\\Delta$z) "
                      f"(dashed), anchored at z = {ref} um")
    fig.colorbar(sm, ax=axes[1], pad=0.01, fraction=0.04,
                 label="standoff z (um)")

    fig.suptitle("Hann-windowed to suppress leakage from the truncated edge; "
                 "diagnostic only - the unbiased estimate uses the interior mask",
                 fontsize=9)
    fig.savefig(F.FIGURES / "spectra.png", bbox_inches="tight")
    plt.close(fig)


def fig_edges(grid, zs):
    """Where does the field touch the domain boundary?

    The percentile-clipped montages hide this: two feed traces leave the coil
    and run out through the x_min edge still carrying current, so the field is
    discontinuous across that boundary when the FFT wraps it. Everything else
    decays inside the domain.
    """
    z = zs[0]
    bz = F.load(z, "Bz")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    v = 20.0
    im = axes[0].imshow(bz, origin="lower", extent=grid.extent, cmap=DIVERGING,
                        vmin=-v, vmax=v, interpolation="nearest")
    axes[0].set_title(f"$B_z$ at z = {z} um, hard-clipped to $\\pm${v:.0f} uT\n"
                      f"(peak is {np.abs(bz).max():.0f} uT)")
    fig.colorbar(im, ax=axes[0], fraction=0.046, label="uT")

    xr, yr = (-42, -10), (-30, -5)
    sy, sx = grid.window(xr, yr)
    im = axes[1].imshow(bz[sy, sx], origin="lower",
                        extent=(xr[0], xr[1], yr[0], yr[1]), cmap=DIVERGING,
                        vmin=-120, vmax=120, interpolation="nearest")
    axes[1].set_title("the two feed traces, leaving through x_min")
    fig.colorbar(im, ax=axes[1], fraction=0.046, label="uT")

    colors, sm = z_colors(zs)
    for zz, c in zip(zs, colors):
        axes[2].plot(grid.y, F.load(zz, "Bz")[:, 0], color=c, lw=1.1)
    axes[2].set_xlabel("y (um)")
    axes[2].set_ylabel("$B_z$ on the x_min column (uT)")
    axes[2].set_title("field at the truncated edge, every standoff")
    fig.colorbar(sm, ax=axes[2], pad=0.01, fraction=0.04, label="standoff z (um)")

    for ax in axes[:2]:
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.grid(False)
    fig.savefig(F.FIGURES / "edges.png", bbox_inches="tight")
    plt.close(fig)


def fig_planes(grid, zs, subdir="planes"):
    """One full-resolution image per (component, standoff) - 44 files.

    The montages are for scanning; these are for looking. Rendered at roughly
    one screen pixel per grid sample so the mesh facets are actually visible.
    """
    out = F.FIGURES / subdir
    out.mkdir(exist_ok=True)
    for comp in F.COMPONENTS:
        for z in zs:
            field = F.load(z, comp)
            fig, ax = plt.subplots(figsize=(9.6, 9.0), constrained_layout=True)
            im = show_plane(ax, field, comp, grid)
            peak = np.abs(field).max()
            ax.set_title(f"{F.LABEL[comp]}   z = {z} um   "
                         f"(peak {peak:.2f} uT, colour limits at "
                         f"{CLIP_PCT} percentile)")
            fig.colorbar(im, ax=ax, fraction=0.046, label="uT")
            fig.savefig(out / f"{comp}_{F.z_tag(z)}.png", bbox_inches="tight")
            plt.close(fig)
        print(f"  wrote figures/{subdir}/{comp}_z*.png  ({len(zs)} files)")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    F.FIGURES.mkdir(exist_ok=True)
    grid = F.grid()
    zs = list(F.z_values())
    print(f"grid {grid.nx}x{grid.ny} @ {grid.dx:.3f} um, {len(zs)} standoffs")

    if "planes" in argv:
        fig_planes(grid, zs)
        return

    for name, fn in [("overview", lambda: fig_overview(grid, zs)),
                     ("edges", lambda: fig_edges(grid, zs)),
                     ("zoom_Bz", lambda: fig_zoom(grid, zs)),
                     ("linecuts", lambda: fig_linecuts(grid, zs)),
                     ("decay", lambda: fig_decay(grid, zs)),
                     ("spectra", lambda: fig_spectra(grid, zs))]:
        fn()
        print(f"  wrote figures/{name}.png")

    for comp in F.COMPONENTS:
        fig_montage(grid, zs, comp)
        print(f"  wrote figures/montage_{comp}.png")


if __name__ == "__main__":
    main()
