"""Shared NumPy pieces for milestones M1 and M2 (M1_M2_PROTOCOL.md). No torch here, so tests run on the Mac.

Conventions as in the rest of the repo: fields [3, H, W] in uT, components (Bx, By, Bz) in the simulator's
(COMSOL export) frame, pixel pitch 0.1 um, k in rad/um, FFT arrays in full (unshifted) layout.
"""
import numpy as np

DX = 0.1
PAD = 128               # padding of the old Tikhonov baseline and of information_budget.py
MU_GRID = 10.0 ** np.arange(-4, 0.001, 0.25)            # M1b: 1e-4 ... 1, 17 values
SIM_STANDOFFS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0)
COARSE_HEIGHTS = np.round(np.arange(0.5, 12.0001, 0.1), 2)   # M1a coarse grid
FINE_STEP, FINE_HALF = 0.01, 0.1
EDGE = (0.5, 12.0)
SHAPE = {'IC_2um': 'IC', 'IC_4um': 'IC', 'L1_C1': 'C1', 'C1_C2': 'C1', 'C1_C2_C3': 'C1',
         'C1_C2_C3_C4': 'C1'}


def wavenumbers(n, dx=DX):
    """kx (along columns), ky (along rows) and |k| on an n x n full FFT grid, rad/um."""
    f = 2 * np.pi * np.fft.fftfreq(n, dx)
    return f[None, :], f[:, None], np.hypot(f[None, :], f[:, None])


def padded_fft(a, pad=PAD):
    """FFT of each [..., H, W] image after zero padding by `pad` on every side (square images)."""
    width = [(0, 0)] * (a.ndim - 2) + [(pad, pad), (pad, pad)]
    return np.fft.fft2(np.pad(np.asarray(a, np.float64), width))


def crop_ifft(spectrum, pad=PAD):
    out = np.real(np.fft.ifft2(spectrum))
    return out[..., pad:out.shape[-2] - pad, pad:out.shape[-1] - pad]


def relative_l2(pred, truth):
    """Per-component relative L2 for [3, H, W] arrays, as a length-3 array."""
    pred, truth = np.asarray(pred, np.float64), np.asarray(truth, np.float64)
    return np.sqrt(((pred - truth) ** 2).sum((-2, -1)) / (truth ** 2).sum((-2, -1)))


# ---------------------------------------------------------------- M1b: data-consistency correction

def tikhonov_correct(x, b, dz, mu, pad=PAD, dx=DX):
    """One Tikhonov data-consistency step per component: X + H/(H^2+mu) (B - H X), H = exp(-k dz).

    x: predicted field at the target height [3, H, W]; b: input field at the input height [3, H, W];
    dz: input height minus target height (um). Same zero padding as the old Tikhonov baseline.
    """
    n = np.asarray(x).shape[-1] + 2 * pad
    _, _, k = wavenumbers(n, dx)
    h = np.exp(-k * dz)
    fx, fb = padded_fft(x, pad), padded_fft(b, pad)
    return crop_ifft(fx + h / (h * h + mu) * (fb - h * fx), pad)


def mu_at(h, standoffs, mus):
    """log10 mu linearly interpolated in h; held constant outside the simulated range (protocol M1b)."""
    return float(10 ** np.interp(h, np.asarray(standoffs, float), np.log10(np.asarray(mus, float))))


# ---------------------------------------------------------------- M1a: height read-out

def sheet_spectra(j, pad=700, dx=DX):
    """rfft of a [2, H, W] sheet current zero padded by `pad`, plus the wavenumbers used by Biot-Savart."""
    jx = np.pad(np.asarray(j[0], np.float64), pad)
    jy = np.pad(np.asarray(j[1], np.float64), pad)
    ny, nx = jx.shape
    wx = 2 * np.pi * np.fft.rfftfreq(nx, dx)[None, :]
    wy = 2 * np.pi * np.fft.fftfreq(ny, dx)[:, None]
    k = np.hypot(wx, wy)
    safe = k.copy()
    safe[0, 0] = 1.0
    fx, fy = np.fft.rfft2(jx), np.fft.rfft2(jy)
    fz = 1j * (wy * fx - wx * fy) / safe
    return (fx, fy, fz), k, (ny, nx)


def field_at(spectra, k, shape, height, image, pad=700):
    """Same formulas as magic_unet.field_from_sheet (thin-sheet Biot-Savart, uT)."""
    mu0_ut = 4e-7 * np.pi * 1e6
    decay = 0.5 * mu0_ut * np.exp(-k * height)
    return np.stack([np.fft.irfft2(f * decay, s=shape)[pad:pad + image, pad:pad + image]
                     for f in spectra])


def height_misfit(j, b, heights, pad=700, dx=DX):
    """misfit(h) = mean over components of ||forward(J, h) - b|| / ||b|| for each h (protocol M1a)."""
    spectra, k, shape = sheet_spectra(j, pad, dx)
    image = b.shape[-1]
    return np.array([relative_l2(field_at(spectra, k, shape, h, image, pad), b).mean()
                     for h in heights])


def coarse_heights(h_max):
    """The M1a coarse grid 0.5, 0.6, ..., h_max um (h_max = 12 in M1; 20 for the 10 um runs)."""
    return np.round(np.arange(0.5, h_max + 1e-4, 0.1), 2)


def read_height(misfit_fn, h_max=EDGE[1]):
    """Coarse grid, then a 0.01 um grid over +-0.1 um around the coarse minimum. Returns (h*, misfit, edge).

    misfit_fn maps an array of heights to an array of misfits. The grid runs from 0.5 um to h_max.
    """
    grid, lo = coarse_heights(h_max), EDGE[0]
    coarse = misfit_fn(grid)
    h0 = grid[int(np.argmin(coarse))]
    fine_h = np.round(np.arange(h0 - FINE_HALF, h0 + FINE_HALF + 1e-9, FINE_STEP), 3)
    fine_h = fine_h[(fine_h >= lo) & (fine_h <= h_max)]
    fine = misfit_fn(fine_h)
    i = int(np.argmin(fine))
    h = float(fine_h[i])
    edge = bool(h <= lo + 0.1 or h >= h_max - 0.1)
    return h, float(fine[i]), edge


# ---------------------------------------------------------------- M2a: Maxwell consistency and noise cutoff

def sheet_consistent(spectrum, kx, ky, k):
    """Least-squares projection of (Bx, By, Bz) spectra [3, n, n] onto fields of one current sheet below.

    Above a sheet, Bx = (ky/k) phi, By = -(kx/k) phi, Bz = i phi in the simulator's frame (simulate_fields.py,
    divergence-free current). The k = 0 term is kept as it is.
    """
    safe = k.copy()
    safe[0, 0] = 1.0
    ux, uy, uz = ky / safe, -kx / safe, 1j * np.ones_like(k)
    phi = (np.conj(ux) * spectrum[0] + np.conj(uy) * spectrum[1] + np.conj(uz) * spectrum[2]) / 2
    out = np.stack([ux * phi, uy * phi, uz * phi])
    out[:, 0, 0] = spectrum[:, 0, 0]
    return out


def inconsistency_profile(targets, pad=PAD, dx=DX, width=0.1, k_max=31.4):
    """Pooled radial profile of total and Maxwell-inconsistent energy over a list of [3, H, W] targets.

    Returns bin lower edges (rad/um), total energy and inconsistent energy per bin.
    """
    edges = np.arange(0, k_max + width, width)
    total = np.zeros(len(edges) - 1)
    bad = np.zeros(len(edges) - 1)
    for t in targets:
        f = padded_fft(t, pad)
        kx, ky, k = wavenumbers(f.shape[-1], dx)
        r = f - sheet_consistent(f, kx, ky, k)
        idx = np.digitize(k, edges) - 1
        ok = (idx >= 0) & (idx < len(total))
        total += np.bincount(idx[ok], (np.abs(f) ** 2).sum(0)[ok], len(total))
        bad += np.bincount(idx[ok], (np.abs(r) ** 2).sum(0)[ok], len(total))
    return edges[:-1], total, bad


def noise_cutoff(lower, total, bad, k_min=1.0, share=0.5):
    """Lower edge of the first bin at or above k_min whose inconsistent share exceeds `share` (M2a)."""
    frac = bad / np.maximum(total, 1e-300)
    for edge, f in zip(lower, frac):
        if edge >= k_min - 1e-9 and f > share:
            return float(edge)
    raise ValueError('no bin above k_min exceeds the share')


# ---------------------------------------------------------------- M2b: band split and four-way score

def snr_crossing(sharp_c, blurred_c, dz, pad=PAD, dx=DX):
    """First-crossing SNR = 1 frequency of information_budget.py for one component (0.05 rad/um bins)."""
    fs, fb = padded_fft(sharp_c, pad), padded_fft(blurred_c, pad)
    _, _, k = wavenumbers(fs.shape[-1], dx)
    h = np.exp(-k * dz)
    sig = np.abs(h * fs) ** 2
    res = np.abs(fb - h * fs) ** 2
    for a in np.arange(0, 8, 0.05):
        m = (k >= a) & (k < a + 0.05)
        if sig[m].sum() < res[m].sum():
            return float(a)
    return 8.0


def fourway(pred, sharp, blurred, dz, k_noise, pad=PAD, dx=DX):
    """Per-component dict of the four M2b scores plus the target's norm share ‖T_band‖/‖T‖ in each band.

    All band quantities are norm fractions of the whole target, so in_band² + prior_band² + noise_band² = raw²."""
    fp, ft = padded_fft(pred, pad), padded_fft(sharp, pad)
    _, _, k = wavenumbers(fp.shape[-1], dx)
    rows = []
    for c in range(3):
        kc = snr_crossing(sharp[c], blurred[c], dz, pad, dx)
        norm = np.sqrt((np.asarray(sharp[c], np.float64) ** 2).sum())
        err = fp[c] - ft[c]
        n_pix = fp.shape[-1] ** 2               # Parseval: sum |F|^2 = N * sum |x|^2

        def band(mask, a):
            return float(np.sqrt((np.abs(a) ** 2)[mask].sum() / n_pix) / norm)
        low = k < k_noise
        lp_t = crop_ifft(ft[c] * low, pad)
        lp_p = crop_ifft(fp[c] * low, pad)
        rows.append(dict(
            raw=float(np.sqrt(((np.asarray(pred[c], np.float64) - sharp[c]) ** 2).sum()) / norm),
            lowpass=float(np.linalg.norm(lp_p - lp_t) / np.linalg.norm(lp_t)),
            in_band=band(k < kc, err), prior_band=band((k >= kc) & (k < k_noise), err),
            noise_band=band(k >= k_noise, err), k_c=kc,
            target_in=band(k < kc, ft[c]), target_prior=band((k >= kc) & (k < k_noise), ft[c]),
            target_noise=band(k >= k_noise, ft[c])))
    return rows
