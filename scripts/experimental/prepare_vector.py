"""Experimental B_NV map -> (Bx, By, Bz) at 0.1 um, ready for MAGIC-UNet, plus the fitted-coil reference.

Steps per case (all fixed before any network output was seen):
  1. replace masked pixels (fit-failure spikes, noisy corners) by a 5x5 median of the good pixels;
  2. subtract the background fitted in fit_coil.py (offset, plane, camera-seam step);
  3. NV projection -> vector, assuming a source-free field above the sample (u from the fit):
        Bz(k) = B_NV(k) / (u_z - i (u_x k_x + u_y k_y)/k),  Bx = -i k_x/k Bz,  By = -i k_y/k Bz
     (165 px zero-padded to 512 with a 10 px cosine taper at the image edge);
  4. Fourier interpolation 1 um -> 0.1 um (165 -> 1650 px);
  5. convert to the simulator / MAGIC-UNet component frame: (Bx_m, By_m, Bz_m) = (-By, Bx, -Bz).
Reference: the fitted DXF coil (orientation, centre, standoff, current) at 0.1 um: sheet current [2, H, W] (A/m)
and its field at 0.5 um and at the fitted standoff, in the same frame.
Usage: python3 prepare_vector.py <fit dir(s), comma-separated> <out.npz> [check]
"""
import glob, json, os, sys
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fit_coil as fc
from simulate_fields import rasterize, fields_from_current

UP = 10           # 1 um -> 0.1 um


def nv_to_vector(b, u, pad_to=512, taper=10):
    n = b.shape[0]
    w = np.ones(n)
    w[:taper] = 0.5 - 0.5*np.cos(np.pi*np.arange(taper)/taper); w[-taper:] = w[:taper][::-1]
    a = np.zeros((pad_to, pad_to)); a[:n, :n] = b*np.outer(w, w)
    k = 2*np.pi*np.fft.fftfreq(pad_to, 1.0)
    KY, KX = np.meshgrid(k, k, indexing="ij"); K = np.hypot(KX, KY); Ks = np.where(K == 0, 1, K)
    F = np.fft.fft2(a)
    den = u[2] - 1j*(u[0]*KX + u[1]*KY)/Ks
    Fz = np.where(K == 0, 0, F/den)
    Fx, Fy = -1j*KX/Ks*Fz, -1j*KY/Ks*Fz
    return [Fx, Fy, Fz], n


def upsample(spectra, n, pad_to=512, up=UP):
    """Fourier interpolation of the padded spectra to `up` x finer pixels; crop to the n x n image."""
    big = pad_to*up
    out = []
    for F in spectra:
        G = np.zeros((big, big), complex)
        h = pad_to//2
        G[:h, :h] = F[:h, :h]; G[:h, -h:] = F[:h, -h:]; G[-h:, :h] = F[-h:, :h]; G[-h:, -h:] = F[-h:, -h:]
        g = np.real(np.fft.ifft2(G))*up*up
        # pixel centre of 1 um pixel j is at 0.5 + j; fine pixel i centre at 0.05 + 0.1 i: shift by -0.45 um
        g = np.roll(g, (up//2 - 1 + 1, up//2 - 1 + 1), axis=(0, 1)) if False else g
        out.append(g[:n*up, :n*up])
    return np.array(out)


def fine_shift(field, shift_um, dx=0.1):
    """Sub-pixel shift by Fourier phase (field sampled at 0.1 um)."""
    n = field.shape[-1]
    k = 2*np.pi*np.fft.fftfreq(n, dx)
    ph = np.exp(-1j*k*shift_um)
    return np.real(np.fft.ifft2(np.fft.fft2(field)*ph[None, :, None]*ph[None, None, :]))


def reference(fit, n_um=165):
    """Fitted DXF coil on the 0.1 um grid of the experimental image (pixel centres 0.05 + 0.1 i)."""
    o = (fit["orientation"]["rot90"], fit["orientation"]["mirror"])
    x0, y0 = fit["centre_um"]
    pad = 600
    n = n_um*UP + 2*pad
    lim = pad*0.1 - 5.0                                               # keep leads inside the padded canvas
    for lead in np.arange(fc.LEAD_END, 20.0, -5.0):
        path = [(-21, lead)] + fc.SPIRAL + [(-15, lead), (-21, lead)]
        q = fc.transform(path, o, (x0, y0))
        if q.min() > -lim and q.max() < n_um + lim:
            break
    p = q/0.1 + pad - 0.5                                             # rasterize: pixel i centre at i*dx
    kx, ky = rasterize(p*0.1, 2.0, fit["current_uA"]*1e-6, (n, n), 0.1, 0)
    crop = (slice(pad, pad + n_um*UP), slice(pad, pad + n_um*UP))
    f05, fh = fields_from_current(kx, ky, 0.1, [0.5, fit["standoff_um"]])
    j = np.stack([kx[crop], ky[crop]])
    return j, np.stack([c[crop] for c in f05]), np.stack([c[crop] for c in fh])


def main():
    dirs, out = sys.argv[1].split(","), sys.argv[2]
    summary = {}
    for d in dirs:
        summary.update(json.load(open(f"{d}/fit_summary.json")))
    arrays, meta = {}, {}
    for case, fit in sorted(summary.items()):
        f = np.load(glob.glob(f"{dirs[0]}/fit_{case}.npz")[0] if os.path.exists(f"{dirs[0]}/fit_{case}.npz")
                    else [p for d in dirs for p in glob.glob(f"{d}/fit_{case}.npz")][0])
        d, m = f["data"], f["full_mask"]
        X, Y = np.meshgrid(f["X"], f["Y"])
        med = ndimage.generic_filter(np.where(m, d, np.nan), np.nanmedian, 5, mode="nearest")
        clean = np.where(m, d, med)
        bg = sum(c*b for c, b in zip(fit["background_uT"], fc.background_cols(X, Y)))
        b = clean - bg
        th, ph = np.deg2rad(fit["nv_theta_deg"]), np.deg2rad(fit["nv_phi_deg"])
        u = fc.unit(th, ph)
        spectra, n = nv_to_vector(b, u)
        phys = upsample(spectra, n)                                   # physical (Bx, By, Bz), fine grid
        phys = fine_shift(phys, 0.45)                                # 1 um pixel centres (0.5+j) -> fine centres (0.05+0.1i)
        model_frame = np.stack([-phys[1], phys[0], -phys[2]])
        j, f05, fh = reference(fit)
        arrays[f"{case}_input"] = model_frame.astype(np.float32)
        arrays[f"{case}_bnv_clean"] = b.astype(np.float32)
        arrays[f"{case}_ref_j"] = j.astype(np.float32)
        arrays[f"{case}_ref_f05"] = f05.astype(np.float32)
        arrays[f"{case}_ref_fh"] = fh.astype(np.float32)
        meta[case] = dict(u=u.tolist(), standoff_fit_um=fit["standoff_um"], current_fit_uA=fit["current_uA"],
                          centre_um=fit["centre_um"], orientation=fit["orientation"])
        print(case, "input max |B|", float(np.abs(model_frame).max()), "uT; ref max |J|", float(np.abs(j).max()), "A/m")
    np.savez_compressed(out, meta=json.dumps(meta), **arrays)


def check():
    """Round trip on the fitted-coil model: B_NV from the true vector, back to the vector."""
    fit = dict(orientation=dict(rot90=1, mirror=0), centre_um=[55.0, 48.0], standoff_um=3.4, current_uA=150.0,
               nv_theta_deg=44.0, nv_phi_deg=172.0)
    j, f05, fh = reference(fit)
    phys_true = np.stack([fh[1], -fh[0], -fh[2]])                       # model frame -> physical
    phys1 = phys_true.reshape(3, 165, 10, 165, 10).mean((2, 4))        # 1 um pixels
    u = fc.unit(np.deg2rad(44.0), np.deg2rad(172.0))
    bnv = np.einsum("c,cyx->yx", u, phys1)
    spectra, n = nv_to_vector(bnv, u)
    rec1 = np.array([np.real(np.fft.ifft2(F))[:n, :n] for F in spectra])
    core = (slice(15, -15), slice(15, -15))
    print("round trip at 1 um, relative L2 Bx/By/Bz (inner 135 um):",
          [f"{np.linalg.norm(rec1[c][core] - phys1[c][core])/np.linalg.norm(phys1[c][core]):.4f}" for c in range(3)])
    rec = fine_shift(upsample(spectra, n), 0.45)
    corefine = (slice(150, -150), slice(150, -150))
    print("after 0.1 um interpolation vs true 0.1 um field:",
          [f"{np.linalg.norm(rec[c][corefine] - phys_true[c][corefine])/np.linalg.norm(phys_true[c][corefine]):.4f}" for c in range(3)])


if __name__ == "__main__":
    check() if len(sys.argv) > 1 and sys.argv[1] == "check" else main()
