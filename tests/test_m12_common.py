"""NumPy checks for scripts/m12_common.py (M1_M2_PROTOCOL.md). Run: python3 -m unittest tests.test_m12_common"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import m12_common as mc  # noqa: E402
from simulate_fields import fields_from_current  # noqa: E402


def sheet(n=96, seed=0):
    """A smooth random sheet current [2, n, n] (A/m), compact in the middle of the image."""
    rng = np.random.default_rng(seed)
    j = rng.standard_normal((2, n, n))
    f = np.fft.fft2(j)
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n)))
    j = np.real(np.fft.ifft2(f * np.exp(-(k / 0.08) ** 2)))
    win = np.zeros((n, n))
    win[n // 4:3 * n // 4, n // 4:3 * n // 4] = 1
    j = j * win * 1e3
    # Divergence-free part, as in magic_unet.solenoidal_sheet (a physical sheet current conserves charge).
    wx = 2 * np.pi * np.fft.fftfreq(n, mc.DX)[None, :]
    wy = 2 * np.pi * np.fft.fftfreq(n, mc.DX)[:, None]
    k2 = wx ** 2 + wy ** 2
    k2[0, 0] = 1.0
    fx, fy = np.fft.fft2(j[0]), np.fft.fft2(j[1])
    # Smooth again and drop the Nyquist row and column, whose wavenumber sign is ambiguous between the
    # full and real FFT layouts; otherwise the projected current is not exactly divergence-free.
    taper = np.exp(-(k2 / (2 * np.pi * 2.0) ** 2))
    taper[n // 2, :] = 0
    taper[:, n // 2] = 0
    taper[0, 0] = 0          # no net current: the simulator drops the k = 0 term, field_at keeps it
    fx, fy = fx * taper, fy * taper
    long = (wx * fx + wy * fy) / k2
    return np.stack([np.real(np.fft.ifft2(fx - wx * long)), np.real(np.fft.ifft2(fy - wy * long))])


class Correction(unittest.TestCase):
    def test_exact_data_small_mu_recovers_truth(self):
        # Truth and input both come from one band-limited field; H applied on the same padded canvas.
        rng = np.random.default_rng(1)
        n, pad, dz = 64, 16, 0.3
        truth = rng.standard_normal((3, n, n))
        f = mc.padded_fft(truth, pad)
        _, _, k = mc.wavenumbers(f.shape[-1])
        f = f * (k < 3.0)                                   # keep H well above sqrt(mu)
        canvas = np.real(np.fft.ifft2(f))
        truth = canvas[:, pad:-pad, pad:-pad]
        blurred_full = np.real(np.fft.ifft2(f * np.exp(-k * dz)))
        # Use the full canvas as the "image" (pad 0) so nothing is lost to cropping.
        x0 = np.zeros_like(canvas)
        out = mc.tikhonov_correct(x0, blurred_full, dz, 1e-10, pad=0)
        np.testing.assert_allclose(out, canvas, atol=1e-6 * np.abs(canvas).max())
        self.assertEqual(truth.shape, (3, n, n))

    def test_zero_residual_leaves_prediction_unchanged(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((3, 40, 40))
        f = np.fft.fft2(x)
        _, _, k = mc.wavenumbers(40)
        b = np.real(np.fft.ifft2(f * np.exp(-k * 0.5)))
        np.testing.assert_allclose(mc.tikhonov_correct(x, b, 0.5, 0.03, pad=0), x, atol=1e-10)

    def test_mu_interpolation(self):
        s, m = [1.0, 3.0], [1e-2, 1e-4]
        self.assertAlmostEqual(mc.mu_at(2.0, s, m), 1e-3)
        self.assertAlmostEqual(mc.mu_at(0.5, s, m), 1e-2)
        self.assertAlmostEqual(mc.mu_at(9.0, s, m), 1e-4)


class Height(unittest.TestCase):
    def test_field_matches_simulator(self):
        # On a divergence-free current the simulator's projection does nothing, so all three agree.
        # The current is divergence-free on the periodic canvas, so both codes use the same canvas (pad 0).
        n = 160
        j = sheet(n)
        ours = mc.field_at(*mc.sheet_spectra(j, 0), 2.0, n, 0)
        ref = np.stack(fields_from_current(j[0], j[1], mc.DX, [2.0])[0])
        np.testing.assert_allclose(ours, ref, atol=1e-6 * np.abs(ref).max())

    def test_read_height_recovers_known_height(self):
        n, pad = 96, 64
        j = sheet(n)
        spectra, k, shape = mc.sheet_spectra(j, pad)
        for h_true in (1.37, 3.0, 6.42):
            b = mc.field_at(spectra, k, shape, h_true, n, pad)
            h, misfit, edge = mc.read_height(lambda hs: mc.height_misfit(j, b, hs, pad))
            self.assertAlmostEqual(h, h_true, places=2)
            self.assertLess(misfit, 1e-3)
            self.assertFalse(edge)


class Cutoff(unittest.TestCase):
    def test_consistent_field_has_no_inconsistent_part(self):
        # Field of a divergence-free sheet on the whole periodic canvas: no truncation, so the
        # inconsistent part must vanish. This checks the component frame and signs of the projection.
        n = 128
        j = sheet(n)
        b = mc.field_at(*mc.sheet_spectra(j, 0), 1.0, n, 0)
        lower, total, bad = mc.inconsistency_profile([b], pad=0, k_max=40)
        self.assertLess(bad.sum() / total.sum(), 1e-20)
        # Swapping Bx and By (a wrong frame) must look badly inconsistent.
        lower, total, bad = mc.inconsistency_profile([b[[1, 0, 2]]], pad=0, k_max=40)
        self.assertGreater(bad.sum() / total.sum(), 0.2)

    def test_cutoff_rule(self):
        lower = np.arange(0, 5, 0.1)
        total = np.ones_like(lower)
        bad = np.where(lower < 0.5, 0.9, np.where(lower < 2.25, 0.2, 0.8))
        self.assertAlmostEqual(mc.noise_cutoff(lower, total, bad), 2.3)


class Fourway(unittest.TestCase):
    def test_bands_add_up_to_raw(self):
        rng = np.random.default_rng(3)
        sharp = rng.standard_normal((3, 64, 64))
        blurred = rng.standard_normal((3, 64, 64))
        pred = sharp + 0.3 * rng.standard_normal((3, 64, 64))
        for row in mc.fourway(pred, sharp, blurred, 2.5, 2.0, pad=16):
            total = row['in_band'] ** 2 + row['prior_band'] ** 2 + row['noise_band'] ** 2
            self.assertAlmostEqual(np.sqrt(total), row['raw'], places=10)
            t = row['target_in'] ** 2 + row['target_prior'] ** 2 + row['target_noise'] ** 2
            self.assertAlmostEqual(t, 1.0, places=10)

    def test_crossing_matches_information_budget_rule(self):
        # Truth white, input = exact blur + small white noise: crossing where e^{-2 k dz} S = noise.
        rng = np.random.default_rng(4)
        n, pad, dz = 128, 32, 1.0
        sharp = rng.standard_normal((n, n))
        f = mc.padded_fft(sharp, pad)
        _, _, k = mc.wavenumbers(f.shape[-1])
        blurred = np.real(np.fft.ifft2(f * np.exp(-k * dz)))[pad:-pad, pad:-pad]
        blurred = blurred + 1e-2 * rng.standard_normal((n, n))
        kc = mc.snr_crossing(sharp, blurred, dz, pad)
        self.assertTrue(3.5 <= kc <= 5.5, kc)    # e^{-2k} = 1e-4 -> k = 4.6


if __name__ == '__main__':
    unittest.main()
