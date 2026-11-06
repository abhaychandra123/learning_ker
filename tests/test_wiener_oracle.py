import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # torch lives on the server only
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import wiener_oracle as wo


class Filters(unittest.TestCase):
    def test_cross_and_model_filters_match_textbook_wiener(self):
        # B = H X + n with white X (S_x = 1) and white n (S_n = s^2), periodic: no padding.
        # H falls about 9% per frequency ring, as on the real 1656 px canvas at dz 2.5 um.
        rng, n, s = np.random.default_rng(0), 128, 0.05
        rings = wo.Rings(n, dx=1.0)
        h = wo.transfer(rings.f, 2.0)
        items = []
        for _ in range(300):
            x = rng.standard_normal((n, n))
            X = np.fft.rfft2(x)
            B = X * h + np.fft.rfft2(s * rng.standard_normal((n, n)))
            items.append(wo.Item('a', 'a', 0, x, X, B, 1.0))
        g = wo.filters(wo.moments(items, h, rings), h, rings)
        truth = h / (h ** 2 + s ** 2)
        keep = (rings.bins >= 4) & (rings.f < 0.4)  # rings with enough coefficients
        for name in ('cross', 'model'):
            np.testing.assert_allclose(g[name][keep], truth[keep], rtol=0.12, atol=0.05)

    def test_log_interpolation_is_exact_for_exponential_trend(self):
        ring = np.array([1.0, 2.0, 3.0])
        spectra = [np.exp(-ring * z) for z in (2.0, 4.0, 6.0)]
        np.testing.assert_allclose(wo.log_interp(5.0, [2.0, 4.0, 6.0], spectra), np.exp(-ring * 5.0))
        np.testing.assert_allclose(wo.log_interp(9.0, [6.0, 2.0, 4.0], [spectra[2], spectra[0], spectra[1]]),
                                   spectra[2])  # clamps beyond the measured heights

    @unittest.skipIf(torch is None, 'needs torch')
    def test_tikhonov_reproduces_existing_baseline(self):
        import csc_lista as csc
        rng = np.random.default_rng(1)
        b = rng.standard_normal((40, 40))
        pad = 8
        f, _ = wo.grid(40 + 2 * pad)
        h = wo.transfer(f, 2.5)
        mine = wo.output(h / (h * h + wo.MU), wo.spectrum(b, pad), pad)
        layer = csc.Dataless(wo.DX, 2.5, wo.MU)
        padded = torch.nn.functional.pad(torch.tensor(b)[None], (pad,) * 4)
        theirs = layer(padded)[0, pad:-pad, pad:-pad].numpy()
        np.testing.assert_allclose(mine, theirs, atol=1e-10)


class Pairs(unittest.TestCase):
    def test_discovers_height_pairs_from_file_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'comparison'
            folder.mkdir()
            for name in ('L1_C1_z03.0_to_z00.5.npz', 'IC_2um_z02.0_to_z01.0.npz', 'notes.npz'):
                (folder / name).touch()
            pairs = wo.discover(tmp, None)
        found = {(p.name, p.shape, p.z_in, p.z_tg) for p in pairs}
        self.assertEqual(found, {('L1_C1', 'C1', 3.0, 0.5), ('IC_2um', 'IC', 2.0, 1.0)})

    def test_manifest_reads_npz_and_keeps_every_height(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            field = np.ones((3, 2, 2))
            np.savez(tmp / 'a.npz', sharp=field, blurred=2 * field)
            with (tmp / 'pairs.csv').open('w', newline='') as stream:
                w = csv.writer(stream)
                w.writerow(['name', 'shape', 'z_input', 'z_target', 'sharp', 'blurred'])
                w.writerow(['A', 'S1', 3.0, 0.5, 'a.npz', 'a.npz'])
                w.writerow(['A', 'S1', 4.0, 0.5, 'a.npz', 'a.npz'])
            pairs = wo.manifest(tmp / 'pairs.csv')
            np.testing.assert_array_equal(wo.load(pairs[0].blurred), 2 * field)
        self.assertEqual([p.group for p in pairs], [(3.0, 0.5), (4.0, 0.5)])
        self.assertEqual(pairs[0].shape, 'S1')


if __name__ == '__main__':
    unittest.main()
