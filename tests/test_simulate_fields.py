import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import simulate_fields as sim


def square(side, centre=70.0):
    a, b = centre - side / 2, centre + side / 2
    return [(a, a), (b, a), (b, b), (a, b), (a, a)]


def canvas_fields(path, width, heights, n=1800, pad=400, dx=.1):
    kx, ky = sim.rasterize(path, width, 600e-6, (n, n), dx, pad)
    return sim.fields_from_current(kx, ky, dx, heights)


class Physics(unittest.TestCase):
    def test_square_loop_matches_closed_form_biot_savart(self):
        side, z = 30.0, 5.0
        field = canvas_fields(square(side), .3, [z])[0]
        centre = 400 + 700
        l, i = side * 1e-6, 600e-6
        z_m = z * 1e-6
        exact = (4e-7 * np.pi * i * l ** 2 / (2 * np.pi * (z_m ** 2 + l ** 2 / 4)
                 * np.sqrt(z_m ** 2 + l ** 2 / 2))) * 1e6
        self.assertAlmostEqual(abs(field[2][centre, centre]) / exact, 1, delta=.01)

    def test_long_strip_matches_closed_form(self):
        z, width = 1.0, 2.0
        # Wide loop: near the middle of a side this is one isolated strip.
        field = canvas_fields(square(120.0), width, [z])[0]
        got = field[0][400 + 100 - 200 + np.arange(400), 1100]
        y = (np.arange(400) - 200) * .1
        peak = 4e-7 * np.pi * 600e-6 / (2 * np.pi * width * 1e-6) * 1e6
        exact = peak * (np.arctan((y + width / 2) / z) - np.arctan((y - width / 2) / z))
        self.assertLess(np.abs(np.abs(got) - exact).max() / exact.max(), .02)

    def test_fields_satisfy_gauss_and_curl_in_export_frame(self):
        path = sim.transform(square(40), 1, False)
        f = canvas_fields(path, 2.0, [3.0], n=1800)[0]
        n = 1800
        w = 2 * np.pi * np.fft.fftfreq(n, .1)
        wx, wy = w[None, :], w[:, None]
        k = np.hypot(wx, wy)
        k[0, 0] = 1
        x, y, z = [np.fft.fft2(a) for a in f]
        got = 1j * (-wx * y + wy * x) / k
        self.assertLess(np.linalg.norm(got - z) / np.linalg.norm(z), 1e-8)

    def test_input_is_exact_upward_continuation_of_target(self):
        low, high = canvas_fields(square(20, 20), 2.0, [.5, 3.0], n=1200, pad=400)
        w = 2 * np.pi * np.fft.fftfreq(1200, .1)
        h = np.exp(-np.hypot(w[None, :], w[:, None]) * 2.5)
        for a, b in zip(low, high):
            up = np.fft.ifft2(np.fft.fft2(a) * h).real
            self.assertLess(np.linalg.norm(up - b) / np.linalg.norm(b), 1e-8)


    def test_zero_k0_is_free_space_and_k0_only_adds_decay(self):
        path = square(20, 20)
        kx, ky = sim.rasterize(path, 2.0, 600e-6, (1200, 1200), .1, 400)
        free = sim.fields_from_current(kx, ky, .1, [3.0])[0]
        same = sim.fields_from_current(kx, ky, .1, [3.0], 0.0, .5)[0]
        low = sim.fields_from_current(kx, ky, .1, [3.0], .095, .5)[0]
        at_ref = sim.fields_from_current(kx, ky, .1, [.5], .095, .5)[0]
        plain = sim.fields_from_current(kx, ky, .1, [.5])[0]
        for i in range(3):
            self.assertTrue(np.allclose(free[i], same[i]))
            self.assertTrue(np.allclose(at_ref[i], plain[i]))
            self.assertLess(np.linalg.norm(low[i]), np.linalg.norm(free[i]))
        with self.assertRaises(ValueError):
            sim.fields_from_current(kx, ky, .1, [.2], .095, .5)

    def test_speckle_matches_requested_correlation(self):
        rng = np.random.default_rng(0)
        field = sim.correlated_noise(rng.standard_normal((800, 800)), .38, .1)
        self.assertAlmostEqual(field.std(), 1, delta=1e-9)
        power = np.abs(np.fft.fft2(field)) ** 2
        corr = np.fft.ifft2(power).real
        corr /= corr[0, 0]
        for lag, measured in ((1, .76), (2, .60), (5, .27)):
            self.assertAlmostEqual(corr[0, lag], measured, delta=.08)


class Layouts(unittest.TestCase):
    def test_recipes_are_closed_clear_and_repeatable(self):
        families = set()
        for i in range(40):
            r = sim.draw_recipe(0, f's{i}', sim.DEFAULTS)
            self.assertEqual(r, sim.draw_recipe(0, f's{i}', sim.DEFAULTS))
            self.assertTrue(sim.paths_clear(r['paths_um'],
                                            r['width_um'] + sim.MIN_GAP_UM))
            families.add(r['family'])
            self.assertEqual(len(r['paths_um']) > 1, r['family'] == 'array')
            for p in map(np.array, r['paths_um']):
                self.assertTrue(np.array_equal(p[0], p[-1]))
                inside = (p > 0) & (p < sim.IMAGE_UM)
                self.assertTrue(inside.all(1).any())
        self.assertEqual(families, {'spiral', 'meander', 'array'})

    def test_array_copies_mirror_the_first_coil(self):
        paths = sim.array_paths(np.random.default_rng(1), 2.0)
        self.assertGreaterEqual(len(paths), 2)
        flipped = paths[0].copy()
        flipped[:, 1] = sim.IMAGE_UM - flipped[:, 1]
        self.assertTrue(np.allclose(flipped, paths[1]))

    def test_overlapping_paths_are_rejected(self):
        self.assertFalse(sim.paths_clear([square(20), square(20.5)], 1.0))
        self.assertTrue(sim.paths_clear([square(20), square(40)], 1.0))

    def test_collinear_joints_do_not_double_the_current(self):
        joined = [(10, 10), (30, 10), (60, 10), (60, 40), (10, 40), (10, 10)]
        merged = sim.merge_collinear(joined)
        self.assertEqual(len(merged), 5)
        kx, _ = sim.rasterize(merged, 2.0, 600e-6, (1800, 1800), .1, 400)
        self.assertLessEqual(np.abs(kx).max(), 600e-6 / 2e-6 * (1 + 1e-12))

    def test_reversal_is_rejected(self):
        self.assertFalse(sim.segments_clear([(0, 0), (10, 0), (5, 0), (5, 9)], 1.0))

    def test_recipes_have_no_collinear_joints(self):
        for i in range(40):
            for p in sim.draw_recipe(0, f's{i}', sim.DEFAULTS)['paths_um']:
                self.assertEqual(len(sim.merge_collinear(p)), len(p))

    def test_crossing_path_is_rejected(self):
        self.assertFalse(sim.segments_clear(
            [(0, 0), (10, 0), (10, 10), (5, 10), (5, -5)], 1.0))


class Dataset(unittest.TestCase):
    def test_noise_touches_input_only_and_data_regenerates(self):
        settings = dict(sim.DEFAULTS, noise_percent=(1.5,))
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / 'set'
            sim.build(out, 2, 1, 0, settings, 1)
            d = sim.SimulatedDataset(out)
            self.assertGreater(d.scale, 0)
            recipe = d.recipes('train')[0]
            a, b = sim.synthesise(recipe, settings), sim.synthesise(recipe, settings)
            self.assertTrue(np.array_equal(a['input'], b['input']))
            noise = (a['input'] - a['clean']).astype(np.float64)
            for i in range(3):
                span = np.ptp(a['clean'][i])
                self.assertAlmostEqual(noise[i].std() / span, .015, delta=.001)
            self.assertLess(abs(np.corrcoef(noise[0].ravel(), noise[1].ravel())[0, 1]), .05)
            self.assertEqual(len(list(d.iter_components('validation'))), 3)
            with self.assertRaises(FileExistsError):
                sim.build(out, 1, 0, 0, settings, 1)

    def test_version_1_recipe_still_regenerates(self):
        old = {k: v for k, v in sim.DEFAULTS.items()
               if k not in ('decay_k0', 'decay_ref_um', 'noise_corr_um')}
        recipe = dict(id='old', family='spiral', width_um=2.0, offset_um=.1,
                      noise_percent=1.0, noise_seed=5,
                      path_um=[list(p) for p in square(30)])
        item = sim.synthesise(recipe, old)
        direct = canvas_fields(recipe['path_um'], 2.0, [3.1], n=2800, pad=700)[0]
        self.assertTrue(np.allclose(item['clean'][2], direct[2][700:-700, 700:-700],
                                    atol=1e-4))


if __name__ == '__main__':
    unittest.main()
