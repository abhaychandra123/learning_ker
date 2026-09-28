"""Numerical and data-alignment checks for the direct crop experiment."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from kernel_crops import (make_crops, assemble, fit_kernel, apply_kernel,
                          split_ids, height_holdout, load_plane, scores)


class CropKernelTests(unittest.TestCase):
    def test_crop_coordinates_and_reassembly(self):
        a = np.arange(3*12*16).reshape(3, 12, 16)
        crops = make_crops(a, 4)
        self.assertEqual(crops.shape, (12, 3, 4, 4))
        np.testing.assert_array_equal(crops[6], a[:, 4:8, 8:12])
        np.testing.assert_array_equal(assemble(crops, (12, 16)), a)
        with self.assertRaises(ValueError):
            make_crops(a, 5)

    def test_recovers_known_circular_kernel_and_unseen_crops(self):
        rng = np.random.default_rng(2026)
        sharp = rng.normal(size=(8, 3, 9, 9))
        impulse = np.zeros((9, 9))
        impulse[0, 0], impulse[0, 1], impulse[1, 0] = .6, .25, .15
        true = np.fft.fft2(impulse)
        blurred = apply_kernel(sharp, true)
        learned, _ = fit_kernel(sharp, blurred, [0, 1, 2, 3])
        np.testing.assert_allclose(learned, true, atol=1e-14)
        np.testing.assert_allclose(apply_kernel(sharp[4:], learned), blurred[4:], atol=1e-14)

    def test_joint_estimator_matches_dense_least_squares(self):
        rng = np.random.default_rng(1)
        sharp = rng.normal(size=(2, 3, 3, 3))
        blurred = rng.normal(size=sharp.shape)
        # Explicit circular convolution design, independent of FFT fitting.
        design = np.column_stack([np.roll(sharp, (y, x), axis=(-2, -1)).ravel()
                                  for y in range(3) for x in range(3)])
        expected = np.linalg.lstsq(design, blurred.ravel(), rcond=None)[0].reshape(3, 3)
        learned, _ = fit_kernel(sharp, blurred)
        np.testing.assert_allclose(np.fft.ifft2(learned).real, expected, atol=1e-14)

    def test_heldout_targets_do_not_affect_fit(self):
        rng = np.random.default_rng(7)
        sharp = rng.normal(size=(25, 3, 4, 4))
        blurred = rng.normal(size=sharp.shape)
        train, test = split_ids(5, 5)
        self.assertEqual(len(train), 20)
        self.assertEqual(len(test), 5)
        self.assertFalse(set(train) & set(test))
        before, _ = fit_kernel(sharp, blurred, train)
        blurred[test] = 1e9
        after, _ = fit_kernel(sharp, blurred, train)
        np.testing.assert_array_equal(before, after)

    def test_height_composition_predicts_unseen_plane(self):
        rng = np.random.default_rng(8)
        reference = rng.normal(size=(4, 3, 8, 8))
        ky, kx = np.meshgrid(np.fft.fftfreq(8), np.fft.fftfreq(8), indexing="ij")
        unit = np.exp(-np.hypot(kx, ky))
        plane3 = apply_kernel(reference, unit**2.5)
        plane4 = apply_kernel(reference, unit**3.5)
        expected5 = apply_kernel(reference, unit**4.5)
        prediction, _, _, _, protocol = height_holdout(reference, plane3, plane4, 1e-12)
        np.testing.assert_allclose(prediction, expected5, atol=1e-14)
        self.assertEqual(protocol["fit_pairs_um"], [[.5, 3.], [3., 4.]])

    def test_zero_signal_and_metric_denominators(self):
        a = np.zeros((2, 3, 4, 4))
        h, info = fit_kernel(a, a)
        self.assertTrue(np.isfinite(h).all())
        self.assertEqual(info["supported_fraction"], 0)
        self.assertIsNone(scores(a, a)["nrmse_range"])

    def test_csv_loader_validates_every_coordinate_and_units(self):
        header = ["% Model,interconnects_F.mph"] + ["% metadata"]*6 + ["% Length unit,µm",
                  "% x,y,mf.Bx (T),mf.By (T),mf.Bz (T),mf.normB (T)"]
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/"plane.csv"
            a = np.array([[x*.1, y*.1, 1e-6, 2e-6, 3e-6, 4e-6] for y in range(3) for x in range(4)])
            def save():
                with p.open("w") as f:
                    f.write("\n".join(header)+"\n")
                    np.savetxt(f, a, delimiter=",")
            save()
            fields, grid = load_plane(p)
            self.assertEqual((grid["ny"], grid["nx"]), (3, 4))
            np.testing.assert_allclose(fields[:, 0, 0], [1, 2, 3])
            a[7, 0] += .03
            save()
            with self.assertRaises(ValueError):
                load_plane(p)


if __name__ == "__main__":
    unittest.main()
