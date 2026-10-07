"""Check the separate forward-height holdout used in FISTA reports."""
from pathlib import Path
import sys
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from deblur import forward_operator
from fista_report import predict_holdout


class FistaReportTests(unittest.TestCase):
    def test_unseen_height_matches_known_semigroup(self):
        rng = np.random.default_rng(5)
        sharps = rng.normal(size=(3, 12, 14))
        k = np.hypot(np.fft.fftfreq(12)[:, None], np.fft.fftfreq(14)[None, :])
        h = np.exp(-k)
        plane1 = [forward_operator(s, h, 0) for s in sharps]
        fitted, predictions = predict_holdout(sharps, plane1, 0)
        np.testing.assert_allclose(fitted, h, atol=1e-12)
        for sharp, prediction in zip(sharps, predictions):
            np.testing.assert_allclose(prediction, forward_operator(sharp, h**3, 0), atol=1e-12)

    def test_padded_holdout_crops_between_steps(self):
        rng = np.random.default_rng(9)
        sharps = rng.normal(size=(3, 8, 10))
        plane1 = rng.normal(size=(3, 8, 10))
        fitted, predictions = predict_holdout(sharps, plane1, 2)
        for sharp, prediction in zip(sharps, predictions):
            expected = sharp
            for _ in range(3):
                expected = forward_operator(expected, fitted, 2)
            np.testing.assert_array_equal(prediction, expected)
            self.assertEqual(prediction.shape, sharp.shape)


if __name__ == '__main__':
    unittest.main()
