"""Numerical checks against dense operators and solvable L1 problems."""
import contextlib
import io
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from deblur import forward_operator, adjoint_operator, deblur_ista
from iterative import solve_l1


class IterativeTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(37)

    def test_padded_adjoint_matches_dense_transpose(self):
        for shape, pad in [((3, 4), 0), ((3, 4), 2), ((4, 5), 1)]:
            h = np.fft.fft2(self.rng.normal(size=tuple(n+2*pad for n in shape)))
            eye = np.eye(np.prod(shape))
            dense = np.column_stack([forward_operator(e.reshape(shape), h, pad).ravel()
                                     for e in eye])
            r = self.rng.normal(size=shape)
            np.testing.assert_allclose(adjoint_operator(r, h, pad).ravel(),
                                       dense.T @ r.ravel(), atol=1e-12)
            bound = np.max(abs(h)**2)
            self.assertLessEqual(np.linalg.norm(dense, 2)**2, bound * (1+1e-12))

    def test_ista_matches_existing_implementation(self):
        shape, pad = (7, 8), 2
        h = np.fft.fft2(self.rng.normal(size=(11, 12)))
        b = self.rng.normal(size=shape)
        with contextlib.redirect_stdout(io.StringIO()):
            expected, _ = deblur_ista(b, h, .03, n_iter=13, pad=pad)
        actual, history = solve_l1(b, h, .03, method="ista", n_iter=13, pad=pad)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual([r['iteration'] for r in history], [0, 10, 13])

    def test_identity_lasso_has_exact_soft_threshold_solution(self):
        shape, pad = (7, 6), 2
        b = self.rng.normal(size=shape)
        h = np.ones((11, 10), dtype=complex)
        lam = .2
        optimum = np.sign(b)*np.maximum(abs(b)-lam, 0)
        for method in ("ista", "fista"):
            x, history = solve_l1(b, h, lam, method=method, n_iter=100, pad=pad,
                                  initial="zero", record_every=10)
            np.testing.assert_allclose(x, optimum, atol=1e-10)
            self.assertLess(history[-1]['pg_rms'], 1e-10)

    def test_fista_iterates_match_dense_reference(self):
        shape, pad, lam = (4, 5), 1, .2
        h = np.fft.fft2(self.rng.normal(size=(6, 7)))
        dense = np.column_stack([forward_operator(e.reshape(shape), h, pad).ravel()
                                 for e in np.eye(20)])
        b = self.rng.normal(size=shape)
        x = b.ravel().copy()
        y, t = x.copy(), 1.
        step = .99/np.max(abs(h)**2)
        for _ in range(17):
            v = y - step*(dense.T @ (dense @ y-b.ravel()))
            next_x = np.sign(v)*np.maximum(abs(v)-step*lam, 0)
            next_t = (1+np.sqrt(1+4*t*t))/2
            y = next_x + (t-1)/next_t*(next_x-x)
            x, t = next_x, next_t
        actual, _ = solve_l1(b, h, lam, method="fista", n_iter=17, pad=pad)
        np.testing.assert_allclose(actual.ravel(), x, rtol=1e-12, atol=1e-12)

    def test_fista_improves_objective_on_ill_conditioned_quadratic(self):
        # In Fourier coordinates this lambda=0 problem has a known solution.
        shape = (16, 18)
        fy = np.fft.fftfreq(shape[0])[:, None]
        fx = np.fft.fftfreq(shape[1])[None, :]
        h = np.exp(-8*np.hypot(fx, fy))
        truth = self.rng.normal(size=shape)
        b = forward_operator(truth, h, 0)
        _, ih = solve_l1(b, h, 0, method="ista", n_iter=40, pad=0, initial="zero")
        _, fh = solve_l1(b, h, 0, method="fista", n_iter=40, pad=0, initial="zero")
        self.assertLess(fh[-1]['objective'], ih[-1]['objective'])

    def test_rejects_invalid_inputs_and_handles_zero_iterations(self):
        b, h = np.ones((3, 4)), np.ones((3, 4))
        for changes in [dict(lam=-1), dict(n_iter=-1), dict(pad=1),
                        dict(record_every=0), dict(lam=float('nan'))]:
            kwargs = dict(lam=.1, n_iter=2, pad=0, method="fista")
            kwargs.update(changes)
            with self.assertRaises(ValueError):
                solve_l1(b, h, **kwargs)
        x, rows = solve_l1(b, h, .1, method="fista", n_iter=0, pad=0)
        np.testing.assert_array_equal(x, b)
        self.assertEqual(len(rows), 1)


if __name__ == '__main__':
    unittest.main()
