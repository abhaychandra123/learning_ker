import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import csc_lista as csc
import fista_net as fn


class Paper(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = fn.FISTANet(.5, 1.0, use_checkpoint=False)

    def test_parameter_count_matches_the_paper_tables(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 74599)

    def test_schedules_follow_equation_15(self):
        mu, theta, rho = self.model.schedules(1)
        sp = torch.nn.functional.softplus
        self.assertAlmostEqual(float(mu), float(sp(torch.tensor(-2.5))), places=5)
        self.assertAlmostEqual(float(theta), float(sp(torch.tensor(-1.2))), places=5)
        self.assertEqual(float(rho), 0)
        self.assertTrue(0 < float(self.model.schedules(7)[2]) < 1)

    def test_code_start_matches_the_authors_schedules(self):
        model = fn.FISTANet(.5, 1.0, use_checkpoint=False, start='code')
        sp = torch.nn.functional.softplus
        for i in range(7):
            mu, theta, rho = model.schedules(i + 1)
            top = sp(torch.tensor(.5 * i))
            self.assertAlmostEqual(float(mu), float(sp(torch.tensor(-.2 * i + .1))), places=5)
            self.assertAlmostEqual(float(theta), float(sp(torch.tensor(-.5 * i - 2))), places=5)
            self.assertAlmostEqual(float(rho), float((top - sp(torch.tensor(0.))) / top), places=5)

    def dense(self, n):
        """Literal matrices on a small periodic grid."""
        h = csc.transfer(n, .5, 1.0, torch.float64)
        size = n * n

        def apply(spectrum, v):
            return torch.fft.irfft2(torch.fft.rfft2(v.reshape(n, n)) * spectrum, s=(n, n)).ravel()

        a = torch.stack([apply(h, e) for e in torch.eye(size, dtype=torch.float64)], 1)
        i = torch.arange(size).reshape(n, n)
        rows = torch.zeros(size, size, dtype=torch.float64)
        cols = torch.zeros(size, size, dtype=torch.float64)
        for p, q in zip(i.ravel(), torch.roll(i, 1, 0).ravel()):
            rows[p, p], rows[p, q] = 1, -1
        for p, q in zip(i.ravel(), torch.roll(i, 1, 1).ravel()):
            cols[p, p], cols[p, q] = 1, -1
        return a, torch.cat((rows, cols))

    def test_fourier_step_equals_the_authors_matrix_formula(self):
        n = 8
        a, l = self.dense(n)
        ata, ltl = a.T @ a, l.T @ l
        b = torch.randn(n * n, dtype=torch.float64)
        y = torch.randn(n * n, dtype=torch.float64)
        system = ata + .001 * ltl
        want = torch.linalg.solve(system, ata @ y - a.T @ b - .001 * ltl @ y)
        h, smooth = self.model.spectra(n, torch.zeros(1, dtype=torch.float64))
        y_hat, b_hat = torch.fft.rfft2(y.reshape(n, n)), torch.fft.rfft2(b.reshape(n, n))
        got = torch.fft.irfft2((h * h * y_hat - h * b_hat - smooth * y_hat) / (h * h + smooth),
                               s=(n, n)).ravel()
        self.assertLess(float((got - want).norm() / want.norm()), 1e-6)

    def test_initial_image_equals_equation_16(self):
        n = 8
        a, l = self.dense(n)
        b = torch.randn(n * n, dtype=torch.float64)
        want = torch.linalg.solve(a.T @ a + .001 * l.T @ l, a.T @ b)
        got = self.model.initial(b.reshape(1, n, n)).ravel()
        self.assertLess(float((got - want).norm() / want.norm()), 1e-6)

    def test_output_can_be_negative_and_one_block_is_shared(self):
        field, symmetry, sparsity = self.model(-torch.ones(1, 32, 32))
        self.assertLess(float(field.min()), 0)
        self.assertGreater(float(symmetry), 0)
        self.assertGreater(float(sparsity), 0)
        self.assertEqual(sum(isinstance(m, fn.ProximalBlock) for m in self.model.modules()), 1)

    def test_loss_weights(self):
        output, target = torch.ones(2, 4, 4), torch.zeros(2, 4, 4)
        self.assertAlmostEqual(float(fn.total_loss(output, target, torch.tensor(2.), torch.tensor(3.))),
                               1 + .02 + .003, places=5)

    def test_training_reaches_all_parameters(self):
        loss = fn.total_loss(*self.pair())
        loss.backward()
        for name, p in self.model.named_parameters():
            self.assertIsNotNone(p.grad, name)

    def pair(self):
        field, symmetry, sparsity = self.model(torch.randn(2, 32, 32))
        return field, torch.zeros_like(field), symmetry, sparsity


if __name__ == '__main__':
    unittest.main()
