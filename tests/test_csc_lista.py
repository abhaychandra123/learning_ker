import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import csc_lista as csc
import train_csc_lista as trainer


class Operators(unittest.TestCase):
    def test_forward_and_adjoint_are_transposes(self):
        torch.manual_seed(0)
        for chunk in (None, 3):
            op = csc.Operator(csc.unit_norm(torch.randn(4, 7, 7)), chunk=chunk)
            x, r = torch.randn(2, 4, 32, 32), torch.randn(2, 32, 32)
            ratio = (op.forward(x) * r).sum() / (x * op.adjoint(r)).sum()
            self.assertAlmostEqual(float(ratio), 1, places=4)

    def test_chunks_do_not_change_results(self):
        torch.manual_seed(1)
        filters, x = csc.unit_norm(torch.randn(5, 7, 7)), torch.randn(1, 5, 24, 24)
        whole, parts = csc.Operator(filters), csc.Operator(filters, chunk=2)
        self.assertTrue(torch.allclose(whole.forward(x), parts.forward(x), atol=1e-5))
        r = torch.randn(1, 24, 24)
        self.assertTrue(torch.allclose(whole.adjoint(r), parts.adjoint(r), atol=1e-5))
        self.assertAlmostEqual(whole.bound(24), parts.bound(24), places=4)

    def test_centred_filter_keeps_position(self):
        impulse = torch.zeros(1, 5, 5)
        impulse[0, 2, 2] = 1
        x = torch.zeros(1, 1, 16, 16)
        x[0, 0, 6, 9] = 1
        y = csc.Operator(impulse).forward(x)
        self.assertAlmostEqual(float(y[0, 6, 9]), 1, places=5)

    def test_bound_is_the_operator_norm(self):
        torch.manual_seed(1)
        op = csc.Operator(csc.unit_norm(torch.randn(3, 5, 5)))
        x = torch.randn(1, 3, 24, 24)
        for _ in range(100):
            x = op.adjoint(op.forward(x))
            x = x / x.norm()
        value = float(op.forward(x).norm() ** 2)
        self.assertLessEqual(value, op.bound(24) * 1.0001)
        self.assertGreater(value, op.bound(24) * .95)

    def test_transfer_matches_composition(self):
        a, b = csc.transfer(32, .5, 1.0), csc.transfer(32, .5, 1.5)
        self.assertTrue(torch.allclose(a * b, csc.transfer(32, .5, 2.5), atol=1e-5))


class Model(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.n, self.pad = 48, 8
        self.atoms = csc.unit_norm(torch.randn(4, 7, 7))

    def build(self, **options):
        return csc.CSCLISTA(self.atoms, .5, 1.0, self.n, self.pad, .01, **options)

    def test_full_width_start_matches_ista(self):
        model = self.build(layers=6, width=self.n - 1, use_checkpoint=False)
        operator, _ = model.physics(self.n)
        b = operator.forward(torch.randn(2, 4, self.n, self.n) * .1)
        got = model.codes(b).detach()
        want = csc.code_ista(b, operator, .01, 6, self.pad, fast=False)
        self.assertLess(float((got - want).norm() / want.norm()), .02)

    def test_full_width_momentum_start_matches_fista(self):
        model = self.build(layers=6, width=self.n - 1, use_checkpoint=False, momentum=True)
        operator, _ = model.physics(self.n)
        b = operator.forward(torch.randn(2, 4, self.n, self.n) * .1)
        got = model.codes(b).detach()
        want = csc.code_ista(b, operator, .01, 6, self.pad, fast=True)
        self.assertLess(float((got - want).norm() / want.norm()), .02)
        self.assertEqual(float(model.beta[0]), 0)

    def test_lasso_objective_matches_its_definition(self):
        model = self.build(layers=2, width=9, use_checkpoint=False)
        b = torch.randn(2, self.n, self.n)
        z = model.codes(b)
        operator, mask = model.physics(self.n)
        r = mask * (operator.forward(z) - b)
        want = (.5 * (r ** 2).sum() + .03 * z.abs().sum()) / 2 / self.n ** 2
        self.assertAlmostEqual(float(model.lasso_objective(b, .03)), float(want), places=4)

    def test_momentum_is_trained(self):
        model = self.build(layers=3, width=9, momentum=True)
        loss = (model(torch.randn(2, self.n, self.n)) ** 2).mean()
        loss.backward()
        self.assertTrue(model.beta.grad[:2].abs().sum() > 0)

    def test_training_lowers_loss_and_reaches_every_layer(self):
        model = self.build(layers=3, width=9)
        operator, _ = model.physics(self.n)
        codes = torch.nn.functional.relu(torch.randn(4, 4, self.n, self.n) - 1.5)
        target = csc.Operator(self.atoms).forward(codes)
        b = operator.forward(codes)
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        first = None
        for _ in range(40):
            loss = ((model(b) - target) ** 2).mean()
            first = first or float(loss)
            optimiser.zero_grad()
            loss.backward()
            self.assertTrue((model.w.grad.flatten(1).abs().sum(1) > 0).all())
            optimiser.step()
        self.assertLess(float(loss), first)

    def test_residual_ignores_the_padding(self):
        model = self.build(layers=2, width=9, use_checkpoint=False)
        b = torch.randn(1, self.n, self.n)
        changed = b.clone()
        changed[:, :self.pad] = 5
        self.assertTrue(torch.equal(model(b), model(changed)))

    def test_model_runs_on_a_larger_canvas(self):
        model = self.build(layers=2, width=9, use_checkpoint=False)
        self.assertEqual(model(torch.randn(1, 96, 96)).shape, (1, 96, 96))

    def test_dictionary_learning_fits_sparse_fields(self):
        spectrum = csc.Operator(csc.unit_norm(torch.randn(3, 7, 7)))
        codes = (torch.rand(32, 3, 64, 64) > .995).float() * torch.randn(32, 3, 64, 64)
        fields = spectrum.forward(codes)
        g, history = csc.learn_dictionary(fields, 6, 7, .01, 150, 8, 48, lr=.01,
                                         log=lambda _: None)
        self.assertTrue(torch.allclose(g.norm(dim=(1, 2)), torch.ones(6), atol=1e-4))
        self.assertLess(history[-1]['relative_l2'], history[0]['relative_l2'])


class Loss(unittest.TestCase):
    def test_relative_loss_matches_validation_metric_and_ignores_scale(self):
        target, output = torch.randn(3, 8, 8), torch.randn(3, 8, 8)
        loss = trainer.crop_loss(output, target, 'relative')
        self.assertAlmostEqual(float(loss), float(trainer.relative_l2(output, target).mean()))
        scaled = trainer.crop_loss(output * 7, target * 7, 'relative')
        self.assertAlmostEqual(float(loss), float(scaled), places=4)

    def test_mse_is_the_default_form(self):
        target, output = torch.randn(3, 8, 8), torch.randn(3, 8, 8)
        self.assertAlmostEqual(float(trainer.crop_loss(output, target, 'mse')),
                               float(((output - target) ** 2).mean()))



class Stream(unittest.TestCase):
    # A coarse 0.4 um grid keeps the generator fast; the code path is the same.
    SETTINGS = dict(nx=350, ny=350, dx=0.4, pad=175, z_input=3.0, z_target=0.5, current_a=600e-6,
                    width_um=(1.6, 3.2), offset_um=(0.0, 0.2), noise_percent=(1.0, 2.5),
                    noise_corr_um=0.38, decay_k0=0.0, decay_ref_um=0.5, psf_sigma_um=0.0)

    def crops(self):
        return trainer.FreshCrops(self.SETTINGS, 0, 100.0, 64)

    def test_crops_are_reproducible_and_shaped(self):
        a, b = self.crops().crop(3), self.crops().crop(3)
        self.assertEqual(a[0].shape, (64, 64))
        self.assertTrue(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))

    def test_every_crop_has_its_own_layout(self):
        first, second = self.crops().crop(0), self.crops().crop(1)
        self.assertFalse(torch.equal(first[1], second[1]))

    def test_stream_never_reuses_stored_layouts(self):
        from simulate_fields import draw_recipe
        fresh = draw_recipe(0, 'stream_00000000', self.SETTINGS)['paths_um']
        for label in ('train_00000', 'validation_00000'):
            self.assertNotEqual(fresh, draw_recipe(0, label, self.SETTINGS)['paths_um'])

    def test_workers_split_the_stream(self):
        loader = torch.utils.data.DataLoader(self.crops(), batch_size=2, num_workers=2)
        it = iter(loader)
        first, second = next(it), next(it)
        self.assertFalse(torch.equal(first[1], second[1]))


if __name__ == '__main__':
    unittest.main()
