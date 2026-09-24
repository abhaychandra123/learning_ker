"""Operator, autograd, initialization, learning and checkpoint checks."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from deblur import forward_operator, adjoint_operator
from iterative import solve_l1
from torch_inverse import PaddedFFT, solve_torch
from lista import ConvLISTA, load_checkpoint
from compare_lista import train_model, cpu_state


class LISTATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)
        self.rng = np.random.default_rng(4)
        self.shape, self.pad = (6, 7), 2
        self.h = np.fft.fft2(self.rng.normal(size=(10, 11))) / 10
        self.b = torch.tensor(self.rng.normal(size=(3, 1, 6, 7)), dtype=torch.float64)

    def test_torch_operator_and_autograd_match_numpy(self):
        op = PaddedFFT(self.h, self.shape, self.pad, torch.float64)
        x = self.b.clone().requires_grad_()
        residual = op(x)
        np.testing.assert_allclose(residual[0, 0].detach(),
                                   forward_operator(x[0, 0].detach().numpy(), self.h, self.pad), atol=1e-12)
        np.testing.assert_allclose(op.adjoint(self.b)[0, 0],
                                   adjoint_operator(self.b[0, 0].numpy(), self.h, self.pad), atol=1e-12)
        grad, = torch.autograd.grad(.5*residual.square().sum(), (x,))
        torch.testing.assert_close(grad, op.adjoint(op(x)), rtol=1e-12, atol=1e-12)

    def test_lista_initialization_equals_ista_for_all_layers(self):
        for initial in ('zero', 'blurred'):
            model = ConvLISTA(self.h, self.shape, self.pad, .1, layers=4,
                              kernel_size=3, initial=initial, dtype=torch.float64)
            for k, x in enumerate(model.iterates(self.b)):
                expected, _ = solve_l1(self.b[0, 0].numpy(), self.h, .1, method='ista',
                                       n_iter=k, pad=self.pad, initial=initial)
                np.testing.assert_allclose(x[0, 0].detach(), expected, atol=1e-12, rtol=1e-12)

    def test_fista_torch_matches_numpy_and_unit_scaling(self):
        scale = 37.
        op = PaddedFFT(self.h, self.shape, self.pad, torch.float64)
        x, rows, _ = solve_torch(self.b/scale, op, .1/scale, method='fista', n_iter=11,
                                 initial='blurred', truth=self.b/scale, scale=scale)
        expected, nr = solve_l1(self.b[0, 0].numpy(), self.h, .1, method='fista',
                                n_iter=11, pad=self.pad, truth=self.b[0, 0].numpy())
        np.testing.assert_allclose(x[0, 0]*scale, expected, atol=1e-12, rtol=1e-12)
        self.assertAlmostEqual(rows[-1]['components'][0]['objective'], nr[-1]['objective'], places=10)

    def test_weight_and_threshold_gradients_against_finite_differences(self):
        model = ConvLISTA(self.h, self.shape, self.pad, .1, layers=3,
                          kernel_size=3, dtype=torch.float64)
        target = self.b*.7
        loss = (model(self.b)-target).square().mean()
        loss.backward()
        for param, index in [(model.input_correction.weight, (0,0,1,1)),
                             (model.state_correction.weight, (0,0,1,1)),
                             (model.threshold, ())]:
            analytic = float(param.grad[index])
            original = float(param.detach()[index])
            values = []
            eps = 1e-6
            for shift in (eps, -eps):
                with torch.no_grad():
                    param[index] = original+shift
                    values.append(float((model(self.b)-target).square().mean()))
            with torch.no_grad():
                param[index] = original
            numerical = (values[0]-values[1])/(2*eps)
            self.assertAlmostEqual(analytic, numerical, delta=1e-7)

    def test_checkpointed_training_has_same_gradients(self):
        a = ConvLISTA(self.h, self.shape, self.pad, .1, layers=3, kernel_size=3, dtype=torch.float64)
        b = ConvLISTA(self.h, self.shape, self.pad, .1, layers=3, kernel_size=3,
                      dtype=torch.float64, checkpoint_layers=True)
        b.load_state_dict(a.state_dict())
        a(self.b).square().mean().backward()
        b(self.b).square().mean().backward()
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-12, atol=1e-12)

    def test_training_updates_weights_and_reload_reproduces_output(self):
        h = np.ones((10, 11), dtype=complex)
        b = self.b.float()
        model = ConvLISTA(h, self.shape, self.pad, .2, layers=3, kernel_size=3)
        rows, epoch, loss, _ = train_model(model, b, b, epochs=12, lr=1e-3)
        self.assertLess(loss, rows[0]['training_mse_normalized'])
        self.assertGreater(float(model.input_correction.weight.detach().norm()), 0)
        self.assertGreater(float(model.state_correction.weight.detach().norm()), 0)
        self.assertGreaterEqual(float(model.threshold.detach()), 0)
        self.assertAlmostEqual(float((model(b)-b).square().mean().detach()), loss, places=7)
        saved = dict(state_dict=cpu_state(model), shape=list(self.shape), pad=self.pad,
                     lambda_normalized=.2, layers=3, kernel_size=3, initial='blurred',
                     norm_bound=model.operator.bound)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.pt'
            torch.save(saved, path)
            loaded, _ = load_checkpoint(path)
            torch.testing.assert_close(loaded(b), model(b), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, 'requires two CUDA GPUs')
    def test_two_gpu_gradients_use_correct_uneven_batch_weighting(self):
        a = ConvLISTA(self.h, self.shape, self.pad, .1, layers=2, kernel_size=3).cuda(0)
        b = ConvLISTA(self.h, self.shape, self.pad, .1, layers=2, kernel_size=3).cuda(0)
        b.load_state_dict(a.state_dict())
        x = self.b.float().cuda(0)
        a(x).square().mean().backward()
        torch.nn.DataParallel(b, device_ids=[0,1])(x).square().mean().backward()
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=3e-4, atol=3e-5)


if __name__ == '__main__':
    unittest.main()
