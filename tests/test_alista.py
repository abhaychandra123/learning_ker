"""Independent numerical checks for ALISTA's equations and experiment pipeline."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0, str(SCRIPTS))
from alista_weights import (analytic_weights, project_columns, validate_dictionary,
                            weight_diagnostics, load_weights)
from alista import ALISTA, load_checkpoint, save_checkpoint, support_threshold
from alista_data import SparseProblem, make_rng
from alista_experiment import baseline_iterates, last_iterate, nmse
from train_alista import train_model, training_stages
from compare_alista import select_baseline_lambdas


class AnalyticWeightTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(32)
        self.d = rng.normal(size=(7, 12))
        self.d /= np.linalg.norm(self.d, axis=0)

    def test_pgd_matches_independent_equality_constrained_kkt_solve(self):
        w, report = analytic_weights(self.d, tolerance=1e-11)
        self.assertTrue(report["converged"])
        q = self.d @ self.d.T
        # A direct augmented-system solve independently enforces each column's
        # first-order condition. Does not reuse the implementation's formula.
        for j in range(self.d.shape[1]):
            dj = self.d[:, j:j+1]
            system = np.block([[q, dj], [dj.T, np.zeros((1, 1))]])
            rhs = np.r_[np.zeros(len(q)), 1.]
            expected = np.linalg.solve(system, rhs)[:-1]
            np.testing.assert_allclose(w[:, j], expected, atol=1e-9)
        self.assertLess(report["objective"], report["initial_objective"])
        self.assertLess(report["max_diagonal_error"], 1e-12)

    def test_svd_matches_pgd_with_redundant_measurements(self):
        d = np.vstack([self.d, self.d[:2]])
        d /= np.linalg.norm(d, axis=0)
        pgd, p = analytic_weights(d, tolerance=1e-11)
        svd, s = analytic_weights(d, method="svd", tolerance=1e-11)
        self.assertTrue(p["converged"] and s["converged"])
        self.assertEqual(s["numerical_rank"], 7)
        np.testing.assert_allclose(svd, pgd, atol=1e-9)

    def test_square_invertible_case_is_inverse_transpose(self):
        d = np.array([[1., .6], [0., .8]])
        for method in ("pgd", "svd"):
            w, report = analytic_weights(d, method=method)
            self.assertTrue(report["converged"])
            np.testing.assert_allclose(w.T @ d, np.eye(2), atol=1e-8)

    def test_projection_direction_and_feasibility(self):
        v = np.random.default_rng(2).normal(size=self.d.shape)
        p = project_columns(v, self.d)
        np.testing.assert_allclose(np.sum(p*self.d, axis=0), 1., atol=1e-14)
        # Projection displacement is parallel to the constraint normal.
        displacement = p-v
        tangent = displacement-self.d*np.sum(displacement*self.d, axis=0)
        np.testing.assert_allclose(tangent, 0., atol=1e-14)

    def test_objective_matches_explicit_gram(self):
        w, _ = analytic_weights(self.d)
        report = weight_diagnostics(self.d, w)
        gram = w.T @ self.d
        self.assertAlmostEqual(report["objective"], np.sum(gram**2), places=11)
        np.fill_diagonal(gram, 0.)
        self.assertAlmostEqual(report["off_diagonal_energy"], np.sum(gram**2), places=11)

    def test_nonconvergence_and_invalid_inputs_are_explicit(self):
        _, report = analytic_weights(self.d, max_iterations=1, tolerance=1e-14)
        self.assertFalse(report["converged"])
        for d in (self.d*2, np.zeros((2, 2)), np.array([[np.nan]]), self.d.astype(complex)):
            with self.assertRaises(ValueError):
                analytic_weights(d)
        for kwargs in ({"tolerance": 0.}, {"method": "ridge"}, {"max_iterations": 0},
                       {"max_workspace_gib": 1e-10}):
            with self.assertRaises(ValueError):
                analytic_weights(self.d, **kwargs)


class ALISTAModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(4)
        self.d = rng.normal(size=(5, 9))
        self.d /= np.linalg.norm(self.d, axis=0)
        self.w, _ = analytic_weights(self.d, method="svd")
        self.b = torch.tensor(rng.normal(size=(3, 5)), dtype=torch.float64)

    def model(self, **kwargs):
        return ALISTA(self.d, self.w, layers=3, dtype=torch.float64, **kwargs)

    def test_every_layer_matches_numpy_equation15(self):
        model = self.model()
        with torch.no_grad():
            for k in range(3):
                model.steps[k].fill_(.4+.1*k)
                model.thresholds[k].fill_(.13+.02*k)
        x = np.zeros((len(self.b), 9))
        states = list(model.iterates(self.b))
        np.testing.assert_array_equal(states[0].detach(), x)
        for k in range(3):
            # Explicit column-vector math, independent of batch-major code.
            columns = []
            for sample in range(len(x)):
                v = x[sample]-(.4+.1*k)*self.w.T@(self.d@x[sample]-self.b[sample].numpy())
                columns.append(np.sign(v)*np.maximum(np.abs(v)-(.13+.02*k), 0))
            x = np.stack(columns)
            np.testing.assert_allclose(states[k+1].detach(), x, atol=1e-13)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 6)
        self.assertFalse(model.D.requires_grad or model.W.requires_grad)
        self.assertEqual(set(dict(model.named_buffers())), {"D", "W"})

    def test_scalar_gradients_finite_differences(self):
        model = self.model()
        with torch.no_grad():
            for p in model.steps:
                p.fill_(.35)
        model(self.b).square().sum().backward()
        for p in model.parameters():
            analytic = float(p.grad)
            original, values = float(p.detach()), []
            for shift in (1e-6, -1e-6):
                with torch.no_grad():
                    p.fill_(original+shift)
                    values.append(float(model(self.b).square().sum()))
            with torch.no_grad():
                p.fill_(original)
            self.assertAlmostEqual(analytic, (values[0]-values[1])/2e-6, delta=1e-6)

    def test_support_selection_strict_ties_and_per_example(self):
        x = torch.tensor([[1., -2., 3., -4., 5.], [10., 10., 10., 0., -20.]], dtype=torch.float64)
        actual = support_threshold(x, torch.tensor(2.), 40.)
        expected = torch.tensor([[0., 0., 1., -4., 5.], [8., 8., 8., 0., -20.]], dtype=torch.float64)
        torch.testing.assert_close(actual, expected)
        for q in (0., 25., 100.):
            one_by_one = torch.cat([support_threshold(row[None], torch.tensor(2.), q) for row in x])
            torch.testing.assert_close(support_threshold(x, torch.tensor(2.), q), one_by_one)

    def test_learned_ista_initialization_is_ista(self):
        model = ALISTA(self.d, self.d, layers=4, mode="learned-ista", initial_lambda=.2,
                       dtype=torch.float64)
        expected = list(baseline_iterates(self.b, model.D, .2, steps=4))
        for actual, target in zip(model.iterates(self.b), expected):
            torch.testing.assert_close(actual, target, atol=1e-13, rtol=1e-13)

    def test_support_percentile_matches_tf112_halfway_rank(self):
        # TF 1.12 sorts descending: round(3 * .5)=2 chooses value 2.
        # Modern ascending quantile chooses 3 instead, altering the support.
        x = torch.tensor([[1., 2., 3., 4.]])
        result = support_threshold(x, torch.tensor(1.), 50.)
        torch.testing.assert_close(result, torch.tensor([[0., 1., 3., 4.]]))
        # At length 6 the reverse-order, ties-to-even choice is the upper value.
        x = torch.tensor([[1., 2., 3., 4., 5., 6.]])
        result = support_threshold(x, torch.tensor(1.), 50.)
        torch.testing.assert_close(result, torch.tensor([[0., 1., 2., 3., 5., 6.]]))

    def test_support_schedule_matches_numpy_all_layers(self):
        model = self.model(support_percent=10., max_support_percent=15.)
        x = np.zeros((len(self.b), 9))
        states = list(model.iterates(self.b))
        for k in range(3):
            theta, step = float(model.thresholds[k].detach()), float(model.steps[k].detach())
            z = x+step*((self.b.numpy()-x@self.d.T)@self.w)
            q = 100.-min((k+1)*10., 15.)
            rank = int(np.rint((z.shape[1]-1)*(1.-q/100.)))
            cutoff = np.sort(np.abs(z), axis=1)[:, ::-1][:, rank:rank+1]
            selected = (np.abs(z) > cutoff) & (np.abs(z) > theta)
            x = np.where(selected, z, np.sign(z)*np.maximum(np.abs(z)-theta, 0.))
            np.testing.assert_allclose(states[k+1].detach(), x, atol=1e-13)

    def test_ista_fista_against_independent_numpy(self):
        step = .99/np.linalg.norm(self.d, 2)**2
        for method in ("ista", "fista"):
            x = np.zeros((3, 9))
            y, t = x.copy(), 1.
            for _ in range(7):
                point = x if method == "ista" else y
                z = point + step*((self.b.numpy()-point@self.d.T)@self.d)
                nx = np.sign(z)*np.maximum(np.abs(z)-step*.12, 0)
                if method == "fista":
                    nt = .5*(1+np.sqrt(1+4*t*t))
                    y, t = nx+((t-1)/nt)*(nx-x), nt
                x = nx
            actual = last_iterate(baseline_iterates(self.b, torch.tensor(self.d), .12,
                                                   method=method, steps=7, step_size=step))
            np.testing.assert_allclose(actual, x, atol=1e-14)

    def test_reload_preserves_float64_and_support_selection(self):
        model = self.model(support_percent=10.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"model.pt"
            save_checkpoint(path, model, {"description": "unit test"})
            loaded, meta = load_checkpoint(path)
            torch.testing.assert_close(loaded(self.b), model(self.b), rtol=0, atol=0)
            self.assertEqual(loaded.D.dtype, torch.float64)
            self.assertEqual(meta["description"], "unit test")

    def test_validation_and_no_extrapolation(self):
        model = self.model()
        for b in (self.b.float(), self.b[:, :2], self.b[:, None], self.b[:0]):
            with self.assertRaises(ValueError):
                model(b)
        with self.assertRaises(ValueError):
            model(self.b, depth=4)
        with self.assertRaises(ValueError):
            ALISTA(self.d, np.zeros_like(self.w))
        with self.assertRaises(ValueError):
            ALISTA(self.d, self.w, support_percent=-1)


class ALISTAExperimentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.d = np.eye(6)
        self.problem = SparseProblem(self.d, probability=.4)

    def test_data_reproducibility_and_independent_splits(self):
        for split in ("train", "validation", "test"):
            b, x = self.problem.sample(20, make_rng(10, split))
            b2, x2 = self.problem.sample(20, make_rng(10, split))
            np.testing.assert_array_equal(b, b2)
            np.testing.assert_array_equal(x, x2)
            np.testing.assert_array_equal(b, x@self.d.T)
        train = self.problem.sample(20, make_rng(10, "train"))[1]
        test = self.problem.sample(20, make_rng(10, "test"))[1]
        self.assertFalse(np.array_equal(train, test))

    def test_noise_scaling_matches_authors_variance_convention(self):
        noisy = SparseProblem(self.d, probability=.4, snr_db=20.)
        b, x = noisy.sample(100000, make_rng(5, "test"))
        expected_noise_energy = np.sum(x.std(axis=1)**2)*6*.01
        self.assertAlmostEqual(np.sum((b-x)**2)/expected_noise_energy, 1., delta=.01)

    def test_stage_schedule_and_training_preserve_fixed_weights(self):
        stages = list(training_stages(3, .001))
        self.assertEqual(len(stages), 9)
        self.assertEqual(stages[6]["active"], [2])
        np.testing.assert_allclose(stages[7]["multipliers"], [.09, .3, 1.])
        model = ALISTA(self.d, self.d, layers=2, initial_lambda=.4)
        validation = self.problem.sample(128, make_rng(3, "validation"))
        b, truth = (torch.tensor(a, dtype=torch.float32) for a in validation)
        before = nmse(model(b), truth)["nmse"]
        d_before, w_before = model.D.clone(), model.W.clone()
        history, summaries, _ = train_model(model, self.problem, validation, seed=3,
                                           steps_per_stage=25, validate_every=5,
                                           patience=30, lr=.01)
        after = nmse(model(b), truth)["nmse"]
        self.assertLess(after, before)
        self.assertAlmostEqual(after, summaries[-1]["best_validation_nmse"], places=12)
        self.assertEqual(len(summaries), 6)
        self.assertGreater(len(history), 0)
        torch.testing.assert_close(model.D, d_before, rtol=0, atol=0)
        torch.testing.assert_close(model.W, w_before, rtol=0, atol=0)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_nmse_is_ratio_of_aggregate_energies(self):
        truth = torch.tensor([[1., 0.], [10., 0.]])
        pred = torch.tensor([[0., 0.], [9., 0.]])
        self.assertAlmostEqual(nmse(pred, truth)["nmse"], 2/101)

    def test_baseline_selection_uses_validation(self):
        truth = torch.tensor([[1., -2., 0., 0., 0., 0.]])
        best, _ = select_baseline_lambdas(truth, truth, torch.eye(6), [0., 1.], [1, 3], 1.)
        self.assertEqual({pair[0] for pair in best.values()}, {0.})

    def test_three_cli_pipeline_and_artifact_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def run(script, *args):
                result = subprocess.run([sys.executable, str(SCRIPTS/script), *map(str, args)],
                                        capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            weights = root/"weights.npz"
            run("alista_weights.py", "--measurements", 8, "--coefficients", 16,
                "--output", weights)
            d, w, meta = load_weights(weights)
            self.assertTrue(meta["converged"])
            run("train_alista.py", "--weights", weights, "--output", root/"train",
                "--layers", 2, "--steps-per-stage", 2, "--validation-samples", 20,
                "--validate-every", 1, "--batch-size", 8)
            run("compare_alista.py", "--checkpoint", root/"train/model.pt",
                "--output", root/"eval", "--test-samples", 20,
                "--long-iterations", 4, "--lambdas", 0., .1, "--benchmark-repeats", 1)
            report = json.loads((root/"eval/metrics.json").read_text())
            self.assertEqual(len(report["results"]), 6)
            self.assertEqual(report["results"][0]["trainable_parameters"], 4)
            self.assertTrue((root/"eval/comparison.png").is_file())
            model, _ = load_checkpoint(root/"train/model.pt")
            with np.load(root/"eval/test_predictions.npz") as saved:
                prediction = model(torch.tensor(saved["b"], dtype=model.D.dtype)).detach()
                np.testing.assert_array_equal(prediction, saved["alista_soft_0"])
            # Altered or unrelated W must not pass an artifact's stored success flag.
            np.savez(root/"bad.npz", D=d, W=w*.5, metadata=json.dumps(meta))
            with self.assertRaises(ValueError):
                load_weights(root/"bad.npz")

    def test_compare_multiple_models_and_reject_changed_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def run(script, *args, success=True):
                result = subprocess.run([sys.executable, str(SCRIPTS/script), *map(str, args)],
                                        capture_output=True, text=True, timeout=90)
                if success:
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("checksum mismatch", result.stderr)
            run("alista_weights.py", "--measurements", 4, "--coefficients", 8,
                "--output", root/"weights.npz")
            for label, extra in (("ss", ("--support-percent", 1.2)),
                                 ("control", ("--mode", "learned-ista"))):
                run("train_alista.py", "--weights", root/"weights.npz", "--output", root/label,
                    "--layers", 2, "--steps-per-stage", 1, "--validation-samples", 20,
                    "--batch-size", 8, *extra)
            run("compare_alista.py", "--checkpoint", root/"ss/model.pt", root/"control/model.pt",
                "--output", root/"eval", "--test-samples", 20, "--long-iterations", 3,
                "--lambdas", 0., .1, "--benchmark-repeats", 1)
            with (root/"ss/validation.npz").open("ab") as stream:
                stream.write(b"changed")
            run("compare_alista.py", "--checkpoint", root/"ss/model.pt", "--output", root/"bad",
                success=False)


if __name__ == "__main__":
    unittest.main()
