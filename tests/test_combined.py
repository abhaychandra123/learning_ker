"""Combined-dataset boundaries, shared physics, streaming, and restart checks."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from combined_data import discover, height_pairs, prepare, read_manifest
from combined_lista import SharedConvLISTA, load_shared_checkpoint
from compare_combined import main, split_pairs, training_scale, dataset_loss, PILOT_DRIVER_SHA256
from compare_lista import fixed_steps
from torch_inverse import PaddedFFT


def write_plane(path, z, geometry_scale=1., shift=0.):
    y, x = np.mgrid[:5, :6]
    # Distinct components, signs, and geometry amplitudes; stable tiny fixture.
    fields = np.stack([np.sin(x*.4+y*.3), np.cos(x*.3-y*.2), np.sin(x*.3)*np.cos(y*.4)])
    fields *= geometry_scale/(1+z*.1)*1e-6
    rows = np.column_stack([x.ravel()+shift, y.ravel(),
                            fields.reshape(3, -1).T, np.linalg.norm(fields, axis=0).ravel()])
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ['% Model,test', '% Version,test', '% Date,test', '% Dimension,2',
              '% Nodes,30', '% Expressions,4', '% Description,test', '% Length unit,µm',
              '% x,y,mf.Bx (T),mf.By (T),mf.Bz (T),mf.normB (T)']
    with path.open('w') as stream:
        stream.write('\n'.join(header)+'\n')
        np.savetxt(stream, rows, delimiter=',')


def make_data(root):
    for geometry, heights, scale in [('L1_C1', [.5, 1., 2.], 1.),
                                     ('IC_4um', [.5, 3.], 2.), ('IC_2um', [1., 2.], 3.)]:
        folder = root if geometry == 'L1_C1' else root/'data2'
        for z in heights:
            name = f'{geometry}_Z{z:.1f}um_test.csv'.replace('.', 'p', 1)
            write_plane(folder/name, z, scale)


class CombinedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)

    def test_inventory_separates_geometry_and_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)/'data'
            make_data(raw)
            self.assertEqual({g: len(v) for g, v in discover(raw).items()},
                             {'L1_C1': 3, 'IC_4um': 2, 'IC_2um': 2})
            self.assertEqual(height_pairs([.5, 1, 2], 'anchor'), [(.5, 1), (.5, 2)])
            self.assertEqual(len(height_pairs([.5, 1, 2], 'all')), 3)
            write_plane(raw/'IC_2um_Z1p0um_duplicate.csv', 1.)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                discover(raw)

    def test_real_conversion_units_and_cross_geometry_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, out = Path(tmp)/'data', Path(tmp)/'prepared'
            make_data(raw)
            m = prepare(raw, out)
            self.assertEqual(len(m['pairs']), 4)
            p = out/'manifest.json'
            read_manifest(p)
            a = np.load(out/m['pairs'][0]['sharp'])
            self.assertGreater(np.abs(a).max(), .1)  # µT, not T
            m['pairs'][0]['blurred'] = m['geometries']['IC_2um']['planes']['2.0']['path']
            p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, 'crosses geometries'):
                read_manifest(p)

    def test_grid_mismatch_and_hash_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, out = Path(tmp)/'data', Path(tmp)/'prepared'
            make_data(raw)
            write_plane(raw/'L1_C1_Z2p0um_test.csv', 2., shift=1.)
            with self.assertRaisesRegex(ValueError, 'Grid mismatch'):
                prepare(raw, out)
        with tempfile.TemporaryDirectory() as tmp:
            raw, out = Path(tmp)/'data', Path(tmp)/'prepared'
            make_data(raw)
            m = prepare(raw, out)
            p = out/m['pairs'][0]['sharp']
            a = np.load(p); a[0, 0, 0] += 1
            np.save(p, a)
            with self.assertRaisesRegex(ValueError, 'Checksum'):
                read_manifest(out/'manifest.json')

    def test_split_requires_truth_independent_operator_and_train_only_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, out = Path(tmp)/'data', Path(tmp)/'prepared'
            make_data(raw)
            m = prepare(raw, out)
            with self.assertRaisesRegex(ValueError, 'analytic'):
                split_pairs(m, ['IC_4um'], ['IC_2um'], 'joint')
            split = split_pairs(m, ['IC_4um'], ['IC_2um'], 'analytic')
            self.assertEqual({p['geometry'] for p in split['train']}, {'L1_C1'})
            q = training_scale(out, split['train'])
            self.assertLess(q, 1.)
            self.assertGreater(training_scale(out, m['pairs']), 2.)
            with self.assertRaises(ValueError):
                split_pairs(m, ['IC_2um'], ['IC_2um'], 'analytic')

    def test_shared_initialization_equals_ista_for_multiple_operators(self):
        b = torch.randn(3, 1, 5, 6, dtype=torch.float64)
        model = SharedConvLISTA(.03, layers=4, kernel_size=3, dtype=torch.float64)
        for gain in (.3, 1., 2.):
            operator = PaddedFFT(np.ones((9, 10))*gain, (5, 6), 2, torch.float64)
            expected = fixed_steps(b, operator, .03, 'ista', 4, 'blurred')
            torch.testing.assert_close(model(b, operator), expected, atol=1e-12, rtol=1e-12)

    def test_checkpointed_gradients_equal_regular_gradients(self):
        b = torch.randn(2, 1, 5, 6)
        operator = PaddedFFT(np.ones((9, 10))*.7, (5, 6), 2)
        a = SharedConvLISTA(.03, layers=3, kernel_size=3)
        bmodel = SharedConvLISTA(.03, layers=3, kernel_size=3, checkpoint_layers=True)
        bmodel.load_state_dict(a.state_dict())
        a(b, operator).square().mean().backward()
        bmodel(b, operator).square().mean().backward()
        for x, y in zip(a.parameters(), bmodel.parameters()):
            torch.testing.assert_close(x.grad, y.grad, atol=1e-6, rtol=1e-6)

    def test_end_to_end_resume_matches_uninterrupted_and_metrics_are_reloadable(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            raw, data = root/'data', root/'prepared'
            make_data(raw)
            prepare(raw, data)
            base = ['--manifest', str(data/'manifest.json'), '--device', 'cpu', '--layers', '2',
                    '--kernel-size', '3', '--pad', '1', '--epsilon', '.1', '--benchmark-repeats', '1',
                    '--max-iterations', '2000', '--save-reconstructions', '--checkpoint-layers']
            a, b = root/'continuous', root/'resumed'
            main(base+['--output', str(a), '--epochs', '2'])
            main(base+['--output', str(b), '--epochs', '1'])
            main(base+['--output', str(b), '--epochs', '2', '--resume'])
            ca = torch.load(a/'last.pt', weights_only=True)
            cb = torch.load(b/'last.pt', weights_only=True)
            for name in ca['model_state']:
                torch.testing.assert_close(ca['model_state'][name], cb['model_state'][name], atol=0, rtol=0)
            self.assertEqual(ca['best_epoch'], cb['best_epoch'])
            self.assertEqual(ca['best_loss'], cb['best_loss'])
            loaded, info = load_shared_checkpoint(b/'best.pt')
            for name in info['model_state']:
                torch.testing.assert_close(loaded.state_dict()[name], cb['best_state'][name])
            self.assertTrue((b/'index.html').is_file())
            self.assertEqual(json.loads((b/'status.json').read_text())['status'], 'complete')
            with self.assertRaisesRegex(ValueError, 'identical'):
                main(base+['--output', str(b), '--epochs', '3', '--resume', '--lr', '.01'])
            with self.assertRaises(FileExistsError):
                main(base+['--output', str(b), '--epochs', '2'])
            # Streaming one vs three components preserves equal-example gradients.
            from compare_combined import build_physics
            from compare_lista import cpu_state
            m = read_manifest(data/'manifest.json')
            physics = json.loads((b/'setup.json').read_text())['physics']
            ctx = dict(load=dict(manifest=m, root=data, output=b, physics=physics, pad=1, device='cpu'),
                       scale=info['scale_uT'], batch_size=1, device='cpu')
            model = SharedConvLISTA(1e-4/ctx['scale'], layers=2, kernel_size=3)
            dataset_loss(model, m['pairs'], ctx, backward=True)
            gradients = [p.grad.clone() for p in model.parameters()]
            model.zero_grad(); ctx['batch_size'] = 3
            dataset_loss(model, m['pairs'], ctx, backward=True)
            for expected, param in zip(gradients, model.parameters()):
                torch.testing.assert_close(param.grad, expected, rtol=2e-5, atol=2e-6)

    def test_baseline_reuse_skips_solvers_preserves_failures_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            raw, data, pilot, full = [root/name for name in ('raw', 'data', 'pilot', 'full')]
            make_data(raw)
            prepare(raw, data)
            base = ['--manifest', str(data/'manifest.json'), '--device', 'cpu', '--layers', '2',
                    '--kernel-size', '3', '--pad', '1', '--epsilon', '.001', '--benchmark-repeats', '1',
                    '--max-iterations', '1', '--save-reconstructions']
            with self.assertRaisesRegex(SystemExit, 'did not converge'):
                main(base+['--output', str(pilot), '--epochs', '0'])
            with self.assertRaisesRegex(SystemExit, 'did not converge'):
                main(base+['--output', str(root/'control'), '--epochs', '2'])
            # Exercise the explicitly audited old pilot driver, not a blanket hash bypass.
            setup = json.loads((pilot/'setup.json').read_text())
            setup['contract']['source_sha256']['compare_combined.py'] = PILOT_DRIVER_SHA256
            (pilot/'setup.json').write_text(json.dumps(setup))
            reuse = base+['--output', str(full), '--reuse-baselines-from', str(pilot)]
            with patch('compare_combined.solve_torch', side_effect=AssertionError('solver rerun')), \
                    patch('compare_combined.fixed_steps', side_effect=AssertionError('fixed solver rerun')):
                with self.assertRaisesRegex(SystemExit, 'did not converge'):
                    main(reuse+['--epochs', '1'])
                with self.assertRaisesRegex(SystemExit, 'did not converge'):
                    main(reuse+['--epochs', '2', '--resume'])
            first = json.loads((pilot/'status.json').read_text())
            second = json.loads((full/'status.json').read_text())
            self.assertEqual(first['nonconverged_batches'], second['nonconverged_batches'])
            self.assertEqual(second['completed_epochs'], 2)
            expected = torch.load(root/'control'/'last.pt', weights_only=True)
            actual = torch.load(full/'last.pt', weights_only=True)
            for key in expected['model_state']:
                torch.testing.assert_close(actual['model_state'][key], expected['model_state'][key],
                                           atol=0, rtol=0)
            for file in (pilot/'comparison').glob('*.json'):
                a, b = json.loads(file.read_text()), json.loads((full/'comparison'/file.name).read_text())
                self.assertEqual([r for r in a['rows'] if r['method'] != 'lista'],
                                 [r for r in b['rows'] if r['method'] != 'lista'])
                self.assertEqual(a['histories'], b['histories'])
                with np.load(file.with_suffix('.npz')) as x, \
                        np.load((full/'comparison'/file.name).with_suffix('.npz')) as y:
                    for key in x.files:
                        if key != 'lista':
                            np.testing.assert_array_equal(x[key], y[key])
            provenance = json.loads((full/'metrics.json').read_text())['baseline_reuse']
            self.assertEqual(provenance['source'], str(pilot.resolve()))
            self.assertTrue(provenance['file_sha256'])
            # Data/settings/code/environment/physics mismatches fail BEFORE creating output.
            original = json.loads((pilot/'setup.json').read_text())
            changes = [
                lambda s: s['contract']['config'].update(epsilon=.01),
                lambda s: s['contract'].update(manifest_sha256='bad'),
                lambda s: s['contract']['source_sha256'].update({'torch_inverse.py': 'bad'}),
                lambda s: s['contract']['source_sha256'].update({'compare_combined.py': 'unknown'}),
                lambda s: s.update(scale_uT=999),
                lambda s: s['environment'].update(torch='different'),
                lambda s: next(iter(s['physics'].values())).update(sha256='bad'),
            ]
            for change in changes:
                altered = json.loads(json.dumps(original))
                change(altered)
                (pilot/'setup.json').write_text(json.dumps(altered))
                with self.assertRaises(ValueError):
                    main(base+['--output', str(root/'rejected'), '--epochs', '0',
                               '--reuse-baselines-from', str(pilot)])
                self.assertFalse((root/'rejected').exists())
            (pilot/'setup.json').write_text(json.dumps(original))
            report_file = next((pilot/'comparison').glob('*.json'))
            report = json.loads(report_file.read_text())
            report['rows'] = [r for r in report['rows'] if r['method'] != 'ista_to_epsilon']
            report_file.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                main(base+['--output', str(root/'rejected'), '--epochs', '0',
                           '--reuse-baselines-from', str(pilot)])
            self.assertFalse((root/'rejected').exists())


if __name__ == '__main__':
    unittest.main()
