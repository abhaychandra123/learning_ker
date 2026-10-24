"""Synthetic training physics, controlled selection, restart and evaluation isolation."""
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from synthetic_data import SyntheticDataset, prepare
from synthetic_experiment import select_samples, verify_actual_run, field_metrics
from compare_synthetic_experiment import compare_runs
from test_synthetic_data import source_fixture

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from train_synthetic import main, RecipeBatches


def fixture(root):
    source = source_fixture(root/'source')
    manifest = json.loads(source.read_text())
    actual = root/'actual'; (actual/'comparison').mkdir(parents=True)
    (actual/'setup.json').write_text(json.dumps({'manifest': manifest}))
    for pair in manifest['pairs']:
        sharp, blur = (np.load(source.parent/pair[k]) for k in ('sharp', 'blurred'))
        np.savez(actual/'comparison'/f"{pair['id']}.npz", sharp=sharp, blurred=blur, lista=blur)
    with contextlib.redirect_stdout(io.StringIO()):
        prepare(root/'dataset', source_manifest=source, height_count=2, min_dz_um=.5,
                noise_percent=[0, 1], rotations=[0, 90, 180, 270], pad=4, interior_margin=2)
    return SyntheticDataset(root/'dataset'), manifest


class ExperimentDataTests(unittest.TestCase):
    def test_matched_original_is_unique_noiseless_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); ds, m = fixture(root)
            small, full = select_samples(ds, m, 'matched-original'), select_samples(ds, m, 'full')
            self.assertEqual(len(small), 2)
            self.assertEqual(len(full), 32)
            for i in small:
                self.assertEqual(ds.samples[i]['target_z_um'], 1.5)
                self.assertEqual(ds.samples[i]['noise_percent'], 0)
                self.assertEqual(ds.samples[i]['rotation_degrees'], 0)
            m['pairs'][0]['blur_z_um'] = 1.6
            with self.assertRaisesRegex(ValueError, 'Expected one'):
                select_samples(ds, m, 'matched-original')
            ds.samples[0]['split'] = 'test'
            with self.assertRaisesRegex(ValueError, 'all-train'):
                select_samples(ds, m, 'full')

    def test_actual_reference_integrity_and_amplitude_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); ds, m = fixture(root)
            _, hashes = verify_actual_run(root/'actual', ds)
            self.assertEqual(len(hashes), 3)
            path = root/'actual/comparison/A_pair.npz'
            with np.load(path) as a:
                values = {k: a[k] for k in a.files}
            values['sharp'][0, 0, 0] += .1
            np.savez(path, **values)
            with self.assertRaisesRegex(ValueError, 'reference fields differ'):
                verify_actual_run(root/'actual', ds)
        truth = np.array([[-3., -1.], [1., 3.]])
        metrics = field_metrics(.1*truth, truth)
        self.assertAlmostEqual(metrics['relative_l2'], .9)
        self.assertAlmostEqual(metrics['amplitude_slope'], .1)

    def test_paired_report_rejects_unmatched_contracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for arm, subset, factor in [('control', 'matched-original', 1.), ('full', 'full', .5)]:
                r = root/arm; r.mkdir()
                setup = dict(contract={'config': {'subset': subset, 'seed': 0}}, scale_uT=1.,
                             model_config={}, pad=4, optimizer_updates_per_epoch=1,
                             sample_ids=['a'] if arm == 'control' else ['a', 'b'],
                             component_pairs=3 if arm == 'control' else 6)
                (r/'setup.json').write_text(json.dumps(setup))
                (r/'status.json').write_text(json.dumps({'status':'complete', 'completed_epochs':2}))
                (r/'final.pt').write_bytes(b'test-report-checkpoint')
                with (r/'actual_comsol.csv').open('w', newline='') as stream:
                    row = dict(pair_id='A_pair', geometry='A', component='Bx', sharp_z_um=.5,
                               blur_z_um=1.5, method='synthetic_final', rmse_uT=factor,
                               nrmse_range=factor*.1, relative_l2=factor*.2, amplitude_slope=.9)
                    w = csv.DictWriter(stream, fieldnames=list(row)); w.writeheader(); w.writerow(row)
            with contextlib.redirect_stdout(io.StringIO()):
                result = compare_runs(root/'control', root/'full', root/'report')
            self.assertAlmostEqual(result[0]['mean_delta_nrmse_range'], -.05)
            self.assertEqual(result[0]['nrmse_improved_pairs'], 1)
            status = root/'full/status.json'
            status.write_text(json.dumps({'status': 'complete', 'completed_epochs': 3}))
            with self.assertRaisesRegex(ValueError, 'same optimizer'):
                compare_runs(root/'control', root/'full', root/'bad-report')
            self.assertFalse((root/'bad-report').exists())


@unittest.skipUnless(HAS_TORCH, 'PyTorch is not installed in this interpreter')
class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_rotated_rectangular_operator_matches_generator(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds, _ = fixture(Path(tmp))
            indices = [i for i,s in enumerate(ds.samples) if s['noise_percent'] == 0]
            batches = RecipeBatches(ds, indices, torch.device('cpu'))
            for b, truth, operator in batches:
                torch.testing.assert_close(operator(truth), b, atol=2e-7, rtol=2e-5)

    def test_resume_matches_uninterrupted_and_report(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp); fixture(root)
            base = ['--dataset', str(root/'dataset'), '--actual-run', str(root/'actual'),
                    '--device', 'cpu', '--layers', '2', '--kernel-size', '3', '--checkpoint-layers']
            control = base+['--subset','matched-original']
            main(control+['--output', str(root/'resumed'), '--epochs','1'])
            main(control+['--output', str(root/'resumed'), '--epochs','2','--resume'])
            main(control+['--output', str(root/'continuous'), '--epochs','2'])
            a = torch.load(root/'resumed/last.pt', weights_only=True)
            b = torch.load(root/'continuous/last.pt', weights_only=True)
            for key in a['model_state']:
                torch.testing.assert_close(a['model_state'][key], b['model_state'][key], atol=0, rtol=0)
            main(base+['--subset','full','--output',str(root/'full'),'--epochs','2'])
            result = compare_runs(root/'continuous', root/'full', root/'report')
            self.assertEqual(result[0]['component_pairs'], 6)
            with self.assertRaisesRegex(ValueError, 'identical'):
                main(control+['--output',str(root/'resumed'),'--epochs','3','--resume','--lr','.002'])
            with self.assertRaises(FileExistsError):
                main(control+['--output',str(root/'continuous'),'--epochs','2'])

    def test_actual_inputs_do_not_affect_training_weights(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp); fixture(root)
            base = ['--dataset',str(root/'dataset'),'--actual-run',str(root/'actual'),
                    '--device','cpu','--layers','2','--kernel-size','3','--epochs','1',
                    '--subset','matched-original']
            main(base+['--output',str(root/'first')])
            for path in (root/'actual/comparison').glob('*.npz'):
                with np.load(path) as a:
                    fields = {k:a[k] for k in a.files}
                fields['blurred'] *= .25
                np.savez(path, **fields)
            main(base+['--output',str(root/'second')])
            a = torch.load(root/'first/final.pt', weights_only=True)
            b = torch.load(root/'second/final.pt', weights_only=True)
            for k in a['model_state']:
                torch.testing.assert_close(a['model_state'][k],b['model_state'][k],atol=0,rtol=0)


if __name__ == '__main__':
    unittest.main()
