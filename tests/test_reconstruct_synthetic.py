"""Verify new-model inference agrees with the experiment's actual-data evaluator."""
import contextlib
import csv
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
from combined_lista import SharedConvLISTA
from reconstruct_synthetic import main
from synthetic_data import digest
from synthetic_experiment import verify_actual_run
from test_synthetic_training import fixture
from train_synthetic import evaluate_actual
from reconstruct_combined import BASELINES


class ReconstructionTests(unittest.TestCase):
    def prepare(self, root, baselines=False):
        ds, manifest = fixture(root)
        if baselines:
            setup_path = root/'actual/setup.json'
            setup = json.loads(setup_path.read_text())
            setup['contract'] = {'config': {'operator':'joint', 'pad':2}}
            setup_path.write_text(json.dumps(setup))
            baseline_rows = []
            for pair in manifest['pairs']:
                path = root/'actual/comparison'/f"{pair['id']}.npz"
                with np.load(path) as archive:
                    values = {k:archive[k] for k in archive.files}
                values.update({k:values['blurred']*.8 for k in BASELINES})
                np.savez(path, **values)
                for c in ('Bx','By','Bz'):
                    for method in BASELINES:
                        baseline_rows.append(dict(pair_id=pair['id'],component=c,method=method,
                            iterations=16,converged='False',stop_reason='safety_limit'))
            with (root/'actual/comparison.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(baseline_rows[0]))
                writer.writeheader(); writer.writerows(baseline_rows)
        _, hashes = verify_actual_run(root/'actual', ds)
        model = SharedConvLISTA(lam=.01/ds.scale, layers=2, kernel_size=3)
        with torch.no_grad():
            model.input_correction.weight[0, 0, 1, 1] = .1
        config = dict(lam=.01/ds.scale, layers=2, kernel_size=3)
        contract = dict(dataset_sha256=digest(root/'dataset/manifest.json'), actual_sha256=hashes)
        run = root/'model'; run.mkdir()
        setup = dict(contract=contract, model_config=config, scale_uT=ds.scale, pad=4)
        (run/'setup.json').write_text(json.dumps(setup))
        torch.save(dict(contract=contract, model_config=config, scale_uT=ds.scale,
                        model_state=model.state_dict(), epoch=200), run/'final.pt')
        pair = manifest['pairs'][0]
        args = ['--run', str(run), '--dataset', str(root/'dataset'), '--actual-run', str(root/'actual'),
                '--geometry', pair['geometry'], '--standoff', str(pair['blur_z_um']),
                '--target-standoff', str(pair['sharp_z_um']), '--output', str(root/'report')]
        return ds, manifest, hashes, model, args

    def test_matches_final_evaluator_and_records_final_provenance(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            ds, manifest, hashes, model, args = self.prepare(root)
            evaluation = root/'evaluation'; evaluation.mkdir()
            evaluate_actual(model, ds, root/'actual', manifest, hashes, evaluation, True)
            with patch('reconstruct_synthetic.render'):
                arrays, rows = main(args)
            with np.load(evaluation/'actual_predictions'/f"{manifest['pairs'][0]['id']}.npz") as saved:
                np.testing.assert_array_equal(arrays['lista'], saved['lista'])
            provenance = json.loads((root/'report/provenance.json').read_text())
            self.assertEqual(provenance['epoch'], 200)
            self.assertEqual(provenance['operator']['pad'], 4)
            self.assertEqual(provenance['checkpoint_sha256'], digest(root/'model/final.pt'))
            self.assertEqual(len(rows), 6)
            self.assertEqual(json.loads((root/'report/status.json').read_text())['status'], 'complete')

    def test_refuses_wrong_checkpoint_and_changed_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, manifest, _, _, args = self.prepare(root)
            with self.assertRaisesRegex(ValueError, 'Checkpoint checksum'):
                main(args+['--expected-checkpoint-sha256', '0'*64])
            path = root/'actual/comparison'/f"{manifest['pairs'][0]['id']}.npz"
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'COMSOL archive differs'):
                main(args)
            self.assertFalse((root/'report').exists())

    def test_baseline_replot_preserves_new_prediction_and_stopping_metadata(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            _, _, _, _, args = self.prepare(root, baselines=True)
            with patch('reconstruct_synthetic.render'):
                main(args)
                before = digest(root/'report/reconstruction_uT.npy')
                main(['--replot',str(root/'report'),'--actual-run',str(root/'actual'),'--include-saved-baselines'])
                # Repeated replot must not duplicate rows or explanatory text.
                main(['--replot',str(root/'report'),'--actual-run',str(root/'actual'),'--include-saved-baselines'])
            self.assertEqual(before, digest(root/'report/reconstruction_uT.npy'))
            rows = json.loads((root/'report/metrics.json').read_text())['rows']
            self.assertEqual(len(rows), 18)
            baseline_rows = [r for r in rows if r['method'] in BASELINES]
            self.assertTrue(all(r['converged'] is False and r['reused'] and r['iterations']==16 for r in baseline_rows))
            status = json.loads((root/'report/status.json').read_text())
            self.assertEqual(status['baseline_nonconverged_rows'], 12)
            provenance = json.loads((root/'report/provenance.json').read_text())
            self.assertEqual(provenance['interpretation'].count('Historical saved ISTA/FISTA:'), 1)
            self.assertEqual(provenance['baselines']['pad'], 2)
            self.assertEqual(provenance['operator']['pad'], 4)
            self.assertEqual(provenance['epoch'], 200)


if __name__ == '__main__':
    unittest.main()
