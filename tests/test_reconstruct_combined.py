"""CLI selection, reporting integrity and optional checkpoint inference checks."""
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import reconstruct_combined as rc


def fixture(root):
    (root/'comparison').mkdir(parents=True)
    (root/'physics').mkdir()
    observed = np.linspace(2,5,60,dtype=np.float32).reshape(3,4,5)
    truth = observed+.3
    arrays = dict(sharp=truth, blurred=observed, lista=observed+.2)
    arrays.update({name:observed+.1 for name in rc.BASELINES})
    np.savez(root/'comparison/pair.npz', **arrays)
    np.save(root/'physics/pair.npy', np.ones((6,7)))
    config = dict(lam=.02, layers=2, kernel_size=3, initial='blurred', checkpoint_layers=False)
    setup = dict(model_config=config, scale_uT=5., contract={'config':{'pad':1}},
        physics={'pair':{'path':'physics/pair.npy','sha256':rc.sha256(root/'physics/pair.npy')}},
        manifest={'field_unit':'uT','length_unit':'um','components':list(rc.COMPONENTS),
            'geometries':{'G':{'grid':dict(nx=5,ny=4,dx=.1,dy=.2,x0=0.,y0=0.)}},
            'pairs':[dict(id='pair',geometry='G',sharp_z_um=.5,blur_z_um=10.)]})
    rc.write_json(root/'setup.json', setup)
    with (root/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['pair_id','component','method','converged','stop_reason','iterations'])
        w.writeheader()
        for c in rc.COMPONENTS:
            for method in rc.BASELINES:
                failed = c=='Bx' and method=='ista_to_epsilon'
                fixed = method.endswith('_16')
                w.writerow(dict(pair_id='pair',component=c,method=method,
                           converged='' if fixed else not failed,
                           stop_reason='fixed_depth' if fixed else 'safety_limit' if failed else 'converged',
                           iterations=16 if fixed else 100000 if failed else 200))
    return setup, arrays


class ReconstructionTests(unittest.TestCase):
    def invoke(self, root, out='out', *flags):
        with contextlib.redirect_stdout(io.StringIO()), patch.object(rc,'render'):
            rc.main(['--run',str(root/'run'),'--geometry','G','--standoff','10',
                     '--output',str(root/out),*flags])

    def test_saved_replay_preserves_fields_metrics_and_nonconvergence(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); _,arrays=fixture(root/'run')
            self.invoke(root,'out','--saved-results','--include-saved-baselines')
            np.testing.assert_array_equal(np.load(root/'out/reconstruction_uT.npy'),arrays['lista'])
            rows=json.loads((root/'out/metrics.json').read_text())['rows']
            r=next(r for r in rows if r['method']=='lista' and r['component']=='Bx')
            expected=100*np.sqrt(np.mean((arrays['lista'][0].astype(float)-arrays['sharp'][0])**2))/np.ptp(arrays['sharp'][0].astype(float))
            self.assertAlmostEqual(r['range_nrmse_percent'],expected)
            self.assertIsNone(r['inference_seconds'])
            self.assertEqual(sum(r['converged'] is False for r in rows),1)
            with self.assertRaises(FileExistsError): self.invoke(root,'out','--saved-results')

    def test_external_input_never_borrows_reference_and_uses_requested_component(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); _,arrays=fixture(root/'run')
            np.save(root/'input.npy',arrays['blurred'][2])
            def fake_infer(run,setup,observed,transfer,pad,device):
                self.assertEqual(observed.shape,(1,4,5))
                np.testing.assert_array_equal(observed[0],arrays['blurred'][2])
                return observed.copy(),[.1],{'test_stub':True}
            with patch.object(rc,'infer',side_effect=fake_infer):
                self.invoke(root,'external','--input',str(root/'input.npy'),'--component','Bz')
            with np.load(root/'external/fields_uT.npz') as a:
                self.assertNotIn('reference',a.files)
            rows=json.loads((root/'external/metrics.json').read_text())['rows']
            self.assertTrue(all(r['range_nrmse_percent'] is None for r in rows))
            self.assertIsNone(rc.metrics(np.ones((2,2)),np.ones((2,2)))['range_nrmse_percent'])

    def test_invalid_pair_source_and_operator_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); setup,_=fixture(root/'run')
            for flags in [('--standoff','8'),('--target-standoff','10'),
                          ('--saved-results','--operator','analytic'),
                          ('--reference',str(root/'no.npy'))]:
                with self.assertRaises(ValueError): self.invoke(root,'invalid',*flags)
                self.assertFalse((root/'invalid').exists())
            np.save(root/'run/physics/pair.npy',np.zeros((6,7)))
            with self.assertRaisesRegex(ValueError,'checksum'):self.invoke(root,'bad','--saved-results')
            (root/'run/physics/pair.npy').unlink()
            self.invoke(root,'replay','--saved-results')
            with self.assertRaisesRegex(FileNotFoundError,'physics'):self.invoke(root,'infer')
            self.assertFalse(json.loads((root/'replay/provenance.json').read_text())['operator']['file_verified'])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch not installed in this Python')
    def test_actual_checkpoint_identity_operator_normalization(self):
        import torch
        from combined_lista import SharedConvLISTA
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); setup,arrays=fixture(root/'run')
            model=SharedConvLISTA(**setup['model_config'])
            torch.save(dict(model_config=setup['model_config'],model_state=model.state_dict(),
                            scale_uT=5.,contract=setup['contract'],best_epoch=0),root/'run/best.pt')
            self.invoke(root,'inferred','--device','cpu')
            # Identity A, two ISTA-equivalent layers: positive values shrink by
            # (0.99 + 0.0099) * normalized lambda * physical scale.
            expected=arrays['blurred']-.9999*.02*5
            np.testing.assert_allclose(np.load(root/'inferred/reconstruction_uT.npy'),expected,atol=2e-6)
            rows=json.loads((root/'inferred/metrics.json').read_text())['rows']
            self.assertTrue(all(r['inference_seconds'] >= 0 for r in rows if r['method']=='lista'))


if __name__=='__main__': unittest.main()
