"""Numerical equivalence and checkpoint continuity for optimized full-field training."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from synthetic_data import SyntheticDataset,propagate,transfer
from synthetic_fast import CachedBatches,RealAnalyticFFT,cached_loss
from train_synthetic import RecipeBatches,loss_pass,main as legacy_main
from train_synthetic_fast import main as fast_main
from combined_lista import SharedConvLISTA
from torch_inverse import PaddedFFT
from compare_synthetic_experiment import compare_runs
from switch_synthetic_run import training_command_matches
from audit_synthetic_roi import energy_box
from test_synthetic_training import fixture


class FastTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)

    def test_real_fft_forward_adjoint_and_gradients_even_odd_grids(self):
        for shape,pad in [((20,24),4),((19,23),3),((17,22),0)]:
            grid=dict(dx=.2,dy=.35)
            full=PaddedFFT(transfer(shape,.2,.35,1.3,pad),shape,pad,dtype=torch.float64)
            real=RealAnalyticFFT(shape,grid,1.3,pad,dtype=torch.float64)
            x=torch.randn(2,1,*shape,dtype=torch.float64,requires_grad=True)
            y=torch.randn_like(x)
            torch.testing.assert_close(full(x),real(x),atol=1e-12,rtol=1e-12)
            torch.testing.assert_close(full.adjoint(y),real.adjoint(y),atol=1e-12,rtol=1e-12)
            a=torch.autograd.grad(full(x).square().sum(),x)[0]
            b=torch.autograd.grad(real(x).square().sum(),x)[0]
            torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)

    def test_cache_preserves_every_input_target_and_reuses_propagations(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            ds,_=fixture(Path(tmp))
            indices=list(range(len(ds.samples)))
            with patch('synthetic_data.propagate',wraps=propagate) as calls:
                cache=CachedBatches(ds,indices,'cpu',batch_size=5,cache='cpu',fft='real')
                # 2 geometries * 2 heights * 3 source components, independent of rotation/noise.
                self.assertEqual(calls.call_count,12)
                self.assertEqual(cache.info['propagation_calls'],12)
                first=list(cache)
                second=list(cache)
                self.assertEqual(calls.call_count,12)
            observed=[(x,t) for b,y,_ in first for x,t in zip(b,y)]
            self.assertEqual(len(observed),3*len(indices))
            for n,(x,y) in enumerate(observed):
                old=ds.component(n//3,n%3,normalized=True)
                np.testing.assert_array_equal(x[0].numpy(),old['blurred'])
                np.testing.assert_array_equal(y[0].numpy(),old['sharp'])
            self.assertLess(cache.info['shared_target_bytes'],cache.info['input_bytes'])
            self.assertEqual(sum(len(b) for b,_,_ in second),len(observed))

    def test_batched_gradient_equals_scalar_accumulation(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            ds,_=fixture(Path(tmp));ids=list(range(len(ds.samples)))
            model=SharedConvLISTA(1e-4/ds.scale,layers=3,kernel_size=3)
            with torch.no_grad():
                model.input_correction.weight.normal_(0,.001)
                model.state_correction.weight.normal_(0,.001)
            other=copy.deepcopy(model)
            old_loss=loss_pass(model,RecipeBatches(ds,ids,torch.device('cpu')),backward=True)
            new_loss=cached_loss(other,CachedBatches(ds,ids,'cpu',5,'cpu','real'),backward=True)
            self.assertAlmostEqual(old_loss,new_loss,places=8)
            for a,b in zip(model.parameters(),other.parameters()):
                torch.testing.assert_close(a.grad,b.grad,atol=3e-7,rtol=3e-5)

    def test_migration_retains_state_and_matches_control_prefix(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);fixture(root)
            base=['--dataset',str(root/'dataset'),'--actual-run',str(root/'actual'),
                  '--device','cpu','--layers','2','--kernel-size','3','--subset','full']
            legacy_main(base+['--output',str(root/'legacy'),'--epochs','1','--checkpoint-layers'])
            parent=torch.load(root/'legacy/last.pt',weights_only=True)
            opts=['--cache','cpu','--batch-size','4','--fft','real','--eval-every','5']
            fast_main(base+opts+['--output',str(root/'fast'),'--epochs','1',
                                '--migrate-from',str(root/'legacy/last.pt')])
            migrated=torch.load(root/'fast/last.pt',weights_only=True)
            for k in parent['model_state']:
                torch.testing.assert_close(parent['model_state'][k],migrated['model_state'][k],atol=0,rtol=0)
            for k in parent['optimizer_state']['state']:
                for name in parent['optimizer_state']['state'][k]:
                    torch.testing.assert_close(parent['optimizer_state']['state'][k][name],
                                               migrated['optimizer_state']['state'][k][name],atol=0,rtol=0)
            fast_main(base+opts+['--output',str(root/'fast'),'--epochs','2','--resume'])
            fast_main(base+opts+['--output',str(root/'replay'),'--epochs','2','--legacy-prefix-epochs','1'])
            a=torch.load(root/'fast/last.pt',weights_only=True)
            b=torch.load(root/'replay/last.pt',weights_only=True)
            for k in a['model_state']:
                torch.testing.assert_close(a['model_state'][k],b['model_state'][k],atol=0,rtol=0)
            control=base.copy();control[-1]='matched-original'
            fast_main(control+opts+['--output',str(root/'control'),'--epochs','2','--legacy-prefix-epochs','1'])
            compare_runs(root/'control',root/'fast',root/'comparison')
            # Changing scientific settings is not a valid migration.
            with self.assertRaisesRegex(ValueError,'configuration mismatch'):
                fast_main(base+opts+['--output',str(root/'bad'),'--epochs','2','--lr','.002',
                                    '--migrate-from',str(root/'legacy/last.pt')])

    def test_sparse_evaluation_does_not_change_final_training_weights(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);fixture(root)
            base=['--dataset',str(root/'dataset'),'--actual-run',str(root/'actual'),
                  '--device','cpu','--layers','2','--kernel-size','3','--subset','matched-original',
                  '--batch-size','2','--cache','cpu','--epochs','3']
            fast_main(base+['--output',str(root/'dense'),'--eval-every','1'])
            fast_main(base+['--output',str(root/'sparse'),'--eval-every','5'])
            a=torch.load(root/'dense/final.pt',weights_only=True)
            b=torch.load(root/'sparse/final.pt',weights_only=True)
            for k in a['model_state']:
                torch.testing.assert_close(a['model_state'][k],b['model_state'][k],atol=0,rtol=0)
            history=json.loads((root/'sparse/training.json').read_text())
            self.assertIsNone(history[1]['train_mse_normalized'])
            self.assertIsNotNone(history[-1]['train_mse_normalized'])

    def test_optimized_resume_exact_and_refuses_code_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);fixture(root)
            base=['--dataset',str(root/'dataset'),'--actual-run',str(root/'actual'),
                  '--device','cpu','--layers','2','--kernel-size','3','--subset','matched-original',
                  '--batch-size','2','--cache','cpu']
            fast_main(base+['--output',str(root/'a'),'--epochs','1'])
            fast_main(base+['--output',str(root/'a'),'--epochs','3','--resume'])
            fast_main(base+['--output',str(root/'b'),'--epochs','3'])
            a=torch.load(root/'a/final.pt',weights_only=True)
            b=torch.load(root/'b/final.pt',weights_only=True)
            for k in a['model_state']:
                torch.testing.assert_close(a['model_state'][k],b['model_state'][k],atol=0,rtol=0)
            saved=torch.load(root/'a/last.pt',weights_only=True)
            saved['contract']['source_sha256']['synthetic_fast.py']='tampered'
            torch.save(saved,root/'a/last.pt')
            with self.assertRaisesRegex(ValueError,'identical'):
                fast_main(base+['--output',str(root/'a'),'--epochs','4','--resume'])

    def test_process_identity_and_roi_energy_guarantee(self):
        root=Path('/tmp/example-experiment').resolve()
        self.assertTrue(training_command_matches(f'python scripts/train_synthetic.py --output {root}',root))
        self.assertFalse(training_command_matches(f'python scripts/other.py --output {root}',root))
        self.assertFalse(training_command_matches('python scripts/train_synthetic.py --output /tmp/other',root))
        y,x=np.mgrid[:60,:70]
        fields=np.stack([np.exp(-((x-20)**2+(y-30)**2)/100),np.exp(-((x-40)**2+(y-10)**2)/50)])
        y0,y1,x0,x1=energy_box(fields)
        for a in fields:
            self.assertGreaterEqual(float((a[y0:y1,x0:x1]**2).sum()/(a*a).sum()),.99)


if __name__=='__main__':
    unittest.main()
