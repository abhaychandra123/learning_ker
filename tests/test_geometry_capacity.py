"""Capacity-model equivalence, checkpoint gradients, selection and experiment integrity."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from combined_lista import SharedConvLISTA
from geometry_lista import UntiedConvLISTA, build_model
from synthetic_fast import RealAnalyticFFT
from train_geometry_capacity import main, select_clean
from summarize_geometry_capacity import summarize
from test_synthetic_training import fixture


class CapacityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(7)

    def test_parameter_counts_and_tied_gradient_is_sum_of_untied_gradients(self):
        for sharing,layers,count in [('tied',16,163),('tied',32,163),('untied',16,2608),('untied',32,5216)]:
            model=build_model(dict(sharing=sharing,layers=layers,kernel_size=9,lam=.02))
            self.assertEqual(sum(t.numel() for t in model.parameters()),count)
        tied=SharedConvLISTA(.02,layers=3,kernel_size=3)
        with torch.no_grad():
            tied.input_correction.weight.normal_(0,.003)
            tied.state_correction.weight.normal_(0,.003)
        untied=UntiedConvLISTA(.02,layers=3,kernel_size=3)
        with torch.no_grad():
            for k in range(3):
                untied.input_corrections[k].weight.copy_(tied.input_correction.weight)
                untied.state_corrections[k].weight.copy_(tied.state_correction.weight)
        op=RealAnalyticFFT((9,11),dict(dx=.2,dy=.3),.4,3)
        b=torch.randn(2,1,9,11)
        x,y=tied(b,op),untied(b,op)
        torch.testing.assert_close(x,y)
        x.square().mean().backward(); y.square().mean().backward()
        torch.testing.assert_close(tied.input_correction.weight.grad,sum(m.weight.grad for m in untied.input_corrections))
        torch.testing.assert_close(tied.state_correction.weight.grad,sum(m.weight.grad for m in untied.state_corrections))
        torch.testing.assert_close(tied.threshold.grad,untied.thresholds.grad.sum())

    def test_checkpoint_backward_uses_correct_layer(self):
        plain=UntiedConvLISTA(.02,layers=4,kernel_size=3)
        with torch.no_grad():
            for p in plain.parameters(): p.add_(torch.randn_like(p)*.002)
        checkpointed=copy.deepcopy(plain); checkpointed.checkpoint_layers=True
        op=RealAnalyticFFT((9,11),dict(dx=.2,dy=.3),.4,3)
        b=torch.randn(2,1,9,11)
        x,y=plain(b,op),checkpointed(b,op)
        torch.testing.assert_close(x,y,atol=0,rtol=0)
        x.square().mean().backward(); y.square().mean().backward()
        for a,c in zip(plain.parameters(),checkpointed.parameters()):
            torch.testing.assert_close(a.grad,c.grad,atol=0,rtol=0)

    def base(self,root):
        ds,manifest=fixture(root)
        geometry=manifest['pairs'][0]['geometry']
        args=['--dataset',str(root/'dataset'),'--actual-run',str(root/'actual'),'--geometry',geometry,
              '--kernel-size','3','--device','cpu','--cache','cpu','--batch-size','2',
              '--eval-every','1','--threads','1']
        return ds,manifest,args

    def test_clean_selection_and_resume_exact(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp); ds,m,base=self.base(root)
            selected=select_clean(ds,m['pairs'][0]['geometry'])
            self.assertEqual(len(selected),2)
            self.assertTrue(all(ds.samples[i]['noise_percent']==0 and ds.samples[i]['rotation_degrees']==0 for i in selected))
            with self.assertRaises(ValueError): select_clean(ds,'missing')
            with self.assertRaises(ValueError): select_clean(ds,m['pairs'][0]['geometry'],[999])
            args=base+['--sharing','untied','--layers','2']
            main(args+['--output',str(root/'direct'),'--epochs','2'])
            main(args+['--output',str(root/'resumed'),'--epochs','1'])
            main(args+['--output',str(root/'resumed'),'--epochs','2','--resume'])
            a=torch.load(root/'direct/last.pt',weights_only=True)
            b=torch.load(root/'resumed/last.pt',weights_only=True)
            for key in a['model_state']:
                torch.testing.assert_close(a['model_state'][key],b['model_state'][key],atol=0,rtol=0)
            for key in a['optimizer_state']['state']:
                for item in a['optimizer_state']['state'][key]:
                    torch.testing.assert_close(a['optimizer_state']['state'][key][item],b['optimizer_state']['state'][key][item],atol=0,rtol=0)
            self.assertTrue(any(r['parameter_update_l2']>0 for r in a['history'] if r['epoch']))
            with self.assertRaisesRegex(ValueError,'identical'):
                main(args+['--output',str(root/'resumed'),'--epochs','3','--resume','--lr','.002'])

    def test_actual_inputs_do_not_affect_training_weights(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp); _,m,base=self.base(root)
            args=base+['--sharing','untied','--layers','2','--epochs','1']
            main(args+['--output',str(root/'first')])
            path=root/'actual/comparison'/f"{m['pairs'][0]['id']}.npz"
            with np.load(path) as a: values={k:a[k] for k in a.files}
            values['blurred']*=3; np.savez(path,**values)
            main(args+['--output',str(root/'second')])
            a=torch.load(root/'first/final.pt',weights_only=True)
            b=torch.load(root/'second/final.pt',weights_only=True)
            for key in a['model_state']:
                torch.testing.assert_close(a['model_state'][key],b['model_state'][key],atol=0,rtol=0)

    def test_four_arm_summary_and_incomplete_rejection(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp); _,_,base=self.base(root)
            for sharing in ('tied','untied'):
                for layers in (16,32):
                    main(base+['--output',str(root/f'{sharing}_{layers}'),'--sharing',sharing,
                              '--layers',str(layers),'--epochs','1'])
            results=summarize(root)
            self.assertEqual(len(results),4)
            status=root/'untied_32/status.json'
            data=json.loads(status.read_text()); data['status']='paused_at_epoch_boundary'
            status.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'not completed'): summarize(root)


if __name__=='__main__':
    unittest.main()
