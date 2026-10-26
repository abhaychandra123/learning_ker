"""One-geometry, clean-height capacity experiment; no changes to prior trainers."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('MPLBACKEND','Agg')
import argparse
import json
import math
import platform
import signal
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from compare_combined import progress, save_checkpoint, write_csv
from compare_lista import cpu_state
from geometry_lista import build_model
from synthetic_data import COMPONENTS, SyntheticDataset, digest, write_json
from synthetic_experiment import field_metrics, verify_actual_run
from synthetic_fast import CachedBatches, RealAnalyticFFT, cached_loss, tune_batches
from torch_inverse import synchronize


def select_clean(dataset, geometry, heights=None):
    if geometry not in dataset.references:
        raise ValueError('Unknown geometry')
    selected = [i for i,s in enumerate(dataset.samples) if s['geometry']==geometry
                and s['rotation_degrees']==0 and s['noise_percent']==0 and s['realization']==0]
    if any(dataset.samples[i]['split']!='train' for i in selected):
        raise ValueError('Selected geometry must be a training geometry')
    if heights:
        selected = [i for i in selected if any(math.isclose(dataset.samples[i]['target_z_um'],h,abs_tol=1e-9) for h in heights)]
        if any(sum(math.isclose(dataset.samples[i]['target_z_um'],h,abs_tol=1e-9) for i in selected)!=1 for h in heights):
            raise ValueError('Each requested height must match exactly one clean recipe')
    selected.sort(key=lambda i:dataset.samples[i]['target_z_um'])
    if not selected or len({dataset.samples[i]['target_z_um'] for i in selected}) != len(selected):
        raise ValueError('Empty or ambiguous clean height selection')
    return selected


@torch.no_grad()
def synthetic_scores(model,batches,dataset,indices):
    model.eval()
    records = [(dataset.samples[i],c) for i in indices for c in COMPONENTS]
    rows=[]; offset=0
    for b,truth,op in batches:
        prediction = model(b,op).cpu().numpy()[:,0]*dataset.scale
        targets = truth.cpu().numpy()[:,0]*dataset.scale
        for pred,target in zip(prediction,targets):
            sample,component=records[offset]; offset+=1
            rows.append(dict(sample_id=sample['id'],geometry=sample['geometry'],component=component,
                input_z_um=sample['target_z_um'],target_z_um=sample['reference_z_um'],
                **field_metrics(pred,target)))
    return rows


@torch.no_grad()
def actual_scores(model,dataset,root,manifest,hashes,geometry,output):
    model.eval(); device=next(model.parameters()).device
    rows=[]; predictions=output/'actual_predictions'; predictions.mkdir(exist_ok=True)
    for pair in manifest['pairs']:
        if pair['geometry']!=geometry: continue
        name=f"comparison/{pair['id']}.npz"
        if digest(root/name)!=hashes[name]: raise ValueError('Actual archive changed')
        grid=manifest['geometries'][geometry]['grid']; values=[]
        op=RealAnalyticFFT((grid['ny'],grid['nx']),grid,pair['dz_um'],dataset.manifest['boundary']['pad_pixels']).to(device)
        with np.load(root/name,allow_pickle=False) as a:
            for c,component in enumerate(COMPONENTS):
                b=torch.tensor(a['blurred'][c]/dataset.scale,device=device)[None,None]
                pred=model(b,op)[0,0].cpu().numpy()*dataset.scale
                if not np.isfinite(pred).all(): raise FloatingPointError('Nonfinite prediction')
                values.append(pred)
                rows.append(dict(pair_id=pair['id'],geometry=geometry,component=component,
                    input_z_um=pair['blur_z_um'],target_z_um=pair['sharp_z_um'],**field_metrics(pred,a['sharp'][c])))
        np.savez_compressed(predictions/f"{pair['id']}.npz",lista=np.stack(values))
        print('  Evaluated '+pair['id'],flush=True)
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--actual-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--geometry',default='L1_C1')
    p.add_argument('--heights',type=float,nargs='+',help='Optional absolute input heights, for a separate small pilot')
    p.add_argument('--sharing',choices=('tied','untied'),required=True)
    p.add_argument('--layers',type=int,default=16)
    p.add_argument('--kernel-size',type=int,default=9)
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--lambda-ista',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--cache',choices=('auto','cpu','device'),default='auto')
    p.add_argument('--batch-size',type=int,default=None)
    p.add_argument('--checkpoint-mode',choices=('on','off'),default=None)
    p.add_argument('--autotune',action='store_true')
    p.add_argument('--eval-every',type=int,default=20)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--no-actual-evaluation',action='store_true',help='Pilot only; suite requires final actual evaluation')
    a=p.parse_args(argv)
    if (a.epochs<1 or a.layers<1 or a.kernel_size<1 or a.kernel_size%2!=1 or a.eval_every<1 or a.threads<1
        or (a.batch_size is not None and a.batch_size<1) or not math.isfinite(a.lr) or a.lr<=0
        or not math.isfinite(a.lambda_ista) or a.lambda_ista<0
        or (a.heights and (len(set(a.heights))!=len(a.heights) or not all(math.isfinite(h) for h in a.heights)))):
        p.error('Invalid numerical configuration')
    if a.autotune and (a.resume or a.device!='cuda'): p.error('Autotune requires a fresh CUDA run')
    if a.device=='cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable')
    a.dataset,a.actual_run,a.output=(v.resolve() for v in (a.dataset,a.actual_run,a.output))
    if a.output.exists() and not a.resume: raise FileExistsError('Choose fresh output or --resume')
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    ds=SyntheticDataset(a.dataset)
    with progress('Verifying actual archives and synthetic reference identity'):
        manifest,hashes=verify_actual_run(a.actual_run,ds)
    indices=select_clean(ds,a.geometry,a.heights)
    prior=torch.load(a.output/'last.pt',map_location='cpu',weights_only=True) if a.resume else None
    if prior:
        cfg=prior['contract']['config']
        if a.batch_size is None: a.batch_size=cfg['batch_size']
        if a.checkpoint_mode is None: a.checkpoint_mode=cfg['checkpoint_mode']
        if a.epochs<prior['epoch']: raise ValueError('Epoch target precedes saved checkpoint')
    a.batch_size=a.batch_size or 3; a.checkpoint_mode=a.checkpoint_mode or 'off'
    model_config=dict(sharing=a.sharing,lam=a.lambda_ista/ds.scale,layers=a.layers,
        kernel_size=a.kernel_size,initial='blurred',checkpoint_layers=a.checkpoint_mode=='on')
    model=build_model(model_config).to(a.device)
    with progress('Caching clean geometry pairs'):
        batches=CachedBatches(ds,indices,a.device,a.batch_size,a.cache,'real')
    tuning=tune_batches(model,batches,candidates=(1,2,3)) if a.autotune else None
    if tuning:
        a.batch_size=tuning['chosen']['batch_size']
        a.checkpoint_mode='on' if tuning['chosen']['checkpoint_layers'] else 'off'
    batches.batch_size=a.batch_size
    model.checkpoint_layers=a.checkpoint_mode=='on'; model_config['checkpoint_layers']=model.checkpoint_layers
    config={k:v for k,v in vars(a).items() if k not in ('dataset','actual_run','output','resume','epochs','autotune')}
    contract=dict(config=config,dataset_sha256=digest(a.dataset/'manifest.json'),actual_sha256=hashes,
        sample_ids=[ds.samples[i]['id'] for i in indices],
        source_sha256={f.name:digest(f) for f in sorted(Path(__file__).parent.glob('*.py'))})
    optimizer=torch.optim.Adam(model.parameters(),lr=a.lr)
    history=[]; start=0; previous=0.
    if prior:
        if prior['contract']!=contract or prior['model_config']!=model_config or prior['scale_uT']!=ds.scale:
            raise ValueError('Resume requires identical data, code, model and configuration')
        model.load_state_dict(prior['model_state']); optimizer.load_state_dict(prior['optimizer_state'])
        history=prior['history']; start=prior['epoch']+1; previous=prior['training_seconds']
    environment=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,
        device=a.device,gpu=torch.cuda.get_device_name(0) if a.device=='cuda' else None,hostname=platform.node())
    count=sum(t.numel() for t in model.parameters())
    if not a.resume:
        a.output.mkdir(parents=True)
        write_json(a.output/'setup.json',dict(contract=contract,model_config=model_config,scale_uT=ds.scale,
            pad=ds.manifest['boundary']['pad_pixels'],parameter_count=count,vector_pairs=len(indices),
            component_pairs=batches.count,environment=environment,cache=batches.info,tuning=tuning,
            interpretation='Clean single-geometry fitting diagnostic; no independent validation. '
                'Primary endpoint final fixed update. Actual COMSOL is evaluated only after training.'))
    stop=[False]
    handler=signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    began=perf_counter()
    print(f'{a.geometry}: {len(indices)} clean heights / {batches.count} components; '
          f'{a.sharing} {a.layers} layers, {count} parameters; target {a.epochs} Adam updates',flush=True)
    try:
        for epoch in range(start,a.epochs+1):
            epoch_start=perf_counter(); pre=None; gradnorm=None; update=None
            if epoch:
                optimizer.zero_grad(set_to_none=True)
                before=[t.detach().clone() for t in model.parameters()]
                with progress(f'Update {epoch}/{a.epochs}: gradient'):
                    pre=cached_loss(model,batches,backward=True)
                gradnorm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
                optimizer.step(); model.project_parameters()
                update=float(torch.sqrt(sum((t.detach()-old).square().sum() for t,old in zip(model.parameters(),before))))
                if not math.isfinite(update): raise FloatingPointError('Nonfinite parameter update')
            measure=epoch==0 or epoch%a.eval_every==0 or epoch==a.epochs or stop[0]
            post=cached_loss(model,batches) if measure else None
            if measure:
                scores=synthetic_scores(model,batches,ds,indices)
                write_csv(a.output/f'synthetic_epoch_{epoch:04d}.csv',scores)
                if epoch==a.epochs: write_csv(a.output/'synthetic_final.csv',scores)
            thresholds=[t.detach().flatten() for n,t in model.named_parameters() if 'threshold' in n]
            theta=torch.cat(thresholds)*ds.scale
            synchronize(a.device); elapsed=previous+perf_counter()-began
            row=dict(epoch=epoch,pre_update_mse=pre,post_update_mse=post,gradient_norm_before_clip=gradnorm,
                gradient_clipped=gradnorm>1 if gradnorm is not None else None,parameter_update_l2=update,
                threshold_min_uT=float(theta.min()),threshold_max_uT=float(theta.max()),
                threshold_zero_count=int((theta==0).sum()),epoch_seconds=perf_counter()-epoch_start,training_seconds=elapsed)
            history.append(row)
            saved=dict(contract=contract,model_config=model_config,scale_uT=ds.scale,model_state=cpu_state(model),
                optimizer_state=optimizer.state_dict(),epoch=epoch,history=history,training_seconds=elapsed)
            save_checkpoint(a.output/'last.pt',saved)
            write_json(a.output/'training.json',history)
            write_json(a.output/'status.json',dict(status='training',completed_epochs=epoch,requested_epochs=a.epochs))
            print(json.dumps(row),flush=True)
            if stop[0]:
                write_json(a.output/'status.json',dict(status='paused_at_epoch_boundary',completed_epochs=epoch))
                return
        save_checkpoint(a.output/'final.pt',dict(contract=contract,model_config=model_config,scale_uT=ds.scale,
            model_state=cpu_state(model),epoch=a.epochs))
        if not a.no_actual_evaluation:
            write_json(a.output/'status.json',dict(status='evaluating_actual',completed_epochs=a.epochs))
            with progress('Final evaluation on actual COMSOL inputs'):
                rows=actual_scores(model,ds,a.actual_run,manifest,hashes,a.geometry,a.output)
            write_csv(a.output/'actual_final.csv',rows)
        write_json(a.output/'status.json',dict(status='complete',completed_epochs=a.epochs,
            actual_evaluated=not a.no_actual_evaluation,training_seconds=history[-1]['training_seconds']))
    except BaseException as exc:
        write_json(a.output/'status.json',dict(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
            error=str(exc),completed_epochs=history[-1]['epoch'] if history else -1))
        raise
    finally:
        signal.signal(signal.SIGTERM,handler)


if __name__=='__main__':
    main()
