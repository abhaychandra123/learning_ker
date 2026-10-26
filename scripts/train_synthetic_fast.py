"""Optimized full-field synthetic training; see README_SYNTHETIC_OPTIMIZATION.md."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MPLBACKEND','Agg')
import argparse
import math
import platform
import signal
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from combined_lista import SharedConvLISTA
from compare_combined import progress,save_checkpoint
from compare_lista import cpu_state
from synthetic_data import SyntheticDataset,digest,write_json
from synthetic_experiment import select_samples,verify_actual_run
from train_synthetic import RecipeBatches,loss_pass,evaluate_actual
from synthetic_fast import CachedBatches,cached_loss,tune_batches
from torch_inverse import synchronize


def validate_migration(saved,dataset,actual_hashes,args):
    """Accept only the audited legacy trainer and unchanged scientific inputs."""
    old=saved['contract']
    expected_driver='a4c4bce5034ed08d21fd7a92a2a833fed0021dfb7d264522f3d53e1e98edfcff'
    if old['source_sha256'].get('train_synthetic.py')!=expected_driver:
        raise ValueError('Migration requires the audited legacy synthetic trainer')
    for name,checksum in old['source_sha256'].items():
        path=Path(__file__).with_name(name)
        if not path.is_file() or digest(path)!=checksum:
            raise ValueError('Legacy source changed: '+name)
    if (old['dataset_sha256']!=digest(dataset.root/'manifest.json')
            or old['actual_sha256']!=actual_hashes or saved['scale_uT']!=dataset.scale):
        raise ValueError('Migration data/provenance/normalization mismatch')
    for key in ('subset','device','layers','kernel_size','lr','lambda_ista','seed','save_predictions'):
        if old['config'][key]!=getattr(args,key):
            raise ValueError('Migration scientific configuration mismatch: '+key)
    if not saved['model_config']['checkpoint_layers']:
        raise ValueError('Audited legacy prefix expects checkpointing enabled')
    if saved['epoch']<0 or saved['epoch']>args.epochs:
        raise ValueError('Invalid migration epoch')
    expected=dict(lam=args.lambda_ista/dataset.scale,layers=args.layers,kernel_size=args.kernel_size,
                  initial='blurred',checkpoint_layers=True)
    if saved['model_config']!=expected:
        raise ValueError('Migration model configuration mismatch')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--actual-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--subset', choices=('full', 'matched-original'), default='full')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--layers', type=int, default=16)
    p.add_argument('--kernel-size', type=int, default=9)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lambda-ista', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--checkpoint-mode', choices=('on','off'), default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--cache', choices=('auto','device','cpu'), default='auto')
    p.add_argument('--fft', choices=('full','real'), default='real')
    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--autotune', action='store_true')
    p.add_argument('--migrate-from', type=Path)
    p.add_argument('--legacy-prefix-epochs', type=int, default=None)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--save-predictions', action='store_true')
    args = p.parse_args(argv)
    if (args.eval_every < 1 or (args.batch_size is not None and args.batch_size < 1)
            or (args.legacy_prefix_epochs is not None and args.legacy_prefix_epochs < 0) or args.epochs < 0 or args.layers < 1 or args.kernel_size < 1 or args.kernel_size % 2 != 1
            or not math.isfinite(args.lr) or args.lr <= 0
            or not math.isfinite(args.lambda_ista) or args.lambda_ista < 0):
        p.error('Invalid numerical configuration')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable; run inside a GPU allocation or on the dedicated GPU server')
    if args.resume and args.migrate_from:
        p.error('Choose resume or migration, not both')
    if args.resume and args.autotune:
        p.error('Resume reuses measured settings; do not retune an existing run')
    device = torch.device(args.device)
    args.dataset, args.actual_run, args.output = (v.resolve() for v in (args.dataset, args.actual_run, args.output))
    if args.output.exists() and not args.resume:
        raise FileExistsError('Use a fresh output directory or --resume')
    dataset = SyntheticDataset(args.dataset)
    with progress('Verifying actual COMSOL archives and matching references'):
        manifest, actual_hashes = verify_actual_run(args.actual_run, dataset)
    indices = select_samples(dataset, manifest, args.subset)
    # Validate control coverage even for the full arm, before a long run.
    select_samples(dataset, manifest, 'matched-original')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parent = None
    if args.migrate_from:
        parent_hash = digest(args.migrate_from)
        parent = torch.load(args.migrate_from, map_location='cpu', weights_only=True)
        validate_migration(parent, dataset, actual_hashes, args)
        if digest(args.migrate_from) != parent_hash:
            raise ValueError('Source checkpoint changed while reading; stop old run before migration')
        if args.legacy_prefix_epochs not in (None,parent['epoch']):
            raise ValueError('Legacy prefix must match migration epoch')
        args.legacy_prefix_epochs = parent['epoch']
    resume_saved = torch.load(args.output/'last.pt',map_location='cpu',weights_only=True) if args.resume else None
    if resume_saved:
        prior = resume_saved['contract']['config']
        if args.batch_size is None: args.batch_size = prior['batch_size']
        if args.checkpoint_mode is None: args.checkpoint_mode = 'on' if prior['checkpoint_layers'] else 'off'
        if args.legacy_prefix_epochs is None: args.legacy_prefix_epochs = prior['legacy_prefix_epochs']
    args.legacy_prefix_epochs = args.legacy_prefix_epochs or 0
    if args.legacy_prefix_epochs > args.epochs:
        raise ValueError('Requested epochs precede legacy prefix')
    args.batch_size = args.batch_size or 4
    args.checkpoint_layers = args.checkpoint_mode == 'on'
    model_config = dict(lam=args.lambda_ista/dataset.scale, layers=args.layers,
                        kernel_size=args.kernel_size, initial='blurred', checkpoint_layers=args.checkpoint_layers)
    model = SharedConvLISTA(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    source = resume_saved or parent
    if source:
        model.load_state_dict(source['model_state'])
        optimizer.load_state_dict(source['optimizer_state'])
        history, start = list(source['training']), source['epoch']+1
        best_loss, best_epoch, best_state = source['best_loss'],source['best_epoch'],source['best_state']
        previous = source['training_seconds']
        if args.epochs < source['epoch']:
            raise ValueError('Requested epochs precede checkpoint')
    else:
        history, start, previous = [],0,0.
        best_loss,best_epoch,best_state = float('inf'),0,cpu_state(model)
    # Cache initialization is outside timed epochs and reported separately.
    with progress('Preparing fixed training inputs once'):
        batches = CachedBatches(dataset,indices,device,args.batch_size,args.cache,args.fft)
    tuning = None
    if args.autotune:
        tuning = tune_batches(model,batches)
        args.batch_size = tuning['chosen']['batch_size']
        args.checkpoint_layers = tuning['chosen']['checkpoint_layers']
    batches.batch_size = args.batch_size
    model.checkpoint_layers = args.checkpoint_layers
    model_config['checkpoint_layers'] = args.checkpoint_layers
    config = {k:v for k,v in vars(args).items() if k not in
              ('dataset','actual_run','output','resume','epochs','migrate_from','autotune','checkpoint_mode')}
    contract = dict(config=config,dataset_sha256=digest(args.dataset/'manifest.json'),
                    actual_sha256=actual_hashes,
                    source_sha256={x.name:digest(x) for x in sorted(Path(__file__).parent.glob('*.py'))})
    if resume_saved and (resume_saved['contract'] != contract or resume_saved['scale_uT'] != dataset.scale):
        raise ValueError('Resume requires identical data, code and configuration')
    environment = dict(python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__),
                       device=str(device),cuda=torch.version.cuda,
                       gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
                       hostname=platform.node(),slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    if not args.resume:
        args.output.mkdir(parents=True,exist_ok=False)
        write_json(args.output/'setup.json',dict(contract=contract,model_config=model_config,
                   scale_uT=dataset.scale,environment=environment,vector_pairs=len(indices),
                   component_pairs=3*len(indices),sample_ids=[dataset.samples[i]['id'] for i in indices],
                   pad=dataset.manifest['boundary']['pad_pixels'],optimizer_updates_per_epoch=1,
                   cache=batches.info,tuning=tuning,legacy_prefix_epochs=args.legacy_prefix_epochs,
                   selection='Primary final epoch. Best training checkpoint only among measured epochs; '
                             'inherited legacy best retained. No actual COMSOL selection.'))
        if parent:
            write_json(args.output/'migration.json',dict(source=str(args.migrate_from.resolve()),
                       sha256=parent_hash,epoch=parent['epoch'],parent_contract=parent['contract'],
                       parent_checkpoint_layers=parent['model_config']['checkpoint_layers'],
                       note='Model and Adam moments retained. Full field, padding, noise, scale and '
                            'one-update-per-epoch objective retained. Batching/FFT arithmetic may differ in rounding.'))
            # Preserve the migrated state even if the next epoch is interrupted.
            save_checkpoint(args.output/'last.pt',dict(parent,contract=contract,model_config=model_config))
    legacy_batches = RecipeBatches(dataset,indices,device) if args.legacy_prefix_epochs and start<=args.legacy_prefix_epochs else None
    print(f'{args.subset}: {len(indices)} vector / {3*len(indices)} scalar pairs; '
          f'{args.epochs} Adam updates; {sum(x.numel() for x in model.parameters())} parameters', flush=True)
    began = perf_counter()
    stop_requested = [False]
    previous_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.__setitem__(0,True))
    try:
        write_json(args.output/'status.json', dict(status='training', requested_epochs=args.epochs,
                   completed_epochs=start-1, environment=environment))
        for epoch in range(start,args.epochs+1):
            epoch_start = perf_counter()
            legacy = args.legacy_prefix_epochs>0 and epoch<=args.legacy_prefix_epochs
            model.checkpoint_layers = True if legacy else args.checkpoint_layers
            gradient_loss = None
            if epoch:
                optimizer.zero_grad(set_to_none=True)
                with progress(f'Epoch {epoch}/{args.epochs}: {"legacy prefix" if legacy else "cached"} training'):
                    gradient_loss = (loss_pass(model,legacy_batches,backward=True) if legacy else
                                     cached_loss(model,batches,backward=True))
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step(); model.project_parameters()
            synchronize(device)
            gradient_seconds = perf_counter()-epoch_start
            measure = legacy or epoch==0 or epoch%args.eval_every==0 or epoch==args.epochs or stop_requested[0]
            loss = None
            evaluation_start = perf_counter()
            if measure:
                with progress(f'Epoch {epoch}/{args.epochs}: measured training loss'):
                    loss = loss_pass(model,legacy_batches) if legacy else cached_loss(model,batches)
                if loss<best_loss:
                    best_loss,best_epoch,best_state=loss,epoch,cpu_state(model)
            synchronize(device)
            seconds=previous+perf_counter()-began
            history.append(dict(epoch=epoch,train_mse_normalized=loss,pre_update_mse_normalized=gradient_loss,
                                training_seconds=seconds,gradient_seconds=gradient_seconds,
                                evaluation_seconds=perf_counter()-evaluation_start,legacy_prefix=legacy))
            saved=dict(contract=contract,model_config=model_config,scale_uT=dataset.scale,
                       model_state=cpu_state(model),optimizer_state=optimizer.state_dict(),epoch=epoch,
                       best_loss=best_loss,best_epoch=best_epoch,best_state=best_state,
                       training=history,training_seconds=seconds)
            save_checkpoint(args.output/'last.pt',saved)
            write_json(args.output/'training.json',history)
            write_json(args.output/'status.json',dict(status='training',requested_epochs=args.epochs,
                       completed_epochs=epoch,training_seconds=seconds))
            print(f'Epoch {epoch}: pre-update MSE={gradient_loss}; post-update MSE={loss}; '
                  f'epoch={perf_counter()-epoch_start:.1f}s; total={seconds:.1f}s',flush=True)
            if stop_requested[0]:
                write_json(args.output/'status.json',dict(status='paused_at_epoch_boundary',completed_epochs=epoch))
                return
        model.checkpoint_layers=args.checkpoint_layers
        for name, state, epoch in [('final', cpu_state(model), args.epochs), ('best', best_state, best_epoch)]:
            save_checkpoint(args.output/f'{name}.pt', dict(model_config=model_config, model_state=state,
                            scale_uT=dataset.scale, epoch=epoch, contract=contract))
        write_json(args.output/'status.json', dict(status='evaluating_actual', completed_epochs=args.epochs))
        with progress('Evaluating final model on actual COMSOL inputs'):
            evaluate_actual(model, dataset, args.actual_run, manifest, actual_hashes,
                            args.output, args.save_predictions)
        write_json(args.output/'status.json', dict(status='complete', completed_epochs=args.epochs,
                   best_training_epoch=best_epoch, primary_checkpoint='final.pt',
                   component_pairs=3*len(indices), environment=environment))
    except BaseException as exc:
        write_json(args.output/'status.json', dict(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                   error=str(exc), completed_epochs=history[-1]['epoch'] if history else -1))
        raise
    finally:
        signal.signal(signal.SIGTERM,previous_handler)


if __name__ == '__main__':
    main()
