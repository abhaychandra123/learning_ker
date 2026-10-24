"""Train the existing SharedConvLISTA on compact synthetic recipes.

One accumulated Adam update per epoch, matching compare_combined.py. Actual
COMSOL fields are evaluated only after training, never used to select a model.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MPLBACKEND', 'Agg')

import argparse
import json
import math
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from combined_lista import SharedConvLISTA
from compare_combined import progress, save_checkpoint, write_csv
from compare_lista import cpu_state
from synthetic_data import COMPONENTS, SyntheticDataset, digest, transfer, write_json
from synthetic_experiment import select_samples, verify_actual_run, field_metrics
from torch_inverse import PaddedFFT, synchronize


class RecipeBatches:
    """Bounded memory: one scalar pair and one operator cached on the device."""
    def __init__(self, dataset, indices, device):
        self.dataset, self.indices, self.device = dataset, indices, device
        self.key, self.operator = None, None

    def get_operator(self, shape, grid, dz):
        pad = self.dataset.manifest['boundary']['pad_pixels']
        key = (tuple(shape), grid['dx'], grid['dy'], dz, pad)
        if key != self.key:
            h = transfer(shape, grid['dx'], grid['dy'], dz, pad)
            self.operator = PaddedFFT(h, shape, pad).to(self.device)
            self.key = key
        return self.operator

    def __iter__(self):
        for i in self.indices:
            for component in COMPONENTS:
                # Fixed epoch=0 noise: both arms have a fixed, auditable dataset.
                item = self.dataset.component(i, component, normalized=True)
                meta = item['metadata']
                operator = self.get_operator(item['sharp'].shape, meta['grid'], meta['dz_um'])
                b = torch.tensor(item['blurred'], dtype=torch.float32, device=self.device)[None, None]
                truth = torch.tensor(item['sharp'], dtype=torch.float32, device=self.device)[None, None]
                yield b, truth, operator


def loss_pass(model, batches, backward=False):
    model.train(backward)
    count = 3*len(batches.indices)
    total = 0.
    with torch.set_grad_enabled(backward):
        for n, (b, truth, operator) in enumerate(batches, 1):
            loss = (model(b, operator)-truth).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite synthetic loss')
            total += float(loss.detach())/count
            if backward:
                (loss/count).backward()
            if n % 60 == 0 or n == count:
                print(f'  {n}/{count} scalar pairs ({"gradient" if backward else "evaluation"})', flush=True)
    return total


@torch.no_grad()
def evaluate_actual(model, dataset, actual_root, manifest, hashes, output, save_predictions=False):
    """Both new arms use identical analytic operators; historical model is contextual."""
    model.eval()
    batches = RecipeBatches(dataset, [], next(model.parameters()).device)
    rows = []
    if save_predictions:
        (output/'actual_predictions').mkdir(exist_ok=True)
    for pair in manifest['pairs']:
        name = f"comparison/{pair['id']}.npz"
        if digest(actual_root/name) != hashes[name]:
            raise ValueError('Actual archive changed during training')
        grid = manifest['geometries'][pair['geometry']]['grid']
        predictions = []
        with np.load(actual_root/name, allow_pickle=False) as a:
            sharp, blur, legacy = a['sharp'], a['blurred'], a['lista']
            operator = batches.get_operator(sharp.shape[-2:], grid, pair['dz_um'])
            for c, component in enumerate(COMPONENTS):
                b = torch.tensor(blur[c]/dataset.scale, device=batches.device, dtype=torch.float32)[None, None]
                prediction = model(b, operator)[0, 0].cpu().numpy()*dataset.scale
                if not np.isfinite(prediction).all():
                    raise FloatingPointError('Nonfinite actual COMSOL prediction')
                predictions.append(prediction)
                for method, field in [('synthetic_final', prediction), ('historical_lista', legacy[c])]:
                    rows.append(dict(pair_id=pair['id'], geometry=pair['geometry'], component=component,
                                     sharp_z_um=pair['sharp_z_um'], blur_z_um=pair['blur_z_um'],
                                     method=method, **field_metrics(field, sharp[c])))
        if save_predictions:
            np.savez_compressed(output/'actual_predictions'/f"{pair['id']}.npz",
                                lista=np.stack(predictions))
        write_csv(output/'actual_comsol.csv', rows)
        print(f"  Actual COMSOL evaluated: {pair['id']}", flush=True)
    summary = []
    for method in ('synthetic_final', 'historical_lista'):
        chosen = [r for r in rows if r['method'] == method]
        means = {}
        for key in ('rmse_uT', 'nrmse_range', 'relative_l2', 'amplitude_slope'):
            values = [r[key] for r in chosen if r[key] is not None]
            means['mean_'+key] = float(np.mean(values)) if values else None
        summary.append(dict(method=method, component_pairs=len(chosen), **means))
    write_json(output/'actual_summary.json', dict(
        checkpoint='final.pt', summary=summary,
        interpretation='Same geometries and sharp targets as training; actual COMSOL inputs only used after training. '
                       'Historical LISTA used joint fitted operators/pad 32; new arms use analytical operators/dataset pad. '
                       'Historical comparison is contextual, not a data-count-only ablation.'))


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
    p.add_argument('--checkpoint-layers', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--save-predictions', action='store_true')
    args = p.parse_args(argv)
    if (args.epochs < 0 or args.layers < 1 or args.kernel_size < 1 or args.kernel_size % 2 != 1
            or not math.isfinite(args.lr) or args.lr <= 0
            or not math.isfinite(args.lambda_ista) or args.lambda_ista < 0):
        p.error('Invalid numerical configuration')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable; run inside a GPU allocation or on the dedicated GPU server')
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
    config = {k: v for k, v in vars(args).items()
              if k not in ('dataset', 'actual_run', 'output', 'resume', 'epochs')}
    contract = dict(config=config, dataset_sha256=digest(args.dataset/'manifest.json'),
                    actual_sha256=actual_hashes,
                    source_sha256={x.name: digest(x) for x in sorted(Path(__file__).parent.glob('*.py'))})
    model_config = dict(lam=args.lambda_ista/dataset.scale, layers=args.layers,
                        kernel_size=args.kernel_size, initial='blurred', checkpoint_layers=args.checkpoint_layers)
    model = SharedConvLISTA(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    batches = RecipeBatches(dataset, indices, device)
    environment = dict(python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__),
                       device=str(device), cuda=torch.version.cuda,
                       gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                       hostname=platform.node(), slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    if args.resume:
        saved = torch.load(args.output/'last.pt', map_location='cpu', weights_only=True)
        if saved['contract'] != contract or saved['scale_uT'] != dataset.scale:
            raise ValueError('Resume requires identical data, code and configuration')
        if args.epochs < saved['epoch']:
            raise ValueError('Requested epochs precede checkpoint')
        model.load_state_dict(saved['model_state'])
        optimizer.load_state_dict(saved['optimizer_state'])
        history, start = saved['training'], saved['epoch']+1
        best_loss, best_epoch, best_state = saved['best_loss'], saved['best_epoch'], saved['best_state']
        previous = saved['training_seconds']
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        history, start, previous = [], 0, 0.
        best_loss, best_epoch, best_state = float('inf'), 0, cpu_state(model)
        write_json(args.output/'setup.json', dict(contract=contract, model_config=model_config,
            scale_uT=dataset.scale, environment=environment, vector_pairs=len(indices),
            component_pairs=3*len(indices), sample_ids=[dataset.samples[i]['id'] for i in indices],
            pad=dataset.manifest['boundary']['pad_pixels'], optimizer_updates_per_epoch=1,
            selection='Primary: final epoch. Secondary best checkpoint: training MSE only. No actual-input selection.'))
    print(f'{args.subset}: {len(indices)} vector / {3*len(indices)} scalar pairs; '
          f'{args.epochs} Adam updates; {sum(x.numel() for x in model.parameters())} parameters', flush=True)
    began = perf_counter()
    try:
        write_json(args.output/'status.json', dict(status='training', requested_epochs=args.epochs,
                   completed_epochs=start-1, environment=environment))
        for epoch in range(start, args.epochs+1):
            if epoch:
                optimizer.zero_grad(set_to_none=True)
                with progress(f'Epoch {epoch}/{args.epochs}: training'):
                    loss_pass(model, batches, backward=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                model.project_parameters()
            with progress(f'Epoch {epoch}/{args.epochs}: synthetic training loss'):
                loss = loss_pass(model, batches)
            if loss < best_loss:
                best_loss, best_epoch, best_state = loss, epoch, cpu_state(model)
            synchronize(device)
            seconds = previous+perf_counter()-began
            history.append(dict(epoch=epoch, train_mse_normalized=loss, training_seconds=seconds))
            saved = dict(contract=contract, model_config=model_config, scale_uT=dataset.scale,
                         model_state=cpu_state(model), optimizer_state=optimizer.state_dict(), epoch=epoch,
                         best_loss=best_loss, best_epoch=best_epoch, best_state=best_state,
                         training=history, training_seconds=seconds)
            save_checkpoint(args.output/'last.pt', saved)
            write_json(args.output/'training.json', history)
            write_json(args.output/'status.json', dict(status='training', requested_epochs=args.epochs,
                       completed_epochs=epoch, training_seconds=seconds))
            print(f'Epoch {epoch}: MSE={loss:.8g}; elapsed={seconds:.1f}s', flush=True)
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


if __name__ == '__main__':
    main()
