"""Cached fixed synthetic data, measured batching, and real analytical FFTs.

The NumPy generator remains the authority for fields and fixed noise. No crops,
resizing, mixed precision, or minibatch optimizer steps are introduced.
"""
from collections import OrderedDict
import gc
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from synthetic_data import COMPONENTS, rotated_grid, rotation_component, transfer
from torch_inverse import PaddedFFT, synchronize


class RealAnalyticFFT(torch.nn.Module):
    """Same zero-pad/convolve/crop analytical operator, half-spectrum storage."""
    def __init__(self, shape, grid, dz, pad, dtype=torch.float32):
        super().__init__()
        self.shape, self.pad = tuple(shape), pad
        self.padded_shape = tuple(n+2*pad for n in shape)
        # Generate exactly the same transfer as the original full-complex path.
        h = transfer(shape, grid['dx'], grid['dy'], dz, pad)
        self.bound = float(np.max(np.abs(h)**2))
        self.register_buffer('h', torch.tensor(h[:, :self.padded_shape[1]//2+1].copy(), dtype=dtype))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 1 or tuple(x.shape[-2:]) != self.shape:
            raise ValueError('Input shape does not match analytical operator')
        p = self.pad
        a = F.pad(x, (p,p,p,p)) if p else x
        y = torch.fft.irfft2(torch.fft.rfft2(a)*self.h, s=self.padded_shape)
        return y[..., p:p+self.shape[0], p:p+self.shape[1]]

    def adjoint(self, x):
        # Analytic transfer is real/even; symmetric zero-padding/cropping.
        return self(x)


class CachedBatches:
    """Cache inputs once, share targets, and batch only identical operators.

    Original scalar order is retained. Device cache eliminates generation and
    host transfers during epochs. CPU fallback uses pinned storage plus one
    asynchronous transfer stream when the compute device is CUDA.
    """
    def __init__(self, dataset, indices, device, batch_size=1, cache='auto', fft='real'):
        self.dataset, self.indices = dataset, indices
        self.device, self.batch_size = torch.device(device), batch_size
        self.groups, self.targets, self.operators = [], {}, {}
        self.count = 3*len(indices)
        pad = dataset.manifest['boundary']['pad_pixels']
        target_keys = set()
        input_bytes = target_bytes = 0
        for i in indices:
            s = dataset.samples[i]
            grid = rotated_grid(dataset.manifest['geometries'][s['geometry']]['grid'], s['rotation_degrees'])
            shape = (grid['ny'], grid['nx'])
            key = (shape, grid['dx'], grid['dy'], s['dz_um'], pad)
            if not self.groups or self.groups[-1]['key'] != key:
                self.groups.append(dict(key=key, grid=grid, records=[]))
            for c in range(3):
                target = (s['geometry'], s['rotation_degrees'], c)
                self.groups[-1]['records'].append((i,c,target))
                input_bytes += 4*int(np.prod(shape))
                if target not in target_keys:
                    target_bytes += 4*int(np.prod(shape)); target_keys.add(target)
        estimated = input_bytes+target_bytes
        if cache not in ('auto','device','cpu') or fft not in ('full','real'):
            raise ValueError('Unknown cache/FFT backend')
        if cache == 'auto':
            cache = ('device' if self.device.type == 'cuda'
                     and estimated < .45*torch.cuda.mem_get_info(self.device)[0] else 'cpu')
        self.cache_device = self.device if cache == 'device' else torch.device('cpu')
        self.pinned = self.cache_device.type == 'cpu' and self.device.type == 'cuda'
        self.info = dict(cache=cache, input_bytes=input_bytes, shared_target_bytes=target_bytes,
                         scalar_pairs=self.count, fft=fft, propagation_calls=0)
        clean_cache = OrderedDict()
        began = perf_counter(); loaded = 0
        print(f'Caching {self.count} inputs: {estimated/2**30:.2f} GiB on {self.cache_device}', flush=True)
        for group in self.groups:
            key = group['key']; shape, dx, dy, dz, pad = key
            if key not in self.operators:
                op = (RealAnalyticFFT(shape, group['grid'], dz, pad) if fft == 'real' else
                      PaddedFFT(transfer(shape, dx, dy, dz, pad), shape, pad))
                self.operators[key] = op.to(self.device)
            group['operator'] = self.operators[key]
            group['inputs'] = torch.empty((len(group['records']),1,*shape), dtype=torch.float32,
                                           device=self.cache_device, pin_memory=self.pinned)
            group['targets'] = []
            for n, (i,c,target) in enumerate(group['records']):
                s = dataset.samples[i]
                source_c, _ = rotation_component(c, s['rotation_degrees'])
                clean_key = (s['geometry'],source_c,s['dz_um'])
                if clean_key in clean_cache:
                    dataset._clean_cache = (clean_key,clean_cache.pop(clean_key))
                else:
                    self.info['propagation_calls'] += 1
                item = dataset.component(i,c,normalized=True)
                clean_cache[clean_key] = dataset._clean_cache[1]
                while len(clean_cache)>3:
                    clean_cache.popitem(last=False)
                group['inputs'][n,0].copy_(torch.from_numpy(np.ascontiguousarray(item['blurred'])))
                if target not in self.targets:
                    t = torch.empty((1,*shape),dtype=torch.float32,device=self.cache_device,pin_memory=self.pinned)
                    t[0].copy_(torch.from_numpy(np.ascontiguousarray(item['sharp'])))
                    self.targets[target] = t
                group['targets'].append(self.targets[target])
                loaded += 1
            if loaded % 60 < len(group['records']) or loaded == self.count:
                print(f'  cached {loaded}/{self.count}',flush=True)
        clean_cache.clear(); dataset._clean_cache = None
        synchronize(self.device)
        self.info['preparation_seconds'] = perf_counter()-began
        self.info['operator_count'] = len(self.operators)

    def raw_batches(self):
        for group in self.groups:
            for start in range(0,len(group['records']),self.batch_size):
                end = start+self.batch_size
                b = group['inputs'][start:end]
                targets = group['targets'][start:end]
                if self.pinned:
                    truth = torch.empty((len(targets),*targets[0].shape),dtype=torch.float32,pin_memory=True)
                    torch.stack(targets,out=truth)
                else:
                    truth = torch.stack(targets)
                yield b, truth, group['operator']

    def __iter__(self):
        if not self.pinned:
            yield from self.raw_batches()
            return
        stream = torch.cuda.Stream(device=self.device)
        iterator = iter(self.raw_batches())
        def preload():
            try:
                b,t,op = next(iterator)
            except StopIteration:
                return None
            with torch.cuda.stream(stream):
                return b.to(self.device,non_blocking=True),t.to(self.device,non_blocking=True),op
        pending = preload()
        while pending is not None:
            torch.cuda.current_stream(self.device).wait_stream(stream)
            current = pending
            current[0].record_stream(torch.cuda.current_stream(self.device))
            current[1].record_stream(torch.cuda.current_stream(self.device))
            pending = preload()
            yield current


def cached_loss(model, batches, backward=False):
    model.train(backward)
    total = torch.zeros((),device=batches.device)
    done = 0; next_report = 120
    with torch.set_grad_enabled(backward):
        for b,truth,op in batches:
            loss = (model(b,op)-truth).square().mean()
            weight = len(b)/batches.count
            total.add_(loss.detach()*weight)
            if backward:
                (loss*weight).backward()
            done += len(b)
            if done >= next_report or done == batches.count:
                print(f'  {done}/{batches.count} cached components',flush=True)
                next_report = done+120
    result = float(total)
    if not np.isfinite(result):
        raise FloatingPointError('Nonfinite cached loss; optimizer has not stepped')
    return result


def tune_batches(model, batches, candidates=(1,2,4,8), repeats=2):
    """Time forward+backward without optimizer updates; retain memory headroom."""
    if batches.device.type != 'cuda':
        raise ValueError('Autotuning requires CUDA; choose batch/checkpoint explicitly on CPU')
    group = max(batches.groups,key=lambda g: np.prod(g['key'][0])*len(g['records']))
    original_checkpoint = model.checkpoint_layers
    rows = []
    free,_ = torch.cuda.mem_get_info(batches.device)
    limit = torch.cuda.memory_allocated(batches.device)+.8*free
    def probe(size):
        b = group['inputs'][:size].to(batches.device)
        truth = torch.stack(group['targets'][:size]).to(batches.device)
        def step():
            model.zero_grad(set_to_none=True)
            loss = (model(b,group['operator'])-truth).square().mean()
            loss.backward()
            return loss
        step(); synchronize(batches.device)
        elapsed=[]
        for _ in range(repeats):
            began=perf_counter(); loss=step(); synchronize(batches.device)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite tuning loss')
            elapsed.append(perf_counter()-began)
        return float(np.median(elapsed))
    try:
        model.train()
        for checkpoint in (False,True):
            model.checkpoint_layers = checkpoint
            for size in candidates:
                if size>len(group['records']):
                    continue
                print(f'Benchmark batch={size}, checkpoint={checkpoint}',flush=True)
                torch.cuda.reset_peak_memory_stats(batches.device)
                try:
                    seconds=probe(size)
                    peak=torch.cuda.max_memory_allocated(batches.device)
                    row=dict(batch_size=size,checkpoint_layers=checkpoint,seconds=seconds,
                             components_per_second=size/seconds,peak_allocated_bytes=peak,
                             accepted=peak<=limit)
                except torch.cuda.OutOfMemoryError:
                    row=dict(batch_size=size,checkpoint_layers=checkpoint,accepted=False,error='CUDA OOM')
                finally:
                    model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
                rows.append(row); print(row,flush=True)
    finally:
        model.checkpoint_layers=original_checkpoint
        model.zero_grad(set_to_none=True)
    valid=[r for r in rows if r['accepted']]
    if not valid:
        raise RuntimeError('No safe CUDA benchmark candidate; use CPU cache or smaller workload')
    chosen=max(valid,key=lambda r:r['components_per_second'])
    return dict(chosen=chosen,trials=rows,
                note='Representative cached group; forward/backward only, no optimizer updates. '
                     '80% of initially free memory budget; actual epoch timing is authoritative.')
