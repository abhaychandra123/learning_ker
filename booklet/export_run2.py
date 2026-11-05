# Runs on the server: FISTA-Net run 2 on the same crop as the first export.
import sys
sys.path.insert(0, 'experiments/fista_net_code_start/scripts')
import numpy as np, torch
import fista_net as fn
device = 'cuda'
old = np.load('/tmp/booklet_export.npz')
b = torch.tensor(old['input'], device=device)
pad = 128
B = torch.nn.functional.pad(b[None], (pad,) * 4)
state = torch.load('runs/fista_net/run_2/best.pt', map_location=device, weights_only=True)
print('best', state['step'], state['score'], flush=True)
f = fn.FISTANet(.1, 2.5, start='code').to(device)
f.load_state_dict(state['state'])
f.eval()
with torch.no_grad():
    out = dict(fista_net_2=f(B)[0][0, pad:-pad, pad:-pad].cpu().numpy(),
               mu=np.array([float(f.schedules(k)[0]) for k in range(1, 8)]),
               theta=np.array([float(f.schedules(k)[1]) for k in range(1, 8)]),
               rho=np.array([float(f.schedules(k)[2]) for k in range(1, 8)]))
np.savez_compressed('/tmp/booklet_run2.npz', **out)
