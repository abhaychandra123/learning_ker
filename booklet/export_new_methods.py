# Runs on the server: the finished Job 3, FISTA-Net and FISTA-300 on the same crop as the first export.
import sys
sys.path.insert(0, 'experiments/fista_net/scripts')
import numpy as np, torch
import conv_lista as cl  # frozen copy keeps its original name
import fista_net as fn
device = 'cuda'
g = torch.load('runs/csc_lista_v2/dictionary_strong/dictionary_0.3.pt', weights_only=True)['g'].to(device)
old = np.load('/tmp/booklet_export.npz')
b = torch.tensor(old['input'], device=device)
pad, n = 128, b.shape[-1]
B = torch.nn.functional.pad(b[None], (pad,) * 4)
cut = lambda y: y[0, pad:-pad, pad:-pad].cpu().numpy()
out = {}
with torch.no_grad():
    h = cl.transfer(B.shape[-1], .1, 2.5, B.dtype, B.device)
    x = cl.code_ista(B, cl.Operator(g, h, 8), .009, 300, pad, True)
    out['fista300'] = cut(cl.Operator(g, chunk=8).forward(x))
    out['density_fista300'] = float((x != 0).float().mean())
    state = torch.load('runs/csc_lista_lfista/layers_32/best.pt', map_location=device, weights_only=True)
    print('lfista best', state['step'], state['score'], flush=True)
    m = cl.ConvLISTA(g, .1, 2.5, n + 2 * pad, pad, .3, 32, 155, chunk=8, momentum=True).to(device)
    m.load_state_dict(state['state']); m.eval()
    z = m.codes(B)
    out['lfista'] = cut(cl.Operator(m.atoms, chunk=8).forward(z))
    out['density_lfista'] = float((z != 0).float().mean())
    out['beta'] = m.beta.cpu().numpy(); out['theta_lfista'] = m.theta.cpu().numpy()
    state = torch.load('runs/fista_net/run_1/best.pt', map_location=device, weights_only=True)
    print('fista_net best', state['step'], state['score'], flush=True)
    f = fn.FISTANet(.1, 2.5).to(device); f.load_state_dict(state['state']); f.eval()
    out['fista_net'] = cut(f(B)[0])
    out['fista_net_start'] = cut(f.initial(B))
    out['scalars'] = np.array([float(getattr(f, k)) for k in ('w1', 'w2', 'w3', 'c1', 'c2', 'c3')])
np.savez_compressed('/tmp/booklet_new.npz', **out)
