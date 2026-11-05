# Runs on the server, on CPU only, so the GPU job is untouched.
import sys, json
sys.path.insert(0, 'experiments/csc_lista_v4/scripts')
import numpy as np, torch
import conv_lista as cl  # frozen copy keeps its original name
torch.set_num_threads(6)
out = {}
g = torch.load('runs/csc_lista_v2/dictionary_strong/dictionary_0.3.pt', weights_only=True)
out['atoms'] = g['g'].numpy()
out['dict_history'] = np.array([[h['step'], h['relative_l2'], h['density']] for h in g['history']])
scale = json.load(open('runs/simulated_3_to_05_v3/manifest.json'))['normalization_ut']
out['scale'] = scale
with np.load('runs/combined_full_200/comparison/L1_C1_z03.0_to_z00.5.npz') as z:
    sharp, blurred = z['sharp'], z['blurred']
print(sharp.shape, blurred.shape, flush=True)
i = 2  # Bz
n = 512
# The 512 px window holding the most target energy, found with a box filter.
energy = torch.tensor(sharp[i] ** 2, dtype=torch.float64)[None, None]
sums = torch.nn.functional.avg_pool2d(energy, n, stride=8)[0, 0]
r, c = [int(v) * 8 for v in np.unravel_index(int(sums.argmax()), sums.shape)]
out['crop_origin'] = np.array([r, c])
print('crop', r, c, flush=True)
target = torch.tensor(sharp[i, r:r+n, c:c+n] / scale, dtype=torch.float32)
b = torch.tensor(blurred[i, r:r+n, c:c+n] / scale, dtype=torch.float32)
pad = 128
B = torch.nn.functional.pad(b[None], (pad,)*4)
out['target'], out['input'] = target.numpy(), b.numpy()
out['wiener'] = cl.Dataless(.1, 2.5, 0.03162277660168379)(B)[0, pad:-pad, pad:-pad].numpy()
canvas = n + 2*pad
for name, layers in (('layers_32', 32), ('layers_64', 64)):
    state = torch.load(f'runs/csc_lista_strict/{name}/best.pt', map_location='cpu', weights_only=True)
    print(name, 'best step', state['step'], state['score'], flush=True)
    m = cl.ConvLISTA(g['g'], .1, 2.5, canvas, pad, .3, layers, 155, chunk=8)
    m.load_state_dict(state['state']); m.eval()
    with torch.no_grad():
        z = m.codes(B); y = cl.Operator(m.atoms, chunk=8).forward(z)
    out[f'strict_{layers}'] = y[0, pad:-pad, pad:-pad].numpy()
    out[f'density_{layers}'] = float((z != 0).float().mean())
    out[f'theta_{layers}'] = m.theta.detach().numpy()
    out[f'w_norm_{layers}'] = m.w.detach().flatten(2).norm(dim=2).mean(1).numpy()
    if layers == 32:
        out['w_first'] = m.w[0, 0].detach().numpy(); out['w_last'] = m.w[-1, 0].detach().numpy()
        out['w_mid'] = m.w[15, 0].detach().numpy()
        start = cl.ConvLISTA(g['g'], .1, 2.5, canvas, pad, .3, 1, 155, chunk=8)
        out['w_start'] = start.w[0, 0].detach().numpy(); out['theta_start'] = float(start.theta[0])
        out['bound'] = float(start.bound_value)
np.savez_compressed('/tmp/booklet_export.npz', **out)
print('done', flush=True)
