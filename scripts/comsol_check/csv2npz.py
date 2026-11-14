import numpy as np, sys, os, glob, re
out = sys.argv[1]; files = sys.argv[2:]
arrs = {}
for f in files:
    d = np.loadtxt(f, comments='%', delimiter=',', dtype=np.float64)
    x, y = d[:,0], d[:,1]
    ux, uy = np.unique(x), np.unique(y)
    nx, ny = len(ux), len(uy)
    assert nx*ny == len(d), (f, nx, ny, len(d))
    # check x fastest
    assert np.allclose(x[:nx], ux)
    B = d[:,2:5].reshape(ny, nx, 3).transpose(2,0,1).astype(np.float32)  # [comp,y,x], tesla
    m = re.search(r'Z([0-9p]+)um', os.path.basename(f)); z = float(m.group(1).replace('p','.'))
    tag = os.path.basename(f).split('_Z')[0] + '_z%.1f' % z
    arrs[tag] = B
    arrs['x'] = ux.astype(np.float32); arrs['y'] = uy.astype(np.float32)
    print(tag, B.shape, ux[0], ux[-1], uy[0], uy[-1], flush=True)
np.savez(out, **arrs)
