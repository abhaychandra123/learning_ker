"""Generate original analytical teaching figures; these are not measured results."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
    'axes.spines.right': False, 'savefig.bbox': 'tight', 'pdf.fonttype': 42})
colors = ['#146C78', '#16324F', '#B07924', '#8E4261']
f = np.linspace(0, 1.5, 500)
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
for gap,c in zip([0.5,1.,2.5,5.],colors):
    H=np.exp(-2*np.pi*f*gap)
    axes[0].semilogy(f,H,color=c,label=f'{gap:g} µm')
    axes[1].semilogy(f,1/H,color=c,label=f'{gap:g} µm')
smallest=float(np.exp(-2*np.pi*f.max()*5.))
axes[0].set(title='Upward propagation',ylabel='Transfer amplitude H (dimensionless)',ylim=(smallest*.8,1.2))
axes[1].set(title='Unregularized inverse',ylabel='Noise gain 1/H (dimensionless)',ylim=(.8,1.2/smallest))
for ax in axes:
    ax.set_xlabel('Spatial frequency (cycles/µm)')
    ax.grid(alpha=.2)
    ax.legend(title='Propagation gap',fontsize=8)
fig.tight_layout()
fig.savefig(OUT/'propagation_and_inverse.pdf')
plt.close(fig)

fig,ax=plt.subplots(figsize=(8.2,3.6))
for delta,c in zip([.1,.25,.5],colors):
    ax.plot(f,np.exp(2*np.pi*f*delta),color=c,label=f'Height overestimate {delta:g} µm')
ax.set(xlabel='Spatial frequency (cycles/µm)',ylabel='Recovered-amplitude multiplier',ylim=(.8,115))
ax.set_yscale('log')
ax.axhline(1,color='#88959B',linewidth=.8)
ax.grid(alpha=.2)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT/'height_sensitivity.pdf')
plt.close(fig)

# Ideal point-dipole source: analytically exact, intentionally unlike finite-width coils.
x=np.linspace(-8,8,257); xx,yy=np.meshgrid(x,x)
mu0=4*np.pi*1e-7; moment=1e-15
heights=[.5,1.,3.,5.]
fields=[]
for h in heights:
    rx,ry,rz=xx*1e-6,yy*1e-6,h*1e-6
    r2=rx*rx+ry*ry+rz*rz
    bz=mu0*moment/(4*np.pi)*(3*rz*rz/r2**2.5-1/r2**1.5)*1e6
    fields.append(bz)
fig,axes=plt.subplots(1,4,figsize=(9.2,3.1),layout='constrained')
vmax=max(float(np.abs(b).max()) for b in fields)
for ax,h,b in zip(axes,heights,fields):
    im=ax.imshow(b,origin='lower',extent=[-8,8,-8,8],cmap='RdBu_r',
                 norm=SymLogNorm(linthresh=.1,vmin=-vmax,vmax=vmax,base=10))
    ax.set(title=f'z = {h:g} µm\npeak {b.max():.1f} µT',xlabel='x (µm)')
    ax.set_xticks([-8,0,8]); ax.set_yticks([-8,0,8])
axes[0].set_ylabel('y (µm)')
fig.colorbar(im,ax=list(axes),shrink=.75,label='Bz (µT), shared symmetric-log scale')
fig.savefig(OUT/'dipole_height_demo.pdf')
plt.close(fig)
print('Generated 3 original analytical figures.')
