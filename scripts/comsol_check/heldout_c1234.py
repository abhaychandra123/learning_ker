"""Held-out check: k0 = 0.10 (fitted on L1_C1) applied unchanged to the C1+C2+C3+C4 configuration."""
import numpy as np, sys
SP=sys.argv[1]; d=0.1; N=1400
c4=np.load(f"{SP}/C1234_sel.npz")
def frame(B): Bx,By,Bz=B.astype(float); return [-By,Bx,Bz]
def mirror(a,pu,pv): a=np.concatenate([a,pu*a[:,::-1]],axis=1); return np.concatenate([a,pv*a[::-1,:]],axis=0)
k=2*np.pi*np.fft.fftfreq(2*N,d); KV,KU=np.meshgrid(k,k,indexing='ij'); K2=np.hypot(KU,KV)
k1=2*np.pi*np.fft.fftfreq(N,d); a,b=np.meshgrid(k1,k1,indexing='ij'); Kl=np.hypot(a,b)<1.0
rel=lambda p,t: np.linalg.norm(p-t)/np.linalg.norm(t)
lp=lambda x: np.real(np.fft.ifft2(np.fft.fft2(x)*Kl))
def prop(src,dz,k0):
    f=np.exp(-np.sqrt(K2**2+k0**2)*dz); E=[mirror(src[0],-1,1),mirror(src[1],1,-1),mirror(src[2],1,1)]
    return [np.real(np.fft.ifft2(np.fft.fft2(e)*f))[:N,:N] for e in E]
src=frame(c4["L1_C1_C2_C3_C4_z0.5"])
for z in (3.0,10.0,12.5):
    t=frame(c4[f"L1_C1_C2_C3_C4_z{z}"])
    for k0 in (0.0,0.10):
        p=prop(src,z-0.5,k0)
        print(f"C1-C4 0.5->{z:4.1f}  box k0={k0:.2f}: full "+"/".join(f"{100*rel(p[c],t[c]):5.1f}" for c in range(3))
              +"   low-passed k<1 "+"/".join(f"{100*rel(lp(p[c]),lp(t[c])):5.1f}" for c in range(3)))
