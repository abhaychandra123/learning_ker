"""Read-only checks behind FABLE_REVIEW.md (10 Nov 2026). NumPy only, no training, about 2 minutes on a laptop.

Run from the repo root: python3 scripts/review_checks.py
Uses the two local COMSOL pairs at 3 -> 0.5 um and the saved MAGIC-UNet run 2 and FISTA-300 predictions in
runs/magic_unet/run2_server_outputs. Part 1: scores, amplitude slope, residual correlation, error split at the
SNR = 1 crossing, Maxwell consistency, implied height of the MAGIC current. Part 2: forward-operator mismatch,
data consistency of each prediction, one Tikhonov correction of the MAGIC output, frequency blends, scores
against a low-passed target. Everything touching COMSOL here is a diagnostic on test data, not a result.
"""
# ---------------------------------------------------------------- part 1
import numpy as np
R='runs/'
M=np.load(R+'magic_unet/run2_server_outputs/comsol_3_to_05_predictions.npz')
F=np.load(R+'magic_unet/run2_server_outputs/fista300_comsol_predictions.npz')
dx=0.1; pad=128; MU0=4e-7*np.pi*1e6
def grid(n):
    wy=2*np.pi*np.fft.fftfreq(n,dx)[:,None]; wx=2*np.pi*np.fft.fftfreq(n,dx)[None,:]
    return wx,wy,np.hypot(wx,wy)
def rel(a,b): return np.linalg.norm(a-b)/np.linalg.norm(b)
for g in ('L1_C1','IC_4um'):
    d=np.load(R+f'combined_full_200/comparison/{g}_z03.0_to_z00.5.npz')
    S=d['sharp'].astype(float); B=d['blurred'].astype(float)
    pm=M[g+'_B'].astype(float); pf=F[g].astype(float)
    n=S.shape[1]+2*pad; wx,wy,k=grid(n); H=np.exp(-k*2.5)
    P=lambda a: np.fft.fft2(np.pad(a,((0,0),(pad,pad),(pad,pad))))
    FS,FB,FM,FF=P(S),P(B),P(pm),P(pf)
    print('=====',g)
    print('rel L2 magic',[round(rel(pm[c],S[c]),3) for c in range(3)],'mean',round(np.mean([rel(pm[c],S[c]) for c in range(3)]),4))
    print('rel L2 fista',[round(rel(pf[c],S[c]),3) for c in range(3)],'mean',round(np.mean([rel(pf[c],S[c]) for c in range(3)]),4))
    print('rel L2 input',[round(rel(B[c],S[c]),3) for c in range(3)])
    # interior (exclude 10 um border)
    b=100; I=(slice(None),slice(b,-b),slice(b,-b))
    print('interior rel L2 magic/fista',[round(rel(pm[c][b:-b,b:-b],S[c][b:-b,b:-b]),3) for c in range(3)],[round(rel(pf[c][b:-b,b:-b],S[c][b:-b,b:-b]),3) for c in range(3)])
    # amplitude slope
    sl=lambda p,t: float((p*t).sum()/(t*t).sum())
    print('slope magic',[round(sl(pm[c],S[c]),3) for c in range(3)],'fista',[round(sl(pf[c],S[c]),3) for c in range(3)])
    # residual correlation
    for c in range(3):
        rm,rf=pm[c]-S[c],pf[c]-S[c]
        cos=(rm*rf).sum()/np.linalg.norm(rm)/np.linalg.norm(rf)
        shared=np.sqrt(max((rm*rf).sum(),0))/np.linalg.norm(S[c])
        avg=rel(0.5*(pm[c]+pf[c]),S[c])
        print(f'  comp{c} residual cosine {cos:.3f} shared-residual norm/truth {shared:.3f} avg-of-two rel L2 {avg:.3f}')
    # band split using SNR=1 crossing (same rule as information_budget.py)
    for c in range(3):
        sig=np.abs(H*FS[c])**2; r=np.abs(FB[c]-H*FS[c])**2
        kc=None
        for a in np.arange(0,8,0.05):
            mm=(k>=a)&(k<a+0.05)
            if sig[mm].sum()<r[mm].sum(): kc=a;break
        tot=(np.abs(FS[c])**2).sum()
        lo=k<kc
        f=lambda X,m: np.sqrt((np.abs(X)**2)[m].sum()/tot)
        print(f'  comp{c} kc={kc:.2f} truth>kc {f(FS[c],~lo):.3f} | magic err in-band {f(FM[c]-FS[c],lo):.3f} lost-band {f(FM[c]-FS[c],~lo):.3f} | fista in {f(FF[c]-FS[c],lo):.3f} lost {f(FF[c]-FS[c],~lo):.3f} | pred energy>kc magic {f(FM[c],~lo):.3f} fista {f(FF[c],~lo):.3f}')
        # finer: lost band split kc..2kc and >2kc
        m1=(k>=kc)&(k<2*kc); m2=k>=2*kc
        print(f'      truth kc-2kc {f(FS[c],m1):.3f} >2kc {f(FS[c],m2):.3f}; magic err {f(FM[c]-FS[c],m1):.3f}/{f(FM[c]-FS[c],m2):.3f}; fista err {f(FF[c]-FS[c],m1):.3f}/{f(FF[c]-FS[c],m2):.3f}')
    # Maxwell consistency in the export frame: Bx=(wy/k)phi, By=-(wx/k)phi, Bz=i phi
    ks=k.copy(); ks[0,0]=1
    def incons(FX):
        phi=((wy/ks)*FX[0]-(wx/ks)*FX[1]-1j*FX[2])/2
        C=np.stack([(wy/ks)*phi,-(wx/ks)*phi,1j*phi]); C[:,0,0]=FX[:,0,0]
        return C
    for name,FX in (('target0.5',FS),('input3.0',FB),('magic',FM),('fista',FF)):
        C=incons(FX); res=np.sqrt((np.abs(FX-C)**2).sum((1,2))/(np.abs(FX)**2).sum((1,2)))
        allr=np.sqrt((np.abs(FX-C)**2).sum()/(np.abs(FX)**2).sum())
        print(f'  Maxwell-inconsistent fraction {name}: per comp {np.round(res,3)} all {allr:.3f}')
    # error of each method against the Maxwell-projected target
    C=incons(FS); 
    den=np.sqrt((np.abs(C)**2).sum((1,2)))
    print('  vs Maxwell-projected target: magic',np.round(np.sqrt((np.abs(FM-C)**2).sum((1,2)))/den,3),'fista',np.round(np.sqrt((np.abs(FF-C)**2).sum((1,2)))/den,3))
    # inconsistent energy of target by band (Bz kc)
    inc=np.abs(FS-C)**2; 
    for a,bb in ((0,0.5),(0.5,1.1),(1.1,2.2),(2.2,5),(5,40)):
        m=(k>=a)&(k<bb); print(f'    band {a}-{bb} rad/um: target energy frac {np.sqrt((np.abs(FS)**2)[:,m].sum()/(np.abs(FS)**2).sum()):.3f} inconsistent/energy-in-band {np.sqrt(inc[:,m].sum()/(np.abs(FS)**2)[:,m].sum()):.3f}')
    # global implied height of MAGIC current
    J=M[g+'_J'].astype(float); J=np.kron(J,np.ones((1,2,2))); p2=700
    jx=np.pad(J[0],p2); jy=np.pad(J[1],p2); N=jx.shape[0]
    wx2=2*np.pi*np.fft.rfftfreq(N,dx)[None,:]; wy2=2*np.pi*np.fft.fftfreq(N,dx)[:,None]; k2=np.hypot(wx2,wy2); s2=k2.copy(); s2[0,0]=1
    fx,fy=np.fft.rfft2(jx),np.fft.rfft2(jy); fz=1j*(wy2*fx-wx2*fy)/s2
    out=[]
    for h in np.arange(1.5,5.01,0.25):
        dec=0.5*MU0*np.exp(-k2*h)
        f=np.stack([np.fft.irfft2(q*dec,s=(N,N))[p2:-p2,p2:-p2] for q in (fx,fy,fz)])
        out.append((h,np.mean([rel(f[c],B[c]) for c in range(3)])))
    print('  implied height scan (h, mean rel L2 to 3um input):',[(round(h,2),round(e,3)) for h,e in out])
    print('  best h',min(out,key=lambda t:t[1]))

# ---------------------------------------------------------------- part 2

R='runs/'
M=np.load(R+'magic_unet/run2_server_outputs/comsol_3_to_05_predictions.npz')
F=np.load(R+'magic_unet/run2_server_outputs/fista300_comsol_predictions.npz')
dx=0.1; pad=128
def rel(a,b): return np.linalg.norm(a-b)/np.linalg.norm(b)
for g in ('L1_C1','IC_4um'):
    d=np.load(R+f'combined_full_200/comparison/{g}_z03.0_to_z00.5.npz')
    S=d['sharp'].astype(float); B=d['blurred'].astype(float)
    pm=M[g+'_B'].astype(float); pf=F[g].astype(float)
    n=S.shape[1]+2*pad
    wy=2*np.pi*np.fft.fftfreq(n,dx)[:,None]; wx=2*np.pi*np.fft.fftfreq(n,dx)[None,:]; k=np.hypot(wx,wy)
    P=lambda a: np.fft.fft2(np.pad(a,((0,0),(pad,pad),(pad,pad))))
    U=lambda X: np.real(np.fft.ifft2(X))[:,pad:-pad,pad:-pad]
    FS,FB,FM,FF=P(S),P(B),P(pm),P(pf)
    print('=====',g)
    for k0 in (0.0,0.095):
        H=np.exp(-np.sqrt(k**2+k0**2)*2.5)
        fw=U(H*FS)
        sl=[float((fw[c]*B[c]).sum()/(fw[c]*fw[c]).sum()) for c in range(3)]
        print(f' forward 0.5->3 k0={k0}: rel L2 to input',[round(rel(fw[c],B[c]),3) for c in range(3)],'slope input/forward',np.round(sl,3))
        # band-limited (k<1 rad/um) forward mismatch
        lo=k<1.0
        e=[np.sqrt((np.abs(H*FS[c]-FB[c])**2)[lo].sum()/(np.abs(FB[c])**2)[lo].sum()) for c in range(3)]
        print('     in-band k<1 mismatch',np.round(e,3))
    H=np.exp(-k*2.5)
    # data consistency: how well does each prediction, re-blurred, explain the input
    for name,FX in (('magic',FM),('fista',FF),('truth',FS)):
        rb=U(H*FX); print(f' re-blurred {name} vs 3um input rel L2',[round(rel(rb[c],B[c]),3) for c in range(3)])
    # one Tikhonov data-consistency correction of MAGIC: x + H/(H^2+mu) (B - H x)
    for mu in (0.0316,0.01,0.003):
        X=FM+H/(H*H+mu)*(FB-H*FM); x=U(X)
        print(f' MAGIC + 1 Tikhonov correction mu={mu}: rel L2',[round(rel(x[c],S[c]),3) for c in range(3)],'mean',round(np.mean([rel(x[c],S[c]) for c in range(3)]),4))
    # frequency blend: FISTA below kx, MAGIC above
    for kx_ in (0.6,0.8,1.0,1.2):
        w=1/(1+np.exp((k-kx_)/0.05)); x=U(w*FF+(1-w)*FM)
        print(f' blend FISTA(k<{kx_}) + MAGIC(k>): rel L2',[round(rel(x[c],S[c]),3) for c in range(3)],'mean',round(np.mean([rel(x[c],S[c]) for c in range(3)]),4))
    # score against a band-limited target (k<2.2 rad/um), to remove mesh speckle from the metric
    lp=k<2.2; St=U(FS*lp)
    print(' vs target low-passed at 2.2 rad/um: magic',[round(rel(U(FM*lp)[c],St[c]),3) for c in range(3)],'fista',[round(rel(U(FF*lp)[c],St[c]),3) for c in range(3)],'| removed target energy',[round(rel(St[c],S[c]),3) for c in range(3)])
    # Wiener oracle and input on the same low-passed target
