"""Lost-band floor and naive spectral height estimate for every local COMSOL pair (run from repo root)."""
import numpy as np, glob, re, json
pad=128; dx=0.1
res=[]
for path in sorted(glob.glob('runs/combined_full_200/comparison/*.npz')):
    m=re.match(r'.*/(.+)_z(\d+\.\d)_to_z(\d+\.\d)\.npz',path); g,zi,zt=m[1],float(m[2]),float(m[3])
    d=np.load(path); S=d['sharp'].astype(float); B=d['blurred'].astype(float)
    n=S.shape[1]+2*pad
    k=2*np.pi*np.hypot(np.fft.fftfreq(n,dx)[:,None],np.fft.fftfreq(n,dx)[None,:])
    H=np.exp(-k*(zi-zt)); row={'geom':g,'z_in':zi,'z_tg':zt}
    fl=[];kcs=[];blind=[]
    for c in range(3):
        FX=np.fft.fft2(np.pad(S[c],pad)); FB=np.fft.fft2(np.pad(B[c],pad))
        sig=np.abs(H*FX)**2; r=np.abs(FB-H*FX)**2; sx=np.abs(FX)**2
        kc=None
        for a in np.arange(0,8,0.05):
            mm=(k>=a)&(k<a+0.05)
            if sig[mm].sum()<r[mm].sum(): kc=a;break
        fl.append(np.sqrt(sx[k>=kc].sum()/sx.sum())); kcs.append(kc)
        # blind standoff-from-spectrum: band chosen from data only (power > 100x high-k floor)
        if c==2:
            P=np.abs(FB)**2; floor=np.median(P[(k>8)&(k<15)])
            edges=np.arange(0.1,6,0.05); cen=0.5*(edges[1:]+edges[:-1])
            p=np.array([P[(k>=a)&(k<b)].mean() for a,b in zip(edges[:-1],edges[1:])])
            ok=np.where(p<100*floor)[0]; kmax=cen[ok[0]] if len(ok) else 6
            mm=(cen>=0.2)&(cen<=kmax)
            est=-np.polyfit(cen[mm],np.log(p[mm]),1)[0]/2
            row['blind_z_est_Bz']=round(float(est),2); row['blind_kmax']=round(float(kmax),2)
    row['floor_pct']=[round(100*v,1) for v in fl]; row['floor_mean']=round(100*float(np.mean(fl)),1)
    row['kc']=[round(v,2) for v in kcs]
    res.append(row)
res.sort(key=lambda r:(r['geom'],r['z_in']))
print("geom    z_in->z_tg  kc(Bx,By,Bz)        floor%(Bx,By,Bz)     mean   | blind Bz z-estimate (band k<=)")
for r in res:
    print(f"{r['geom']:7s} {r['z_in']:4.1f}->{r['z_tg']:.1f}  {str(r['kc']):20s} {str(r['floor_pct']):22s} {r['floor_mean']:5.1f}  | {r['blind_z_est_Bz']:5.2f} ({r['blind_kmax']})")
