import numpy as np, sys, glob, re, os
d=0.1; N=1400
def load(f):
    a=np.loadtxt(f,comments='%',delimiter=',')
    nx=len(np.unique(a[:,0])); B=a[:,2:5].reshape(-1,nx,3).transpose(2,0,1)
    Bx,By,Bz=B; return [-By,Bx,Bz], a[0,:2], a[-1,:2]
pad=300; k=2*np.pi*np.fft.fftfreq(N+2*pad,d); KV,KU=np.meshgrid(k,k,indexing='ij'); K=np.hypot(KU,KV)
win=np.outer(np.hanning(N),np.hanning(N)); F=lambda a: np.fft.fft2(np.pad(a*win,pad))
bands=[(0.04,0.08),(0.08,0.12),(0.12,0.18),(0.18,0.25),(0.25,0.35),(0.35,0.5)]
def rate(B1,B2,dz):
    F1=[F(c) for c in B1]; F2=[F(c) for c in B2]; out=[]
    for lo,hi in bands:
        m=(K>=lo)&(K<hi); num=sum(np.vdot(a[m],b[m]) for a,b in zip(F1,F2)).real; den=sum(np.vdot(a[m],a[m]) for a in F1).real
        kc=np.sqrt(np.mean(K[m]**2)); r=-np.log(num/den)/dz; out.append((kc,r,np.sqrt(max(r*r-kc*kc,0))))
    return out
base=os.path.expanduser('~/learning_ker/data/data2/IC_4um_Z%sum_600uA_0p1PR__MF.csv')
D={z:load(base%z) for z in ('0p5','3p0','6p0','10p0')}
print('IC_4um window', D['0p5'][1], D['0p5'][2])
for z1,z2,dz in (('0p5','3p0',2.5),('3p0','6p0',3.0),('6p0','10p0',4.0)):
    r=rate(D[z1][0],D[z2][0],dz)
    print(f' {z1}->{z2}: '+'  '.join(f'k={kc:.3f} rate={x:.3f} k0={k0:.3f}' for kc,x,k0 in r))
    Bu,Bv,Bz=D[z1][0]; print('   wall normal check (|Bu| at u-walls, |Bv| at v-walls)/rms:', 
      round(float(np.sqrt(np.mean(Bu[:,[0,-1]]**2))/np.sqrt(np.mean(Bu**2+Bv**2+Bz**2))),3), round(float(np.sqrt(np.mean(Bv[[0,-1],:]**2))/np.sqrt(np.mean(Bu**2+Bv**2+Bz**2))),3), 'mean Bz/rms', round(float(Bz.mean()/np.sqrt(np.mean(Bz**2))),4))
