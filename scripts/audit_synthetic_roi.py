"""Measure field support and crop-induced forward error without changing data."""
import argparse
from pathlib import Path
import numpy as np
from synthetic_data import SyntheticDataset, propagate, write_json


def energy_box(fields, retained=.99):
    """Union of per-component marginal boxes; >=retained squared-field energy."""
    if not 0<retained<1:
        raise ValueError('Retained energy must lie between zero and one')
    boxes=[]
    for field in fields:
        energy=np.asarray(field,dtype=np.float64)**2
        if not energy.sum():
            continue
        bounds=[]
        for axis in (1,0):
            marginal=energy.sum(axis=axis)
            cumulative=np.cumsum(marginal)/marginal.sum()
            tail=(1-retained)/4
            bounds.extend([int(np.searchsorted(cumulative,tail)),
                           min(len(marginal),int(np.searchsorted(cumulative,1-tail))+1)])
        boxes.append(bounds)
    if not boxes:
        raise ValueError('No nonzero field energy')
    return [min(b[0] for b in boxes),max(b[1] for b in boxes),
            min(b[2] for b in boxes),max(b[3] for b in boxes)]


def audit(dataset,output):
    ds=SyntheticDataset(dataset)
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    rows=[]
    fig,axes=plt.subplots(len(ds.references),2,figsize=(10,4*len(ds.references)),squeeze=False)
    for row,(geometry,reference) in enumerate(ds.references.items()):
        entry=ds.manifest['geometries'][geometry];grid=entry['grid']
        dz=entry['max_existing_z_um']-entry['reference_z_um']
        pad=ds.manifest['boundary']['pad_pixels']
        clean=np.stack([propagate(x,dz,grid['dx'],grid['dy'],pad) for x in reference])
        a,b=energy_box(reference),energy_box(clean)
        y0,y1,x0,x1=a
        crop=[max(0,y0-128),min(grid['ny'],y1+128),max(0,x0-128),min(grid['nx'],x1+128)]
        cy0,cy1,cx0,cx1=crop
        cropped=np.stack([propagate(x[cy0:cy1,cx0:cx1],dz,grid['dx'],grid['dy'],pad) for x in reference])
        errors=[]
        for c in range(3):
            target=clean[c,y0:y1,x0:x1]
            prediction=cropped[c,y0-cy0:y1-cy0,x0-cx0:x1-cx0]
            denom=float(np.linalg.norm(target))
            errors.append(float(np.linalg.norm(prediction-target)/denom) if denom else None)
        rows.append(dict(geometry=geometry,reference_energy_box_yxyx=a,highest_clean_energy_box_yxyx=b,
                         proposed_crop_with_128px_halo_yxyx=crop,max_dz_um=dz,
                         crop_forward_relative_l2_on_reference_box=errors))
        for col,(fields,box,title) in enumerate([(reference,a,'Reference'),(clean,b,'Highest propagated field')]):
            ax=axes[row,col]
            magnitude=np.sqrt(np.sum(fields.astype(float)**2,axis=0))
            # Log display reveals weak tails; boxes use linear squared-field energy.
            im=ax.imshow(np.log10(np.maximum(magnitude,float(magnitude.max())*1e-5)),origin='lower',cmap='magma')
            ya,yb,xa,xb=box
            ax.add_patch(Rectangle((xa,ya),xb-xa,yb-ya,fill=False,edgecolor='cyan',linewidth=1.5))
            ax.set_title(f'{geometry}: {title}\n99% energy box {yb-ya} × {xb-xa} px')
            ax.set_xlabel('x pixel');ax.set_ylabel('y pixel')
            fig.colorbar(im,ax=ax,label='log10 vector magnitude [µT]',shrink=.8)
        print(rows[-1],flush=True)
    fig.suptitle('Field coverage audit: cyan boxes retain at least 99% energy in every component',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(output/'field_coverage.png',dpi=130);plt.close(fig)
    write_json(output/'roi_audit.json',dict(rows=rows,
        meaning='Squared magnetic-field energy, not current-support or reconstruction-error guarantee. '
                'Forward crop errors compare analytical crop propagation with full-plane propagation '
                'at highest standoff; no inverse training or cropped dataset has been performed.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.dataset,a.output)
