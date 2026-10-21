"""Frozen full-domain and region diagnostics. Saved reference only; no BP calls."""
import time
from pathlib import Path
import numpy as np
from polim_cpu.geometry_probe_v2.common import ROOT,arrays,read,write
from polim_cpu import packed_storage_v1 as st

SOURCE=ROOT/'polim_cpu/runs/v32r4-historical-1470-1570-v1'
MASKS=ROOT/'polim_cpu/runs/v38-large-step-phase2-v1/masks-1552.npz'
def reference(year):
    j=read(SOURCE/f'year-{year}.json');path=SOURCE/f'frame-{j["step"]:06d}'
    z,m=st.restore(path,read(SOURCE/'contract.json')['contract']);assert m['year']==year
    return z,m,path

def annual(root,year,H,u,ledger,z,bp):
    start=time.perf_counter();ref,rm,rp=reference(year);_,im,_=reference(1552)
    masks=arrays(MASKS);area=z['geo_dual_area'];mask=masks['evaluation'];assert mask.all()
    dh=H-ref['h'];surf=u.reshape(-1,2,2)[:,1,:];rs=ref['u'].reshape(-1,2,2)[:,1,:];du=surf-rs
    vr=float(area@ref['h']);exportref=rm['exported']-im['exported'];limit=max(abs(exportref)*.01,ledger['V0']*1e-5)
    ix=int(np.argmax(abs(dh)))
    out=dict(year=year,volume_relative=abs(float(area@H)-vr)/vr,h_rms=float(np.sqrt(np.sum(area*dh**2)/area.sum())),
        h_max=float(abs(dh).max()),velocity_relative=float(np.sqrt(np.sum(area[:,None]*du**2)/max(np.sum(area[:,None]*rs**2),1e-300))),
        export_difference=abs(ledger['exported']-exportref),export_limit=limit,export_reference=exportref,
        maximum_node=ix,maximum_xy_m=z['xy'][ix].tolist(),signed_H_difference_at_max=float(dh[ix]),
        candidate_H_at_max=float(H[ix]),reference_H_at_max=float(ref['h'][ix]),
        original_BP=bp,reference_checkpoint=str(rp),ledger=ledger,diagnostic_regions={})
    out['gates']=dict(volume=out['volume_relative']<=.005,h_rms=out['h_rms']<=.2,h_max=out['h_max']<=2.,
        velocity=out['velocity_relative']<=.02,export=out['export_difference']<=limit);out['passed']=all(out['gates'].values())
    for key in ['ice_edge','junction','original_fast']:
        m=masks[key];ids=np.flatnonzero(m);worst=int(ids[np.argmax(abs(dh[m]))])
        out['diagnostic_regions'][key]=dict(count=int(m.sum()),h_rms=float(np.sqrt(np.sum(area[m]*dh[m]**2)/area[m].sum())),
            h_max=float(abs(dh[m]).max()),maximum_node=worst,velocity_relative=float(np.sqrt(np.sum(area[m,None]*du[m]**2)/max(np.sum(area[m,None]*rs[m]**2),1e-300))))
    with (Path(root)/f'differences-{year}.npz').open('xb') as f:
        np.savez(f,xy=z['xy'],h_difference=dh,surface_velocity_difference=du,
            surface_speed_difference=np.linalg.norm(surf,axis=1)-np.linalg.norm(rs,axis=1),**masks)
    out['evaluation_seconds']=time.perf_counter()-start;write(Path(root)/f'annual-{year}.json',out)
    return out

def plots(root):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    start=time.perf_counter();paths=[]
    for p in sorted(Path(root).glob('differences-*.npz')):
        a=arrays(p);xy=a['xy']/1000;xy=xy-xy.min(axis=0);year=p.stem.split('-')[-1]
        fields=[(a['h_difference'],'Thickness difference (m)',True),
                (a['surface_velocity_difference'][:,0],'Surface u difference (m/yr)',True),
                (a['surface_velocity_difference'][:,1],'Surface v difference (m/yr)',True)]
        fig,axes=plt.subplots(3,4,figsize=(16,11),layout='constrained')
        for j,key in enumerate(['evaluation','ice_edge','junction','original_fast']):
            mask=a[key]
            for i,(field,label,_) in enumerate(fields):
                ax=axes[i,j];ax.scatter(xy[:,0],xy[:,1],s=1,c='#dddddd');bound=max(float(abs(field).max()),1e-12)
                sc=ax.scatter(xy[mask,0],xy[mask,1],c=field[mask],s=3,cmap='RdBu_r',vmin=-bound,vmax=bound)
                fig.colorbar(sc,ax=ax,label=label);ax.set_aspect('equal');ax.set_title(key);ax.set_xlabel('x (km)');ax.set_ylabel('y (km)')
        fig.suptitle(f'1552 start, fixed quarter steps | {year} | candidate minus retained v32')
        target=Path(root)/f'differences-{year}.png';fig.savefig(target,dpi=150);plt.close(fig);paths.append(str(target))
    return dict(paths=paths,seconds=time.perf_counter()-start,new_BP=0,new_tangent=0)
