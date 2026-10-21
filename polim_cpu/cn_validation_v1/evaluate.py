"""Frozen-area endpoint comparison against retained v32; array-only."""
import numpy as np
from pathlib import Path
from polim_cpu.geometry_probe_v2.common import ROOT,arrays,write
from polim_cpu.geometry_window_v1.evaluation import reference

def pair(H,u,export,Hr,ur,exportr,area,V0,masks):
    dh=H-Hr;v=u.reshape(-1,2,2)[:,1,:];vr=ur.reshape(-1,2,2)[:,1,:];du=v-vr
    volume=float(area@H);volume_ref=float(area@Hr);vn=float(np.sqrt(np.sum(area[:,None]*vr**2)))
    va=float(np.sqrt(np.sum(area[:,None]*du**2)));limit=max(abs(exportr)*.01,V0*1e-5)
    out=dict(volume_relative=abs(volume-volume_ref)/abs(volume_ref) if abs(volume_ref)>1e-12 else None,
        volume_absolute=volume-volume_ref,h_rms=float(np.sqrt(np.sum(area*dh**2)/area.sum())),h_max=float(abs(dh).max()),
        velocity_relative=va/vn if vn>1e-12 else None,velocity_absolute_L2=va,export_difference=abs(export-exportr),
        signed_export_difference=export-exportr,export_limit=limit,export_reference=exportr,
        maximum_node=int(np.argmax(abs(dh))),regions={})
    out['gates']=dict(volume=out['volume_relative'] is not None and out['volume_relative']<=.005,h_rms=out['h_rms']<=.2,
        h_max=out['h_max']<=2,velocity=out['velocity_relative'] is not None and out['velocity_relative']<=.02,export=out['export_difference']<=limit)
    out['passed']=all(out['gates'].values())
    for key,m in masks.items():
        if m.dtype==bool and m.shape==H.shape:
            out['regions'][key]=dict(H_RMS=float(np.sqrt(np.sum(area[m]*dh[m]**2)/area[m].sum())),H_max=float(abs(dh[m]).max()),
                signed_mean=float(area[m]@dh[m]/area[m].sum()),signed_min=float(dh[m].min()),signed_max=float(dh[m].max()))
    return out,dh,du

def annual(root,year,H,u,ledger,z,window_start,mask_year,bp):
    ref,rm,_=reference(year);_,im,_=reference(window_start)
    masks=arrays(ROOT/f'polim_cpu/runs/v38-large-step-phase2-v1/masks-{mask_year}.npz')
    out,dh,du=pair(H,u,ledger['exported'],ref['h'],ref['u'],rm['exported']-im['exported'],z['geo_dual_area'],ledger['V0'],masks)
    out.update(year=year,window_start=window_start,ledger=ledger,original_BP=bp)
    with (Path(root)/f'differences-{year}.npz').open('xb') as f:np.savez_compressed(f,H_difference=dh,surface_velocity_difference=du,xy=z['xy'],**masks)
    write(Path(root)/f'annual-{year}.json',out);return out
