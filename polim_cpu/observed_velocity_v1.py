"""Read-only spatial comparison; no dynamics, calibration, or source writes."""
import json,sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.features import rasterize
from pyproj import Transformer
from scipy.ndimage import map_coordinates
import geopandas as gpd
from .v3.runner import PROJECT,digest,write_json
from .thermal_attribution_v1 import validate
from .spatial_pair_guard import require_paired_source
from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.paths import resolve

OUT=PROJECT/'polim_cpu/reports/observed-velocity-v1'
BASE=PROJECT/'Himalaya_greatGlacier/With_Rec_climate_organized'
VEL=BASE/'data_raw/velocity/its_live/v2.1'
PACKAGE=PROJECT/'polim_cpu/lia-inputs-v1/system-2387/spatial-source-paired-v1/input-package.json'
MESH=PROJECT/'polim_cpu/runs/v6-global-bp-r1/system2387-native-mesh/mesh.npz'
RUN=PROJECT/'polim_cpu/runs/v12-fixed-material-relaxation-v1'
IDS=['RGI60-15.04813','RGI60-15.04811','RGI60-15.04814']

def readnp(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def stats(v):
    v=np.asarray(v);v=v[np.isfinite(v)]
    return dict(n=int(v.size),median=float(np.median(v)) if v.size else None,p95=float(np.quantile(v,.95)) if v.size else None,max=float(v.max()) if v.size else None)

def main():
    validate();OUT.mkdir(parents=True,exist_ok=False)
    package=json.loads(PACKAGE.read_text(encoding='utf-8'));require_paired_source(PACKAGE,PROJECT/package['network']['path'])
    cfgp=BASE/'ice_surface_reconstruction/config/paleo_surface_validation_v0.json';cfg=json.loads(cfgp.read_text(encoding='utf-8'))
    registry=json.loads((VEL/'download_registry.json').read_text(encoding='utf-8'))
    a=readnp(PROJECT/package['array_archive']['path']);mesh=readnp(MESH)
    geompath=PROJECT/'polim_cpu/reports/geometry-source-audit-v1/audit-fields.npz';g=readnp(geompath)
    initial=readnp(RUN/'real/frame-000/state.npz');final=readnp(RUN/'real/frame-024/state.npz')
    x,y=a['x'],a['y'];dx=float(x[1]-x[0]);dy=float(y[1]-y[0]);shape=a['thickness_m'].shape;X,Y=np.meshgrid(x,y)
    local=package['metadata']['crs_wkt'];transform=Affine(dx,0,x[0]-dx/2,0,dy,y[0]-dy/2)
    rows,cols,nodes=mesh['pixel_rows'],mesh['pixel_cols'],mesh['pixel_center_nodes'];ice=np.zeros(shape,bool);ice[rows,cols]=True
    def raster(values):
        r=np.full(shape,np.nan);r[rows,cols]=values;return r
    sim0=raster(np.linalg.norm(initial['u'][nodes,-1],axis=1));sim1=raster(np.linalg.norm(final['u'][nodes,-1],axis=1))
    H0=raster(initial['h'][nodes]);H1=raster(final['h'][nodes]);surface0=a['bed_elevation_m']+H0;surface1=a['bed_elevation_m']+H1
    def slope(s):
        gy,gx=np.gradient(s,dy,dx);return np.degrees(np.arctan(np.hypot(gx,gy)))
    slope0=slope(surface0);slope1=slope(surface1);modern_slope=slope(g['dem'])
    modern=g['modern_mask'].astype(bool)
    outlines_path=resolve(cfg['modern_geometry_path']);outlines=gpd.read_file(outlines_path);key=next(k for k in ['RGIId','rgi_id','RGI_ID'] if k in outlines.columns)
    members=outlines[outlines[key].isin(IDS)].to_crs(local)
    if len(members)!=3:raise ValueError('Modern members missing')
    membermask=rasterize([(row.geometry,IDS.index(row[key])+1) for _,row in members.iterrows()],out_shape=shape,transform=transform,fill=0)
    sources=[PACKAGE,PROJECT/package['array_archive']['path'],MESH,geompath,cfgp,outlines_path,VEL/'download_registry.json',RUN/'result.json',RUN/'real/frame-000/state.npz',RUN/'real/frame-024/state.npz']
    arrays={};layer_metadata={};w=None
    for rec in registry['files']:
        if rec['role']!='target_clip':continue
        p=VEL/rec['path'];actual=digest(p)
        if actual!=rec['sha256'].lower():raise ValueError('Source hash changed: '+str(p))
        sources.append(p)
        with rasterio.open(p) as ds:
            if w is None:
                obscrs=ds.crs;forward=Transformer.from_crs(local,obscrs,always_xy=True);backward=Transformer.from_crs(obscrs,local,always_xy=True)
                ox,oy=forward.transform(X,Y);fc,fr=(~ds.transform)*(ox,oy);r0=max(0,int(np.floor(fr.min()))-1);c0=max(0,int(np.floor(fc.min()))-1)
                r1=min(ds.height,int(np.floor(fr.max()))+2);c1=min(ds.width,int(np.floor(fc.max()))+2);w=Window(c0,r0,c1-c0,r1-r0);ot=ds.window_transform(w);oshape=(r1-r0,c1-c0)
                oi=(np.floor(fr).astype(int)-r0)*oshape[1]+np.floor(fc).astype(int)-c0
                rr,cc=np.indices(oshape);ocx,ocy=ot*(cc+.5,rr+.5);lx,ly=backward.transform(ocx,ocy)
            if ds.crs!=obscrs or ds.window_transform(w)!=ot:raise ValueError('Misaligned observation layers')
            b=ds.read(1,window=w).astype(float);b[b==ds.nodata]=np.nan;arrays[rec['layer']]=b
            layer_metadata[rec['layer']]={k:v for k,v in ds.tags(1).items()};layer_metadata[rec['layer']]['source_sha256']=actual
    N=int(np.prod(oshape));flatidx=oi.ravel();n_all=np.bincount(flatidx,minlength=N)
    def aggregate(v,mask=ice):
        m=mask&np.isfinite(v);n=np.bincount(oi[m],minlength=N);s=np.bincount(oi[m],weights=v[m],minlength=N)
        return np.divide(s,n,out=np.full(N,np.nan),where=n>0).reshape(oshape)
    frac=aggregate(ice.astype(float),np.ones(shape,bool));mfrac=aggregate(modern.astype(float),np.ones(shape,bool))
    # One result per native observation cell, never count upsampled pixels as observations.
    v=arrays['v'];err=arrays['v_error'];count=arrays['count'];valid=np.isfinite(v)&(v>=0)&np.isfinite(arrays['vx'])&np.isfinite(arrays['vy'])&np.isfinite(count)&(count>0)
    resolved=valid&np.isfinite(err)&(err>=0)&(v>2*err)
    core=(frac>=.5)&(mfrac>=.5);common=core&valid;reliable=core&resolved
    s0=aggregate(sim0);s1=aggregate(sim1);h0=aggregate(H0);h1=aggregate(H1);sl0=aggregate(slope0);sl1=aggregate(slope1);sldem=aggregate(modern_slope)
    ux=aggregate(raster(final['u'][nodes,-1,0]));uy=aggregate(raster(final['u'][nodes,-1,1]));us=np.hypot(ux,uy)
    # Rotate observed directions through a one-metre displacement; retain speed magnitude.
    norm=np.hypot(arrays['vx'],arrays['vy']);tx,ty=backward.transform(ocx+np.divide(arrays['vx'],norm,out=np.zeros_like(norm),where=norm>0),ocy+np.divide(arrays['vy'],norm,out=np.zeros_like(norm),where=norm>0))
    odx,ody=tx-lx,ty-ly;onorm=np.hypot(odx,ody);cos=(ux*odx+uy*ody)/np.maximum(us*onorm,1e-30);angle=np.degrees(np.arccos(np.clip(cos,-1,1)));angle[~reliable|(us<=0)]=np.nan
    subsets={}
    for name,mask in [('modern_overlap',core),('lia_footprint',frac>=.5)]:
        ok=mask&valid;good=mask&resolved
        subsets[name]=dict(native_cells=int(mask.sum()),valid_cells=int(ok.sum()),coverage=float(ok.sum()/max(mask.sum(),1)),resolved_cells=int(good.sum()),observed=stats(v[ok]),observed_error=stats(err[ok]),model_initial=stats(s0[ok]),model_day1=stats(s1[ok]),resolved_direction_difference_degrees=stats(angle[good]))
    branchrows=[];profiles=[]
    for p in sorted((PROJECT/'polim_cpu/lia-inputs-v1/system-2387/spatial-profiles-v3-paired').glob('branch-*.npz')):
        sources.append(p);b=readnp(p);bx,by=b['easting_m'],b['northing_m'];xx,yy=forward.transform(bx,by);cc,rr=(~ot)*(xx,yy);br=np.floor(rr).astype(int);bc=np.floor(cc).astype(int)
        sr=np.rint((by-y[0])/dy).astype(int);sc=np.rint((bx-x[0])/dx).astype(int);mo=modern[sr,sc];indices=np.unique((br*oshape[1]+bc)[mo]);vr=v.ravel()[indices];good=valid.ravel()[indices]
        mo=modern[sr,sc];obsline=v[br,bc].copy();obsline[~valid[br,bc]|~mo]=np.nan
        obs_err=err[br,bc].copy();obs_err[~np.isfinite(obsline)]=np.nan
        row=dict(branch=int(b['branch_index'].flat[0]),points=len(bx),native_observation_cells=len(indices),modern_points=int(mo.sum()),observed_unique_cells=stats(vr[good]),model_day1=stats(sim1[sr,sc]))
        branchrows.append(row);profiles.append(dict(branch=row['branch'],station_m=b['station_m'].tolist(),observed=obsline.tolist(),error=obs_err.tolist(),model_initial=sim0[sr,sc].tolist(),model_day1=sim1[sr,sc].tolist(),H=H1[sr,sc].tolist(),slope=sl1[br,bc].tolist(),x=bx.tolist(),y=by.tolist()))
    memberrows=[]
    for label,rgi in enumerate(IDS,1):
        f=aggregate((membermask==label).astype(float),np.ones(shape,bool));mask=(f>=.5)&(frac>=.5);ok=mask&valid
        memberrows.append(dict(rgi_id=rgi,native_cells=int(mask.sum()),valid_cells=int(ok.sum()),observed=stats(v[ok]),model_day1=stats(s1[ok])))
    peak=int(np.argmax(np.linalg.norm(final['u'][:,-1],axis=1)));px,py=mesh['xy'][peak];pox,poy=forward.transform(px,py);pc,pr=(~ot)*(pox,poy);pr,pc=int(np.floor(pr)),int(np.floor(pc));gr=int(round((py-y[0])/dy));gc=int(round((px-x[0])/dx))
    peakinfo=dict(node=peak,xy=[float(px),float(py)],lon_lat=list(Transformer.from_crs(local,4326,always_xy=True).transform(px,py)),speed_m_year=float(np.linalg.norm(final['u'][peak,-1])),inside_modern_mask=bool(modern[gr,gc]),modern_fraction_of_observation_cell=float(mfrac[pr,pc]),observation_cell_valid=bool(valid[pr,pc]),observation_m_year=float(v[pr,pc]) if valid[pr,pc] else None,observation_error_m_year=float(err[pr,pc]) if np.isfinite(err[pr,pc]) else None,model_cell_mean_m_year=float(s1[pr,pc]),H_m=float(final['h'][peak]),model_surface_slope_degrees=float(slope1[gr,gc]),modern_dem_slope_degrees=float(modern_slope[gr,gc]))
    junctionxy=np.array([-134.49697636368546,3158715.5030236365]);jmask=(np.hypot(lx-junctionxy[0],ly-junctionxy[1])<=240)&(frac>=.5)&valid
    junction=dict(radius_m=240,observed=stats(v[jmask]),model_day1=stats(s1[jmask]),modern_overlap_cells=int((jmask&core).sum()))
    fields=dict(observed=v,error=err,count=count,outlier_percent=arrays['outlier_percent'],valid=valid,resolved=resolved,lia_fraction=frac,modern_fraction=mfrac,model_initial=s0,model_day1=s1,H_initial=h0,H_day1=h1,slope_initial_deg=sl0,slope_day1_deg=sl1,modern_dem_slope_deg=sldem,direction_difference_deg=angle,obs_x=ocx,obs_y=ocy,local_x=lx,local_y=ly,observed_direction_x=odx/onorm,observed_direction_y=ody/onorm,model_direction_x=ux/np.maximum(us,1e-30),model_direction_y=uy/np.maximum(us,1e-30),x=x,y=y,modern_mask=modern,lia_mask=ice,native_model_speed=sim1,native_H=H1,native_slope_deg=slope1)
    np.savez_compressed(OUT/'comparison-fields.npz',**fields)
    summary=dict(status='completed-read-only-comparison',system=2387,new_simulations=0,calibration=False,scientific_acceptance=False,accepted_count_effect=0,observation=registry['dataset'],observation_grid=dict(crs=str(obscrs),window=[w.col_off,w.row_off,w.width,w.height],transform=list(ot),resolution_m=120,shape=oshape),method=dict(spatial='Model 15.75m pixel-centre values averaged by membership of each 120m observation cell. Equal native-grid area weights; no interpolation of observations for statistics. Modern and LIA membership each require >=50percent of native sample centres in the cell.',quality='Finite nonnegative v,finite vx/vy,count>0. Preserve all valid speeds including unresolved low speeds. Extra direction-only subset v>2*v_error is a disclosed diagnostic screen,not a scientific acceptance gate.',time='ITS_LIVE2014-2024 climatology evaluated at2018-01-01 time intercept,not a single-year measurement. Compare to instantaneous LIA-geometry BP day1,not a modern-geometry simulation.',slope='arctan central-difference gradient in degrees; invalid derivatives near masked boundaries retained NaN. Report original and day1 model surface plus modern DEM.',profiles='Branch paths are original LIA network; observations plotted only where the sampled path is inside modern mask. Stepwise120m samples,never new15.75m observations.'),subsets=subsets,members=memberrows,branches=branchrows,peak=peakinfo,junction=junction,layer_metadata=layer_metadata,source_hashes={p.relative_to(PROJECT).as_posix():digest(p) for p in sources},code_sha256=digest(Path(__file__)))
    write_json(OUT/'summary.json',summary);write_json(OUT/'profiles.json',profiles)
    print(json.dumps({k:summary[k] for k in ['status','subsets','members','peak','junction']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
