"""One modern diagnostic BP solve; original experiments and sources immutable."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
os.environ['NUMBA_NUM_THREADS']='8'
import json,copy,time,traceback,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from ..v3.runner import PROJECT,digest,seal,write_json
from ..thermal_attribution_v1 import validate,ref

ROOT=Path(__file__).resolve().parent
NAME='polim_v15_modern_raw_f19_BP_control_v1'
EVIDENCE=PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME
INPUT=ROOT.parent/'lia-inputs-v1/system-2387/modern-raw-f19-bp-control-v1'
RUN=ROOT.parent/'runs/v15-modern-raw-f19-bp-control-v1'
REPORT=ROOT.parent/'reports/modern-raw-f19-bp-control-v1'
PACKAGE=ROOT.parent/'lia-inputs-v1/system-2387/spatial-source-paired-v1/input-package.json'
ARCHIVE=PACKAGE.with_name('geometry.npz')
OLDMESH=ROOT.parent/'runs/v6-global-bp-r1/system2387-native-mesh/mesh.npz'
MATERIAL=ROOT.parent/'runs/v12-fixed-material-relaxation-v1/frozen-material.npz'
OBS=ROOT.parent/'reports/observed-velocity-v1'
BASE=PROJECT/'Himalaya_greatGlacier/With_Rec_climate_organized'
GROOT=BASE/'outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/stage0_bed_full_farinotti_v0/oggm_workdir/per_glacier/RGI60-15/RGI60-15.04'
IDS=['RGI60-15.04813','RGI60-15.04811','RGI60-15.04814']
RAW=BASE/'data_raw/thickness/farinotti_2019/release_2019/target_subset/RGI60-15'
STATS=BASE/'outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/thickness_work/farinotti_glacier_stats.csv'
REPAIR=PROJECT/'validation_data/results/paleo_surface_validation_v0/polim_modern_member_source_repair_v1'

def readnp(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def implementation():
    paths=list(ROOT.glob('*.py'))+[ROOT.parent/p for p in ['v13r1/bp.py','v6r1/mesh.py','v6r1/storage.py','v9/storage.py','thermal_attribution_v1.py']]
    paths+=list((ROOT.parent/'v3').glob('*.py'))
    paths+=[BASE/'ice_surface_reconstruction/drivers'/p for p in ['lia_system_grid.py','gridded_bed.py']]
    return {p.relative_to(PROJECT).as_posix():digest(p) for p in sorted(paths)}

def active():
    validate();state=json.loads((PROJECT/'LIA_CURRENT_STATE.json').read_text(encoding='utf-8'))
    b=state['research_branches'][NAME]
    if not b['launch_authorized'] or state['current_focus']['state']=='decision-required':raise ValueError('Launch not authorized')
    p=PROJECT/b['plan']['path']
    if digest(p)!=b['plan']['sha256'].lower():raise ValueError('Plan changed')
    plan=json.loads(p.read_text(encoding='utf-8'))
    if plan['implementation']!=implementation():raise ValueError('Implementation changed')
    for p,h in plan['input_hashes'].items():
        if digest(PROJECT/p)!=h:raise ValueError('Source changed: '+p)
    return plan

def register():
    validate();sp=PROJECT/'LIA_CURRENT_STATE.json';before=sp.read_bytes();s=json.loads(before)
    if NAME in s['research_branches'] or s['current_focus']['state']=='decision-required':raise ValueError('Duplicate or decision required')
    inputs=[PACKAGE,ARCHIVE,OLDMESH,MATERIAL,OBS/'summary-v2.json',OBS/'comparison-fields-v2.npz',OBS/'profiles-v2.json',ROOT.parent/'reports/geometry-source-audit-v1/audit.json']
    for rgi in IDS:inputs.extend([GROOT/rgi/'gridded_data.nc',GROOT/rgi/'glacier_grid.json',RAW/f'{rgi}_thickness.tif'])
    inputs.extend([STATS,REPAIR/'audit.json'])
    # Match previously verified modern source hashes before registering a new control.
    audit=json.loads((ROOT.parent/'reports/geometry-source-audit-v1/audit.json').read_text(encoding='utf-8'))
    for record in audit['sources']:
        if Path(record['path']).name=='gridded_data.nc' and digest(record['path'])!=record['sha256']:raise ValueError('Verified modern source changed')
    plan=dict(schema='modern-fixed-material-BP-control-v1',user_authorization='User says start after proposed modern geometry plus current BP/material comparison.',
        implementation=implementation(),input_hashes={p.relative_to(PROJECT).as_posix():digest(p) for p in inputs},
        geometry='Use original25m Farinotti2019 member thickness rasters,whose individual integrals match saved product statistics. Full positive-product footprint projected nearest to15.75m target;bilinear H including true exterior zero,paired with bilinear stage0 topo for that member. Select overlaps by maximum raw-footprint interior distance;B=S-H. No inverse-geometry redistribution,per-member rescaling,smoothing or velocity fit. Preserve originals;report projection/masking integral change for each member. Derived target domain differs from old coarse gdir masks and is reported explicitly.',
        material='Frozen v12 GPBLD A at sigma0,.5,1. New mesh uses nearest old mesh node at identical xy wherever available;else nearest spatial extension,indices/distances saved. Preserve all transferred A values bitwise. Prior coarse-mask audit found5429of70078centres outside old LIA mesh;new raw-product footprint coverage is measured during preparation. Report exact-coordinate and extended subsets separately. This is a controlled old-material experiment,NOT a measured modern temperature field.',
        dynamics='Exactly one fixed-geometry diagnostic BP solve,three sigma nodes,basal no-slip,strong elimination;existing natural lateral variational boundary. No evolution,SMB,thermal update,sliding inversion,parameter search or Full-Stokes. Reuse v13r1 BP and prior numerical/restart fixtures unchanged.',
        resources=dict(threads=8,device='CPU',estimated_seconds=[60,900],maximum_seconds=1800,estimated_peak_memory_GiB=12),
        checkpoint='Immutable prepared geometry and A;every Newton iterate sealed. Unique contract and output root. No duplicate launch;technical interruption can resume only through separately registered exact-checkpoint recovery.',
        checks=dict(momentum_relative_residual=1e-6,geometry_relative_volume=1e-12,source_centres_exact=True,A_transfer_bitwise=True),
        comparison='Primary modern coverage>=50percent and valid ITS_LIVE120m observations;also previous953shared cells and all-nodes-exact-material footprint. Pixel-centre speed mean and mean vector reported separately;direction restricted to positive-error SNR>2. No post-hoc fitting. Modern geometry nominal dates differ from2014-2024velocity climatology;F19 thickness estimated,thermal and basal assumptions unchanged. Control constrains attribution but is not an independent solver validation.',
        stop_conditions=['source or code mismatch','invalid modern S/H','cropped modern mask','mesh volume mismatch','nonfinite A or velocity','BP nonconvergence or1800s budget'],scientific_acceptance=False,accepted_count_effect=0,production_replacement_authorized=False,source_facts=s['scientific_contract'],source_repair=ref(REPAIR/'audit.json'))
    EVIDENCE.mkdir(parents=True,exist_ok=False);pp=EVIDENCE/'plan.frozen.json';write_json(pp,plan);now=datetime.now(timezone.utc).isoformat()
    ep=EVIDENCE/'development-registration.json';write_json(ep,dict(status='modern-control-authorized',plan=ref(pp),source_route=s['authoritative_route'],created_at=now))
    b=dict(status='modern-control-authorized',launch_authorized=True,real_momentum_authorized=True,real_glacier_evolution_authorized=False,full_stokes_authorized=False,allowed_stages=['prepare-modern-paired-geometry','one-diagnostic-BP-solve','observation-comparison-and-delivery'],plan=ref(pp),evidence=ref(ep),accepted_count_effect=0,production_replacement_authorized=False,next_legal_action='One frozen modern control;no automatic extension or calibration')
    route=json.loads((PROJECT/s['authoritative_route']['path']).read_text(encoding='utf-8'));route['version']+='+polim-v15-modern-control';route[NAME]=b;rp=EVIDENCE/'route-development.json';write_json(rp,route)
    new=copy.deepcopy(s);new['research_branches'][NAME]=b;new['authoritative_route']={**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']};new['state_id']+='+polim-v15-modern-control';new['updated_at']=now
    if sp.read_bytes()!=before:raise ValueError('Concurrent state update')
    tmp=sp.with_name(f'.LIA_CURRENT_STATE.modern.{os.getpid()}.tmp');write_json(tmp,new);os.replace(tmp,sp)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {now} — Modern geometry BP control authorized\n\nUser starts proposed modern geometry control. Original member-resolved F19 footprint and thickness,paired modern topo,frozen GPBLD A transferred with explicit nearest spatial extension,one diagnostic BP solve and modern observation comparison. No closed smoothing rerun,evolution,calibration,Full-Stokes or acceptance. Plan `{ref(pp)["path"]}`.\n')
    write_json(EVIDENCE/'development-validation.json',validate());print('modern-control-authorized',flush=True)

def prepare():
    active()
    from scipy.spatial import cKDTree
    import rasterio,csv
    from rasterio.warp import reproject
    from scipy.ndimage import distance_transform_edt
    from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers import lia_system_grid as grid
    from ..v6r1.mesh import build
    INPUT.mkdir(parents=True,exist_ok=False);p=json.loads(PACKAGE.read_text(encoding='utf-8'));a=readnp(ARCHIVE);x,y=a['x'],a['y'];shape=a['thickness_m'].shape
    target=grid.SystemGrid(p['metadata']['crs_wkt'],float(x[0]),float(y[0]),len(x),len(y),float(x[1]-x[0]),float(y[1]-y[0]),INPUT/'target.json')
    S=np.full(shape,np.nan);H=S.copy();owner=np.zeros(shape,np.int16);quality=np.full(shape,-np.inf);modern=np.zeros(shape,bool);members=[]
    with STATS.open(encoding='utf-8-sig',newline='') as f:references={row['rgi_id']:row for row in csv.DictReader(f)}
    for k,rgi in enumerate(IDS,1):
        d=GROOT/rgi;source,topo,oldbed,oldmask=grid._load_source(d,'farinotti2019')
        st=grid._reproject(topo,source,target,str(source['proj']),resampling=grid.Resampling.bilinear)
        with rasterio.open(RAW/f'{rgi}_thickness.tif') as ds:
            raw=ds.read(1).astype(float)
            if not np.isfinite(raw).all() or np.any(raw<0):raise ValueError('Invalid original thickness')
            valid=raw>0;source_volume=float(raw[valid].sum()*abs(ds.transform.a*ds.transform.e));reference=float(references[rgi]['far_volume_km3'])*1e9
            if not np.isclose(source_volume,reference,rtol=1e-10):raise ValueError('Original product volume mismatch')
            def project(values,method):
                out=np.zeros(shape,float);reproject(values,out,src_transform=ds.transform,src_crs=ds.crs,dst_transform=target.transform,dst_crs=target.crs,resampling=method,src_nodata=None,dst_nodata=0);return out
            ht=project(raw,grid.Resampling.bilinear);mt=project(valid.astype(float),grid.Resampling.nearest)>.5
            q=project(distance_transform_edt(np.pad(valid,1))[1:-1,1:-1]*abs(ds.transform.a),grid.Resampling.bilinear)
            raw_pixels=int(valid.sum());raw_area=float(raw_pixels*abs(ds.transform.a*ds.transform.e));raw_max=float(raw.max())
        if np.any(mt&(~np.isfinite(st)|~np.isfinite(ht)|(ht<0))):raise ValueError('Invalid modern projection')
        choose=mt&((owner==0)|(q>quality));S[choose]=st[choose];H[choose]=ht[choose];quality[choose]=q[choose];owner[choose]=k;modern|=mt
        members.append(dict(rgi=rgi,source_pixels=raw_pixels,target_mask_pixels=int(mt.sum()),source_grid_resolution_m=25.,source_area_m2=raw_area,source_volume_m3=source_volume,source_max_H_m=raw_max,legacy_distributed_volume_m3=float((topo-oldbed)[oldmask].sum()*abs(np.prod(source['dxdy'])))))
    for k,member in enumerate(members,1):
        use=owner==k;member['selected_target_volume_m3']=float(H[use].sum()*abs(target.dx*target.dy));member['projection_volume_change_fraction']=member['selected_target_volume_m3']/member['source_volume_m3']-1
    if modern[0].any() or modern[-1].any() or modern[:,0].any() or modern[:,-1].any():raise ValueError('Target clips modern ice')
    mask=modern&(H>1e-6);B=S-H;mesh,metrics=build(mask,x,y,B,H)
    old=readnp(OLDMESH);A0=readnp(MATERIAL)['A'];distance,index=cKDTree(old['xy']).query(mesh['xy']);A=A0[index].copy();exact=distance<1e-7
    if not np.isfinite(A).all() or np.any(A<=0):raise ValueError('Invalid transferred material')
    if not np.array_equal(A,A0[index]):raise ValueError('Material transfer changed A')
    covered=np.zeros(shape,bool);covered[old['pixel_rows'],old['pixel_cols']]=True
    # A cell is wholly unextended only if its centre and four corner nodes match old coordinates.
    strict=np.zeros(shape,bool);strict[mesh['pixel_rows'],mesh['pixel_cols']]=exact[mesh['pixel_center_nodes']]&np.all(exact[mesh['pixel_corner_nodes']],axis=1)
    np.savez_compressed(INPUT/'geometry.npz',surface=S,thickness=H,bed=B,modern_mask=modern,mask=mask,member=owner,x=x,y=y,old_coverage=covered,strict_material_coverage=strict)
    np.savez_compressed(INPUT/'mesh.npz',**mesh);np.savez_compressed(INPUT/'material.npz',A=A,source_node=index,distance_m=distance,exact_coordinate=exact)
    summary=dict(status='prepared',members=members,mesh=metrics,modern_mask_pixels=int(modern.sum()),positive_pixels=int(mask.sum()),excluded_zero_or_trace_pixels=int((modern&~mask).sum()),excluded_trace_volume_m3=float(H[modern&~mask].sum()*abs(target.dx*target.dy)),outside_old_mesh_pixels=int((mask&~covered).sum()),exact_material_nodes=int(exact.sum()),extended_material_nodes=int((~exact).sum()),maximum_material_extension_m=float(distance.max()),strict_material_pixels=int(strict.sum()),material_transfer_bitwise=True,paired_surface_identity_error_m=float(np.max(abs(mesh['bed'][mesh['pixel_center_nodes']]+mesh['thickness'][mesh['pixel_center_nodes']]-S[mask]))),geometry_provenance='Original25m F19 spatial thickness,stage0 modern topo;bed derived anew. Original F19 is an estimate,not observed thickness or exact2018geometry. Old system-total inversion distribution is superseded for this control only.',fixtures_reused=['v8 BP boundary benchmarks','v12 independent sealed restart','v13r1 dry-node algebraic equivalence'],new_numerical_fixtures=0)
    write_json(INPUT/'summary.json',summary);write_json(INPUT/'manifest.json',dict(files={q.name:digest(q) for q in INPUT.iterdir() if q.is_file()},implementation=implementation()));print(json.dumps(summary,ensure_ascii=False),flush=True)

def run():
    plan=active();manifest=json.loads((INPUT/'manifest.json').read_text(encoding='utf-8'))
    for name,h in manifest['files'].items():
        if digest(INPUT/name)!=h:raise ValueError('Prepared input changed')
    if manifest['implementation']!=implementation():raise ValueError('Preparation code changed')
    from ..v13r1 import bp
    from ..v9 import storage
    from ..v6r1.storage import callback
    RUN.mkdir(parents=True,exist_ok=False);contract=seal(dict(plan=seal(plan),inputs=digest(INPUT/'manifest.json'),implementation=implementation(),python=sys.version,numpy=np.__version__));start=time.perf_counter()
    write_json(EVIDENCE/f'launch-fingerprint-{contract}.json',dict(contract=contract,pid=os.getpid(),output=str(RUN)))
    write_json(RUN/'launch.json',dict(contract=contract,pid=os.getpid(),start_utc=datetime.now(timezone.utc).isoformat(),command='python -m polim_cpu.v15 run',resources=plan['resources'],checkpoint=plan['checkpoint'],diagnostic_only=True))
    try:
        mesh=readnp(INPUT/'mesh.npz');A=readnp(INPUT/'material.npz')['A'];storage.status(RUN,dict(status='running',stage='diagnostic-BP',pid=os.getpid()))
        u,info=bp.solve(mesh,A=A,max_seconds=1800,checkpoint=callback(RUN/'newton',contract))
        speed=np.linalg.norm(u[:,-1],axis=1);peak=int(speed.argmax());checks=dict(momentum_converged=bool(info['converged']),relative_residual=float(info['relative_residual']),nonnegative_H=bool(np.all(mesh['thickness']>=0)),finite_velocity=bool(np.isfinite(u).all()),basal_no_slip_exact=bool(np.all(u[:,0]==0)),geometry_unchanged=True,material_unchanged=True)
        storage.save(RUN,0,dict(h=mesh['thickness'],A=A,time=0.,step=0),u,contract,dict(momentum=info,checks=checks))
        result=dict(status='completed',contract=contract,checks=checks,momentum=info,max_speed_m_year=float(speed.max()),peak_node=peak,peak_xy_m=mesh['xy'][peak].tolist(),peak_H_m=float(mesh['thickness'][peak]),geometry=json.loads((INPUT/'summary.json').read_text(encoding='utf-8')),wall_seconds=time.perf_counter()-start,diagnostic_solves=1,evolution_steps=0,scientific_acceptance=False,accepted_count_effect=0,implementation=implementation())
        write_json(RUN/'result.json',result)
    except Exception as exc:
        result=dict(status='technical-failure',error=repr(exc),traceback=traceback.format_exc(),contract=contract,wall_seconds=time.perf_counter()-start);write_json(RUN/'failure.json',result);print(result['traceback'],flush=True)
    storage.status(RUN,dict(status=result['status']));write_json(RUN/'terminal-evidence.json',dict(status=result['status'],implementation=implementation(),output_hashes={q.relative_to(RUN).as_posix():digest(q) for q in RUN.rglob('*') if q.is_file() and not q.name.startswith('status')},accepted_count_effect=0));print(json.dumps({k:result[k] for k in ['status','wall_seconds']}),flush=True)
