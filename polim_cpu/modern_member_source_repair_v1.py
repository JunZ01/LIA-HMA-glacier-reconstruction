"""Read-only source reconciliation; retire the unlaunched legacy control."""
import json,csv,copy,re,os,hashlib
from datetime import datetime,timezone
import numpy as np
import rasterio
from .v15.experiment import PROJECT,BASE,RAW,STATS,REPAIR,IDS
from .v14.experiment import NAME as OLDNAME,EVIDENCE as OLDEVIDENCE,RUN as OLDRUN
from .v3.runner import digest,write_json
from .thermal_attribution_v1 import validate,ref

def reconcile():
    validate();sp=PROJECT/'LIA_CURRENT_STATE.json';before=sp.read_bytes();s=json.loads(before);facts=PROJECT/'LIA_PROJECT_FACTS.md';original=facts.read_bytes()
    if digest(facts)!=s['scientific_contract']['sha256'].lower():raise ValueError('Facts mismatch')
    if OLDRUN.exists():raise ValueError('Legacy momentum already launched')
    status=BASE/'outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/stage0_bed_full_farinotti_v0/inversion_member_status.csv'
    system=status.with_name('inversion_system_summary.csv')
    with status.open(encoding='utf-8-sig',newline='') as f:members=[r for r in csv.DictReader(f) if r['lia_unique_id']=='2387']
    with system.open(encoding='utf-8-sig',newline='') as f:systems=[r for r in csv.DictReader(f) if r['lia_unique_id']=='2387']
    with STATS.open(encoding='utf-8-sig',newline='') as f:refs={r['rgi_id']:r for r in csv.DictReader(f) if r['rgi_id'] in IDS}
    rows=[];sources=[status,system,STATS,BASE/'src/stage0_bed/run_inversion_pilot.py']
    for row in members:
        rgi=row['rgi_id'];p=RAW/f'{rgi}_thickness.tif';sources.append(p)
        with rasterio.open(p) as ds:
            H=ds.read(1).astype(float);positive=H>0;pixel_area=abs(ds.transform.a*ds.transform.e);volume=float(H[positive].sum()*pixel_area)
            reference=float(refs[rgi]['far_volume_km3'])*1e9
            if not np.isclose(volume,reference,rtol=1e-10):raise ValueError('Raw raster does not match product reference')
            legacy=float(row['distributed_volume_km3'])*1e9
            rows.append(dict(rgi=rgi,reference_volume_m3=reference,raw_raster_volume_m3=volume,legacy_volume_m3=legacy,legacy_to_reference_ratio=legacy/reference,raw_mean_H_m=float(H[positive].mean()),raw_max_H_m=float(H.max()),legacy_max_H_m=float(row['max_thickness_m']),raw_area_m2=float(positive.sum()*pixel_area)))
    now=datetime.now(timezone.utc).isoformat();REPAIR.mkdir(parents=True,exist_ok=False)
    audit=dict(status='resolved-source-misallocation',members=rows,legacy_system_summary=systems,mechanism='System-total inversion tunes one Glen A to sum(member volumes);it records individual targets but does not constrain individual residuals. A correct system total conceals nearly1000-fold underallocation to04811,overallocation to04813 and underallocation to04814. Spatial shape also differs:legacy04813max278.913m versus original F19max104.665m.',resolution='For new modern control only,use original member-resolved F19 thickness rasters and modern DEM-derived bed. Preserve all originals and historic scientific outputs. Never scale the almost-zero legacy04811 shape by~1000. No physical calibration or numerical momentum run performed in this source reconciliation.',sources={p.relative_to(PROJECT).as_posix():digest(p) for p in sources},created_at=now)
    ep=REPAIR/'audit.json';write_json(ep,audit)
    b={**s['research_branches'][OLDNAME],'status':'withdrawn-before-momentum-member-geometry-failed','launch_authorized':False,'real_momentum_authorized':False,'allowed_stages':[],'evidence':ref(ep),'next_legal_action':'Preserve prepared legacy geometry;source-resolved modern control requires its own versioned plan.'}
    number=max(map(int,re.findall(r'^## (\d+)\.',original.decode('utf-8'),re.M)))+1
    addition=f'''\n\n## {number}. Modern-member thickness source misallocation resolved before BP launch\n\n- User starts modern geometry BP control. Read-only preparation exposes RGI60-15.04811 distributed modern thickness mean0.02832m,max0.8169946m. Original stage0member ledger target volume0.0491807036021351km3 versus distributed0.00004927397568653261km3,about0.1002percent of target. Main04813target0.6743017027094961km3 versus distributed0.7258765196591971km3;04814target0.015369909712076099km3 versus distributed0.012926522853394032km3. System total nevertheless passes at relative residual1.5463852420793955e-11. Source code tunes one system Glen A against the summed volume,without member-volume constraints. This resolves a concrete mismatch between member reference volumes and the provisional system-total inverse geometry;neither original files nor their historical provenance is changed.\n- All3original25m Farinotti2019 target-subset rasters exist locally and their integrals match the saved product statistics to1e-10relative. Original04811mean30.1029555m,max70.6252518m;original04813max104.6649094m versus inverse-distributed278.9126892m. Thus a field named distributed_thickness_f19_fs0_full_v0 is an inverse redistribution calibrated only in system total,not the original F19spatial thickness. The old field remains an exact source for older prototypes,but is not suitable as an unquestioned modern-member geometry control.\n- V14modern control is withdrawn before any momentum solve;prepared inputs preserved. New authorized modern control may use original F19spatial H with modern topo,derivedB=S-H,and explicit reprojection/footprint errors. No multiplication of the degenerate old04811shape,no global source replacement,Full-Stokes,acceptance or mainline change. Modern product H remains estimated,not observed truth. Evidence `{ref(ep)['path']}`.\n'''
    newhash=hashlib.sha256(original+addition.encode('utf-8')).hexdigest().upper();route=json.loads((PROJECT/s['authoritative_route']['path']).read_text(encoding='utf-8'));route['version']+='+modern-member-source-resolved';route[OLDNAME]=b
    if 'scientific_contract' in route:route['scientific_contract']={**s['scientific_contract'],'sha256':newhash}
    rp=REPAIR/'route-source-resolved.json';write_json(rp,route);new=copy.deepcopy(s);new['research_branches'][OLDNAME]=b;new['scientific_contract']['sha256']=newhash;new['authoritative_route']={**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']};new['state_id']+='+modern-member-source-resolved';new['updated_at']=now
    if sp.read_bytes()!=before or facts.read_bytes()!=original:raise ValueError('Concurrent update')
    tmp=sp.with_name(f'.LIA_CURRENT_STATE.sourcefix.{os.getpid()}.tmp');write_json(tmp,new);os.replace(tmp,sp)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {now} — Modern member source mismatch resolved\n\nSystem-total inverse geometry hides severe member mismatch. Original product rasters verified;v14withdrawn before momentum,new modern control will use original member-resolved thickness. Facts{number};evidence `{ref(ep)["path"]}`. Original sources and unrelated branches preserved.\n')
    with facts.open('ab') as f:f.write(addition.encode('utf-8'));f.flush();os.fsync(f.fileno())
    write_json(REPAIR/'validation.json',validate());print(json.dumps(dict(status=audit['status'],facts_section=number,members=rows)),flush=True)

if __name__=='__main__':reconcile()
