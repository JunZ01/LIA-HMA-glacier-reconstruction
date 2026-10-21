import json,copy,re,hashlib,os
from datetime import datetime,timezone
import numpy as np
from .v15.experiment import PROJECT,NAME,EVIDENCE,INPUT,RUN,REPORT,REPAIR,implementation,readnp
from .v3.runner import digest,write_json
from .thermal_attribution_v1 import validate,ref
from .v9.storage import restore

def complete():
    validate();result=json.loads((RUN/'result.json').read_text(encoding='utf-8'));terminal=json.loads((RUN/'terminal-evidence.json').read_text(encoding='utf-8'));summary=json.loads((REPORT/'summary.json').read_text(encoding='utf-8'))
    if result['status']!='completed' or terminal['implementation']!=implementation():raise ValueError('Result incomplete or code changed')
    for p,h in terminal['output_hashes'].items():
        if digest(RUN/p)!=h:raise ValueError('Saved result changed')
    prepared=json.loads((INPUT/'manifest.json').read_text(encoding='utf-8'))
    for p,h in prepared['files'].items():
        if digest(INPUT/p)!=h:raise ValueError('Prepared input changed')
    state,u,meta=restore(RUN/'frame-000',result['contract']);mesh=readnp(INPUT/'mesh.npz');A=readnp(INPUT/'material.npz')['A']
    if not np.array_equal(state['h'],mesh['thickness']) or not np.array_equal(state['A'],A):raise ValueError('Geometry/material changed during solve')
    delivery=json.loads((REPORT/'delivery-v2.json').read_text(encoding='utf-8'))
    for p,h in delivery['files'].items():
        if digest(REPORT/p)!=h:raise ValueError('Delivery changed')
    sp=PROJECT/'LIA_CURRENT_STATE.json';before=sp.read_bytes();s=json.loads(before);facts=PROJECT/'LIA_PROJECT_FACTS.md';original=facts.read_bytes()
    if digest(facts)!=s['scientific_contract']['sha256'].lower():raise ValueError('Facts mismatch')
    b=s['research_branches'][NAME]
    if not b['launch_authorized']:raise ValueError('Already closed')
    now=datetime.now(timezone.utc).isoformat();number=max(map(int,re.findall(r'^## (\d+)\.',original.decode('utf-8'),re.M)))+1
    completion=dict(status='completed-modern-raw-F19-BP-control',result=ref(RUN/'result.json'),comparison=ref(REPORT/'summary.json'),source_repair=ref(REPAIR/'audit.json'),delivery=ref(REPORT/'delivery-v2.json'),created_at=now,scientific_acceptance=False,accepted_count_effect=0,numerical_solves=1,legacy_v14_momentum_solves=0)
    cp=EVIDENCE/'completion.json';write_json(cp,completion);g=result['geometry'];f=summary['subsets']['full_modern'];j=summary['subsets']['junction240m'];d=summary['direction_difference_degrees'];exact=summary['subsets']['exact_material_only']
    addition=f'''\n\n## {number}. Modern raw-F19 geometry BP control completed\n\n- User authorizes proposed modern geometry control. Legacy v14input preparation exposed source-member misallocation and was retired with zero momentum solves(facts672). New v15uses original25m F19member spatial thickness,modern stage0topo andB=S-H. Full raw-positive footprint reprojected to15.75m:{g['mesh']['source_pixels']}pixels,{g['mesh']['nodes']}nodes,{g['mesh']['triangles']}triangles,{g['mesh']['components']}separate components. Reprojection/member-mask volume losses0.5291252%,0.8434035%,1.4899547%for04813/04811/04814,explicitly retained without renormalization. Target volume{g['mesh']['source_volume_m3']:.9f}m3equals mesh volume to1e-12relative;source centres exact. No artificial ice or changed original files.\n- Reused existing v13r1BP with three sigma nodes,basal no-slip and unchanged natural lateral form. Frozen v12GPBLD A transferred from old spatial nodes:{g['exact_material_nodes']}coordinate matches,{g['extended_material_nodes']}nearest spatial extensions,max{g['maximum_material_extension_m']:.9f}m. Exact and extended support reported separately;this is not observed modern temperature. One diagnostic solve completed in{result['wall_seconds']:.6f}s,{result['momentum']['iterations']}Newton iterations,residual{result['checks']['relative_residual']:.12g}. Original A andH bitwise unchanged in sealed final state;basal velocity zero. Prior BP/restart fixtures reused,no time evolution or extra numeric replay. Native maximum{result['max_speed_m_year']:.9f}m/year atH{result['peak_H_m']:.9f}m,on an exactly matched original material node.\n- On{f['model']['n']}valid independent120m modern cells,observed/model median{f['observed']['median']:.9f}/{f['model']['median']:.9f},p95{f['observed']['p95']:.9f}/{f['model']['p95']:.9f},max{f['observed']['max']:.9f}/{f['model']['max']:.9f}m/year;model above1000m/year cells{f['above1000_model_cells']}. Junction240m medians observed/model{j['observed']['median']:.9f}/{j['model']['median']:.9f}m/year. Entirely exact-material subset{exact['model']['n']}cells,observed/model medians{exact['observed']['median']:.9f}/{exact['model']['median']:.9f}. Direction median{d['median']:.9f}degrees,p95{d['p95']:.9f}degrees on{d['n']}positive-errorSNR>2cells;stored zero error never means exact observation.\n- Interpretation: extreme LIA-prototype velocities are not intrinsic to every run of this BP/material implementation;geometry/source distribution and footprint strongly affect response. This modern control remains generally too slow and is not a calibration or independent validation;uncertain thickness/DEM epochs,thermal assumptions,basal no-slip and spatial material extension remain relevant. No claim that LIAvelocities are already repaired or that omitted sliding alone explains every residual. Deliver source-volume table,modern velocity/ratio/direction maps,original8path modern comparisons. Plot export recovered from a list/array coordinate-format exception using already-computed fields andmaps,without another simulation. No Full-Stokes,sourceoverwrite,acceptance,automatic extension or population promotion. Evidence `{ref(cp)['path']}`.\n'''
    newhash=hashlib.sha256(original+addition.encode('utf-8')).hexdigest().upper();branch={**b,'status':completion['status'],'launch_authorized':False,'real_momentum_authorized':False,'allowed_stages':[],'evidence':ref(cp),'next_legal_action':'Use modern source correction and completed control to design LIA geometry repair or constrained modern thermal/sliding study under a new explicit version;no automatic launch.'}
    route=json.loads((PROJECT/s['authoritative_route']['path']).read_text(encoding='utf-8'));route['version']+='+polim-v15-modern-terminal';route[NAME]=branch
    if 'scientific_contract' in route:route['scientific_contract']={**s['scientific_contract'],'sha256':newhash}
    rp=EVIDENCE/'route-terminal.json';write_json(rp,route);new=copy.deepcopy(s);new['research_branches'][NAME]=branch;new['scientific_contract']['sha256']=newhash;new['authoritative_route']={**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']};new['updated_at']=now;new['state_id']+='+polim-v15-modern-terminal'
    for k in ['current_focus','counts','population','next_action']:
        if new[k]!=s[k]:raise ValueError('Mainline changed')
    if sp.read_bytes()!=before or facts.read_bytes()!=original:raise ValueError('Concurrent update')
    tmp=sp.with_name(f'.LIA_CURRENT_STATE.modernclose.{os.getpid()}.tmp');write_json(tmp,new);os.replace(tmp,sp)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {now} — Modern raw-F19 BP control completed\n\nOne solve,source geometry corrected from original member rasters,no extreme modern-model speeds,but generally slow. Modern/LIAand thermal/sliding limitations explicit. Facts{number};evidence `{ref(cp)["path"]}`. No automatic follow-up launch or mainline change.\n')
    with facts.open('ab') as f:f.write(addition.encode('utf-8'));f.flush();os.fsync(f.fileno())
    write_json(EVIDENCE/'terminal-validation.json',validate());print(json.dumps(dict(status=completion['status'],facts_section=number)),flush=True)

if __name__=='__main__':complete()
