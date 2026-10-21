"""Close the integrated prototype after delivery; preserve concurrent work."""
import copy,json,os,re
from datetime import datetime,timezone
from pathlib import Path
from .v11.registration import NAME,EVIDENCE,RUN,implementation
from .v11.report import REPORT
from .v11.storage import restore
from .v3.runner import PROJECT,digest,write_json
from .thermal_attribution_v1 import validate,ref


def complete():
    validate();path=PROJECT/'LIA_CURRENT_STATE.json';before=path.read_bytes();state=json.loads(before)
    facts=PROJECT/'LIA_PROJECT_FACTS.md';original=facts.read_bytes()
    if digest(facts)!=state['scientific_contract']['sha256'].lower():raise ValueError('Concurrent facts conflict')
    final=json.loads((RUN/'terminal-evidence.json').read_text());result=json.loads((RUN/'result.json').read_text())
    if final['status']!='completed' or result['status']!='completed':raise ValueError('Prototype did not complete')
    if final['implementation']!=implementation():raise ValueError('Frozen implementation changed')
    for relative,sha in final['output_hashes'].items():
        if digest(RUN/relative)!=sha:raise ValueError(f'Evidence changed: {relative}')
    for frame in result['timeline']:restore(RUN/f'real/frame-{frame["frame"]:03d}',result['contract'])
    delivery=json.loads((REPORT/'delivery.json').read_text())
    for relative,sha in delivery['files'].items():
        if digest(REPORT/relative)!=sha:raise ValueError('Delivery changed')
    branch=state['research_branches'][NAME]
    if not branch['launch_authorized']:raise ValueError('Already closed')
    now=datetime.now(timezone.utc).isoformat();first=result['timeline'][0];last=result['timeline'][-1]
    completion=dict(status='completed-integrated-prototype',result=ref(RUN/'result.json'),numerical_evidence=ref(RUN/'terminal-evidence.json'),
        delivery=ref(REPORT/'delivery.json'),entry=ref(PROJECT/'polim_cpu/Open_Model.cmd'),all_output_hashes_verified=True,
        coupled_checkpoints_verified=len(result['timeline']),independent_restart='synthetic coupled state bitwise exact',accepted_count_effect=0,scientific_acceptance=False,created_at=now)
    write_json(EVIDENCE/'completion.json',completion)
    number=max(map(int,re.findall(r'^## (\d+)\.',original.decode('utf-8'),re.M)))+1
    addition=f"""\n\n## {number}. Constrained native BP bed-seam transition experiment delivered\n\n- User accepts source geometry diagnosis and evidenced repair. Read-only source reconstruction exactly reproduced original float32 bed. Peak node31985 lies at the modern-mask interface; the nearby 15.75m cell edge has raw DEM rise about9m versus bed rise34m. This establishes a computational source-selection seam,not proof that every modern margin has zero ice or no real cliff.\n- One disclosed derived-bed family blends the modern bed toward regional DEM within63m,chosen as the coarsest contributing source pixel(63/28/21m),with no speed target. All saved centreline bilinear supports plus one source-cell ring are fixed,with a source-scale transition away from pins. Original H,V,mesh/tributary connectivity,initial E and forcing,and v10 GPBLD parameters remain fixed. Raw source arrays remain immutable. Actual bed delta and exact derived mesh saved. This is a constrained reconstruction assumption,not recovered observed basal topography or formal geometry acceptance.\n- One-hour BP/volume/enthalpy evolution completed in{result['total_substeps']}substeps,{result['wall_seconds']:.6f}seconds. New initial maximum speed{first['max_speed_m_year']:.6f}m/year and final{last['max_speed_m_year']:.6f}m/year,compared to saved v10 initial2997.451299m/year without replay. Relative volume budget residual{result['checks']['relative_mass_budget']:.12g};horizontal enthalpy transport error{result['checks']['relative_transport_energy_budget']:.12g}. Material-law and independent synthetic restart checks passed; seven real sealed states and output hashes verified.\n- Deliver source/bed/surface profiles at peak and main confluence,planform correction map,and saved velocity/thickness player. No speed clipping,Full-Stokes,population promotion,mainline or unrelated branch change. Remaining high speeds and original thick ice on steep terrain are not resolved simply by this margin treatment. No automatic follow-up simulation. Evidence `{ref(EVIDENCE/'completion.json')['path']}`.\n"""
    newhash=__import__('hashlib').sha256(original+addition.encode('utf-8')).hexdigest().upper()
    record=dict(stage='terminal',status=completion['status'],plan=branch['plan'],terminal_evidence=ref(EVIDENCE/'completion.json'),source_route=state['authoritative_route'],created_at=now,accepted_count_effect=0)
    rp=EVIDENCE/'terminal-registration.json';write_json(rp,record)
    newbranch={**branch,'status':completion['status'],'launch_authorized':False,'real_momentum_authorized':False,'real_glacier_evolution_authorized':False,'allowed_stages':[],
        'evidence':ref(rp),'next_legal_action':'Use the delivered runnable prototype; subsequent durations or physics require a disclosed new version,not repeated completed states'}
    route=json.loads((PROJECT/state['authoritative_route']['path']).read_text(encoding='utf-8'));route['version']+='+polim-v11-bed-seam-terminal';route[NAME]=newbranch
    if 'scientific_contract' in route:route['scientific_contract']={**state['scientific_contract'],'sha256':newhash}
    target=EVIDENCE/'route-terminal.json';write_json(target,route)
    new=copy.deepcopy(state);new['research_branches'][NAME]=newbranch;new['scientific_contract']['sha256']=newhash
    new['authoritative_route']={**ref(target),'version':route['version'],'supersedes':state['authoritative_route']['version']};new['updated_at']=now;new['state_id']+='+polim-v11-bed-seam-terminal'
    for key in ('counts','population','current_focus','next_action'):
        if new[key]!=state[key]:raise ValueError('Mainline changed')
    if path.read_bytes()!=before or facts.read_bytes()!=original:raise ValueError('Concurrent controller/facts update')
    temp=path.with_name(f'.LIA_CURRENT_STATE.integratedclose.{os.getpid()}.tmp');write_json(temp,new);os.replace(temp,path)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {now} — Integrated BP prototype delivered\n\nNative2387 constrained modern-margin bed transition completed with source profiles and seven sealed states. Original H,volume,branch supports,material law and initial thermal/forcing state preserved. No source overwrite or physical acceptance claim. Facts section{number},evidence `{ref(EVIDENCE/"completion.json")["path"]}`. No acceptance,mainline or unrelated branch changes.\n')
    with facts.open('ab') as f:f.write(addition.encode('utf-8'));f.flush();os.fsync(f.fileno())
    write_json(EVIDENCE/'terminal-controller-validation.json',validate())
    print(json.dumps(dict(status=completion['status'],facts_section=number,checks=result['checks'])))


if __name__=='__main__':complete()
