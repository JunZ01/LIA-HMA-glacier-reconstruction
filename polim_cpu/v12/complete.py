"""Close the integrated prototype after delivery; preserve concurrent work."""
import copy,json,os,re
from datetime import datetime,timezone
from pathlib import Path
from .registration import NAME,EVIDENCE,RUN,implementation
from .report import REPORT
from ..v9.storage import restore
from ..v3.runner import PROJECT,digest,write_json
from ..thermal_attribution_v1 import validate,ref


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
        coupled_checkpoints_verified=len(result['timeline']),independent_restart='synthetic zero-source fixed-material H,A,u restart verified',accepted_count_effect=0,scientific_acceptance=False,created_at=now)
    write_json(EVIDENCE/'completion.json',completion)
    number=max(map(int,re.findall(r'^## (\d+)\.',original.decode('utf-8'),re.M)))+1
    addition=f"""\n\n## {number}. One-day fixed-material native BP relaxation completed\n\n- User explicitly starts one-day BP dynamical relaxation from current default v10 original mesh. Initial H and bed match the authoritative paired source exactly; saved v10 initial velocity reused after same-equation residual certification,not solved again. Compute material A once from saved initial E,H using v10 GPBLD law,then freeze A bitwise at horizontal/sigma nodes. No climate read,temperature/enthalpy evolution,geothermal heating,accumulation,ablation or basal melt. Basal no-slip and open zero-effective-lateral-traction footprint retained; boundary outflow recorded,no external inflow or outside-domain advance.\n- Completed24hours with24 hourly BP updates and{result['total_substeps']}CFL0.35 transport substeps in{result['wall_seconds']:.6f}seconds. Initial/final maximum velocity{first['max_speed_m_year']:.6f}/{last['max_speed_m_year']:.6f}m/year. Initial-fastest-region mean{first['initial_hot_region_mean_speed_m_year']:.6f}/{last['initial_hot_region_mean_speed_m_year']:.6f}m/year;region fixed as initial fastest1percent among H>=1m nodes,area weighted. Maximum local H change{last['maximum_H_change_m']:.6f}m. Boundary export{result['exported_volume_m3']:.9f}m3;global budget error{result['mass_error_m3']:.12g}m3;relative{result['checks']['relative_mass_budget']:.12g}. Original major-junction126m dual-volume budget error{result['junction_budget_error_m3']:.12g}m3,relative{result['checks']['relative_junction_budget']:.12g}.\n- Constant A,zero mass sources,finite nonnegative H and exact final time verified. One combined small zero-source fixed-material fixture and independent H,A,u checkpoint restart passed; previous BP benchmarks and status-survival evidence reused. All25 real endpoint seals and recorded output hashes verified. This is a one-day response diagnostic,not proof of steady equilibrium,actual ice geometry or a historical reconstruction. No speed clipping,geometry smoothing,Full-Stokes,accepted-count change or automatic extension. Evidence `{ref(EVIDENCE/'completion.json')['path']}`.\n"""
    newhash=__import__('hashlib').sha256(original+addition.encode('utf-8')).hexdigest().upper()
    record=dict(stage='terminal',status=completion['status'],plan=branch['plan'],terminal_evidence=ref(EVIDENCE/'completion.json'),source_route=state['authoritative_route'],created_at=now,accepted_count_effect=0)
    rp=EVIDENCE/'terminal-registration.json';write_json(rp,record)
    newbranch={**branch,'status':completion['status'],'launch_authorized':False,'real_momentum_authorized':False,'real_glacier_evolution_authorized':False,'allowed_stages':[],
        'evidence':ref(rp),'next_legal_action':'Use the delivered runnable prototype; subsequent durations or physics require a disclosed new version,not repeated completed states'}
    route=json.loads((PROJECT/state['authoritative_route']['path']).read_text(encoding='utf-8'));route['version']+='+polim-v12-relaxation-terminal';route[NAME]=newbranch
    if 'scientific_contract' in route:route['scientific_contract']={**state['scientific_contract'],'sha256':newhash}
    target=EVIDENCE/'route-terminal.json';write_json(target,route)
    new=copy.deepcopy(state);new['research_branches'][NAME]=newbranch;new['scientific_contract']['sha256']=newhash
    new['authoritative_route']={**ref(target),'version':route['version'],'supersedes':state['authoritative_route']['version']};new['updated_at']=now;new['state_id']+='+polim-v12-relaxation-terminal'
    for key in ('counts','population','current_focus','next_action'):
        if new[key]!=state[key]:raise ValueError('Mainline changed')
    if path.read_bytes()!=before or facts.read_bytes()!=original:raise ValueError('Concurrent controller/facts update')
    temp=path.with_name(f'.LIA_CURRENT_STATE.integratedclose.{os.getpid()}.tmp');write_json(temp,new);os.replace(temp,path)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {now} — Integrated BP prototype delivered\n\nNative2387 one-day fixed-material zero-source BP relaxation completed with25 sealed states,explicit global/junction volume budgets and a saved player. Original bed and source inputs preserved; no equilibrium or physical acceptance claim. Facts section{number},evidence `{ref(EVIDENCE/"completion.json")["path"]}`. No acceptance,mainline or unrelated branch changes.\n')
    with facts.open('ab') as f:f.write(addition.encode('utf-8'));f.flush();os.fsync(f.fileno())
    write_json(EVIDENCE/'terminal-controller-validation.json',validate())
    print(json.dumps(dict(status=completion['status'],facts_section=number,checks=result['checks'])))


if __name__=='__main__':complete()
