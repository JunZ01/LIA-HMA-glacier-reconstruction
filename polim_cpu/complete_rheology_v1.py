"""Close the integrated prototype after delivery; preserve concurrent work."""
import copy,json,os,re
from datetime import datetime,timezone
from pathlib import Path
from .v10.registration import NAME,EVIDENCE,RUN,implementation
from .v10.report import REPORT
from .v10.storage import restore
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
    addition=f"""\n\n## {number}. Native BP material-rheology correction delivered\n\n- User explicitly requests repair of anomalous speed. The prior adapter assigned the inverse flowline effective A=7.538801981725612e-23 SI to pressure-adjusted -10C without that reference temperature being specified in the baseline table. This was an experimental assumption and is now superseded only for v10 BP material rheology; the original flowline calibration and historical outputs remain unchanged. Adopt PISM documented GPBLD configuration defaults,enhancement1,R8.31441,observed rheology water limit0.01; thermal drainage cap0.03 retained. No velocity clipping,parameter fitting,source geometry smoothing or Full-Stokes. Evidence and parameter sources are frozen.\n- Same paired native initial geometry,climate and open computational boundary as v9. One-hour BP/volume/enthalpy evolution completed in{result['total_substeps']}substeps. Initial maximum speed{first['max_speed_m_year']:.6f}m/year,final{last['max_speed_m_year']:.6f}m/year; old v9 initial509244.860715m/year read without replay. The difference diagnoses the material-law change on identical initial state; it does not certify the steep reconstructed surface or physical boundary conditions. This is not observed-speed calibration or scientific acceptance.\n- Relative mass budget error{result['checks']['relative_mass_budget']:.12g};horizontal enthalpy transport error{result['checks']['relative_transport_energy_budget']:.12g};finite,nonnegative and exact endpoint checks passed. Combined material-law/units/legacy-A-invariance and independent synthetic coupled restart passed; recorded output hashes and seven real checkpoint seals verified. Updated player and Open_Model.cmd point to v10. No accepted-count,mainline,unrelated branch or source-data changes. Subsequent runs need their own disclosed versioned contract. Evidence `{ref(EVIDENCE/'completion.json')['path']}`.\n"""
    newhash=__import__('hashlib').sha256(original+addition.encode('utf-8')).hexdigest().upper()
    record=dict(stage='terminal',status=completion['status'],plan=branch['plan'],terminal_evidence=ref(EVIDENCE/'completion.json'),source_route=state['authoritative_route'],created_at=now,accepted_count_effect=0)
    rp=EVIDENCE/'terminal-registration.json';write_json(rp,record)
    newbranch={**branch,'status':completion['status'],'launch_authorized':False,'real_momentum_authorized':False,'real_glacier_evolution_authorized':False,'allowed_stages':[],
        'evidence':ref(rp),'next_legal_action':'Use the delivered runnable prototype; subsequent durations or physics require a disclosed new version,not repeated completed states'}
    route=json.loads((PROJECT/state['authoritative_route']['path']).read_text(encoding='utf-8'));route['version']+='+polim-v10-rheology-terminal';route[NAME]=newbranch
    if 'scientific_contract' in route:route['scientific_contract']={**state['scientific_contract'],'sha256':newhash}
    target=EVIDENCE/'route-terminal.json';write_json(target,route)
    new=copy.deepcopy(state);new['research_branches'][NAME]=newbranch;new['scientific_contract']['sha256']=newhash
    new['authoritative_route']={**ref(target),'version':route['version'],'supersedes':state['authoritative_route']['version']};new['updated_at']=now;new['state_id']+='+polim-v10-rheology-terminal'
    for key in ('counts','population','current_focus','next_action'):
        if new[key]!=state[key]:raise ValueError('Mainline changed')
    if path.read_bytes()!=before or facts.read_bytes()!=original:raise ValueError('Concurrent controller/facts update')
    temp=path.with_name(f'.LIA_CURRENT_STATE.integratedclose.{os.getpid()}.tmp');write_json(temp,new);os.replace(temp,path)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {now} — Integrated BP prototype delivered\n\nNative2387 material-rheology repair completed with seven sealed coupled states and a before/after player. Unsupported inverse-A temperature normalization removed; same geometry retained. Combined checks passed; physical geometry and boundary limitations disclosed. Facts section{number},evidence `{ref(EVIDENCE/"completion.json")["path"]}`. No acceptance,mainline or unrelated branch changes.\n')
    with facts.open('ab') as f:f.write(addition.encode('utf-8'));f.flush();os.fsync(f.fileno())
    write_json(EVIDENCE/'terminal-controller-validation.json',validate())
    print(json.dumps(dict(status=completion['status'],facts_section=number,checks=result['checks'])))


if __name__=='__main__':complete()
