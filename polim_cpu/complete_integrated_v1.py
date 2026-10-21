"""Close the integrated prototype after delivery; preserve concurrent work."""
import copy,json,os,re
from datetime import datetime,timezone
from pathlib import Path
from .v9.registration import NAME,EVIDENCE,RUN,implementation
from .v9.report import REPORT
from .v9.storage import restore
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
    addition=f'''\n\n## {number}. Runnable native multi-tributary BP evolution prototype delivered\n\n- User explicitly asks to stop extending checks and start the agreed simple runnable integrated model. One newly registered native2387 short evolution completed using exact paired178230node353096triangle mesh; no horizontal coarsening,thickness floor,smoothing or artificial branch connections. Real geometry comes only from paired mesh; prior v3 package supplies only permitted climate and parameters. BP,conservative median-dual volume/enthalpy transport,vertical thermal columns,basal melt and local Arrhenius/water fluidity feedback are connected. BP3 vertical nodes,thermal9 nodes; initial and6 ten-minute global momentum solves; one hour advanced in{result['total_substeps']}CFL0.35 substeps. Runtime{result['wall_seconds']:.6f}seconds.\n- Initial volume{result['initial_volume_m3']:.9f}m3;final{result['final_volume_m3']:.9f}m3;added{result['added_volume_m3']:.9f}m3;removed(surface/basal){result['removed_volume_m3']:.9f}m3;exported{result['exported_volume_m3']:.9f}m3. Full volume budget residual{result['mass_error_m3']:.12g}m3;relative{result['checks']['relative_mass_budget']:.12g}. Maximum normalized horizontal enthalpy transport budget error{result['checks']['relative_transport_energy_budget']:.12g};this does not independently certify the entire split thermal energy balance. Final state finite and nonnegative; all recorded momentum solves meet1e-6.\n- Maximum surface speed{first['max_speed_m_year']:.6f}to{last['max_speed_m_year']:.6f}m/year with temperature-dependent A;mean ice temperature{first['mean_temperature_C']:.9f}to{last['mean_temperature_C']:.9f}C. These are uncalibrated computational values. Prior constant-A diagnostic137244m/year remains immutable; new speeds do not resolve physical suitability. Open computational footprint uses zero effective lateral BP traction,one-way exterior ice export and no outside-domain advancing front. Actual rock/air/front classification remains unverified. This declared prototype assumes fixed1441 initial-elevation surface forcing for one hour,geothermal0.05W/m2,thin-column surface closure and omits resolved vertical advection. No historical reconstruction or accepted-count change.\n- Combined fixture gave exact constant-A inherited BP residual/tangent/load/energy and bitwise independent coupled restart for H,E,melt,u; status writer survived unavailable status destination. All7 real coupled checkpoint seals and recorded output hashes verified. Actual computed timeline has a player and a single Open_Model.cmd entry; opening completed results does not rerun them. No further simulation or repeat fingerprint authorized by closure. Evidence `{ref(EVIDENCE/'completion.json')['path']}`.\n'''
    newhash=__import__('hashlib').sha256(original+addition.encode('utf-8')).hexdigest().upper()
    record=dict(stage='terminal',status=completion['status'],plan=branch['plan'],terminal_evidence=ref(EVIDENCE/'completion.json'),source_route=state['authoritative_route'],created_at=now,accepted_count_effect=0)
    rp=EVIDENCE/'terminal-registration.json';write_json(rp,record)
    newbranch={**branch,'status':completion['status'],'launch_authorized':False,'real_momentum_authorized':False,'real_glacier_evolution_authorized':False,'allowed_stages':[],
        'evidence':ref(rp),'next_legal_action':'Use the delivered runnable prototype; subsequent durations or physics require a disclosed new version,not repeated completed states'}
    route=json.loads((PROJECT/state['authoritative_route']['path']).read_text(encoding='utf-8'));route['version']+='+polim-v9-integrated-terminal';route[NAME]=newbranch
    if 'scientific_contract' in route:route['scientific_contract']={**state['scientific_contract'],'sha256':newhash}
    target=EVIDENCE/'route-terminal.json';write_json(target,route)
    new=copy.deepcopy(state);new['research_branches'][NAME]=newbranch;new['scientific_contract']['sha256']=newhash
    new['authoritative_route']={**ref(target),'version':route['version'],'supersedes':state['authoritative_route']['version']};new['updated_at']=now;new['state_id']+='+polim-v9-integrated-terminal'
    for key in ('counts','population','current_focus','next_action'):
        if new[key]!=state[key]:raise ValueError('Mainline changed')
    if path.read_bytes()!=before or facts.read_bytes()!=original:raise ValueError('Concurrent controller/facts update')
    temp=path.with_name(f'.LIA_CURRENT_STATE.integratedclose.{os.getpid()}.tmp');write_json(temp,new);os.replace(temp,path)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {now} — Integrated BP prototype delivered\n\nNative2387 one-hour BP/volume/enthalpy/temperature timeline completed with seven sealed states and a working playback entry. Frozen combined checks passed; raw abnormal speeds retained and physical limitations disclosed. Facts section{number},evidence `{ref(EVIDENCE/"completion.json")["path"]}`. No acceptance,mainline or unrelated branch changes.\n')
    with facts.open('ab') as f:f.write(addition.encode('utf-8'));f.flush();os.fsync(f.fileno())
    write_json(EVIDENCE/'terminal-controller-validation.json',validate())
    print(json.dumps(dict(status=completion['status'],facts_section=number,checks=result['checks'])))


if __name__=='__main__':complete()
