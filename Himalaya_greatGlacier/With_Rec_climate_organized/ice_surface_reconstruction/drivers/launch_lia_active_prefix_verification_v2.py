"""Register and supervise one independent optimized-backend verification."""

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from lia_gpu_four_group_search_v2 import ROOT, Candidate, digest, file_hash
from lia_gpu_matrix_io_v2 import WriterLock, atomic_json
from prepare_lia_gpu_four_group_v2 import OUTPUT
from register_lia_gpu_matrix_implementation_v2 import PATCHER, relative
from run_lia_gpu_four_group_v2 import check_inputs

HERE=Path(__file__).resolve().parent
TX=ROOT/'validation_data/results/paleo_surface_validation_v0/lia_gpu_active_prefix_long_verification_v2'
RUN=ROOT/'wgpfx2'
BRANCH='lia_gpu_active_prefix_long_verification_v2'
AUTH=TX/'authorization.json'
WORKER=HERE/'run_lia_active_prefix_verification_v2.py'


def now():
    return datetime.now(timezone.utc).isoformat()


def validate():
    subprocess.run([sys.executable,'-B',str(HERE/'validate_lia_execution_state_v1.py')],check=True)


def transition(label,evidence,launch=False):
    validate()
    state_path=ROOT/'LIA_CURRENT_STATE.json'
    before=file_hash(state_path)
    state=json.loads(state_path.read_text(encoding='utf-8-sig'))
    facts_path=ROOT/'LIA_PROJECT_FACTS.md'
    facts=facts_path.read_text(encoding='utf-8-sig')
    facts_before=file_hash(facts_path)
    if facts_before!=state['scientific_contract']['sha256']:
        raise RuntimeError('facts changed concurrently')
    prior=state['authoritative_route']
    route=json.loads((ROOT/prior['path']).read_text(encoding='utf-8-sig'))
    proof=dict(path=relative(evidence),sha256=file_hash(evidence))
    route['independent_gpu_active_prefix_verification']=dict(status=label,evidence=proof,launch_authorized=launch)
    route['superseded_route']=prior
    route_path=TX/('route-'+label+'.json')
    if route_path.exists():
        raise FileExistsError('immutable transition exists')
    atomic_json(route_path,route)
    descriptions={
        'authorized':'This is the one versioned technical continuation of the requested verification after V1 failed at model year zero because the monitoring counter attempted to serialize the initial infinite minimum timestep. V1 output is retained as terminal evidence. V2 represents an unavailable monitoring timestep as null, without changing the actual solver clock or numerical state. The user request starts the proposed real-sample continuous and disk-recovery comparison for the independent active-prefix GPU optimization. This scoped authorization permits only twelve fixed validation lanes, systems 2387, 584, 2534 and 2692 selected at grid-size ranks 0,63,126,190, each at delta_t_eq -1,0,+1. Both frozen and optimized implementations run from the same original inputs to year 330, with comparisons every 30 years. Optimized state is hash-committed at year 300; a separate process restores it and runs to year 330. These are labeled validation computations, not duplicate production results. The 4 percent validation GPU memory budget and separate wgpfx2 root are independent of existing wg2g1 production. This new route authorizes this research validation only, notwithstanding the separate CPU recovery decision; all CPU focus, recovery, queue and launch fields remain unchanged. No CPU launch, production interruption, production restart, forward or later parameter group is authorized. No physics, timestep, precision, domain or gate is changed.',
        'active':'The one authorized independent active-prefix validation has consumed launch authority. Its foreground supervisor and sequential numerical children own only wgpfx2. Existing GPU production and every CPU field remain unchanged; a validation failure affects only this verification. No implicit retry or production replacement is permitted.',
        'terminal':'The independent active-prefix validation supervisor and its children have terminated. Referenced evidence records actual comparisons, checkpoint restoration, timing and any failure. This is numerical verification, not glacier acceptance. Launch authority is false; no automatic production handoff, restart, forward run, later group or CPU control is performed.'}
    section=max(map(int,re.findall(r'^## (\d+)\.',facts,re.MULTILINE)))+1
    description=descriptions[label]
    addition='\n\n## '+str(section)+'. Independent active-prefix GPU verification '+label+' (2026-09-05)\n\n- '+description+'\n- Evidence: `'+relative(evidence)+'`. Accepted count effect is zero.\n'
    log_path=ROOT/'LIA_DECISION_LOG.md'
    log=log_path.read_text(encoding='utf-8-sig')
    patch='*** Begin Patch\n'
    for target,old,extra in ((facts_path,facts,addition),(log_path,log,'\n\n## GPU active-prefix verification '+label+'\n\n- '+description+'\n- Evidence: `'+relative(evidence)+'`.\n')):
        patch+='*** Update File: '+str(target).replace('\\','/')+'\n@@\n'
        patch+='\n'.join(' '+line for line in old.rstrip().splitlines()[-4:])+'\n'
        patch+='\n'.join('+'+line for line in extra.splitlines())+'\n'
    patch+='*** End Patch'
    if file_hash(state_path)!=before or file_hash(facts_path)!=facts_before:
        raise RuntimeError('concurrent shared update; no overwrite')
    subprocess.run([str(PATCHER),'--codex-run-as-apply-patch',patch],check=True)
    if file_hash(state_path)!=before:
        raise RuntimeError('concurrent state update after append; reconcile')
    updated=deepcopy(state)
    updated['scientific_contract']['sha256']=file_hash(facts_path)
    updated['state_id']=state['state_id']+'+gpu-prefix-verification-'+label
    updated['updated_at']=now()
    updated['authoritative_route']=dict(version=prior['version']+'+gpu-prefix-'+label,
        path=relative(route_path),sha256=file_hash(route_path),supersedes=prior['version'])
    updated['research_branches'][BRANCH]=dict(status=label,evidence=proof,launch_authorized=launch,
        authorization_sha256=file_hash(AUTH),output_root=relative(RUN),accepted_count_effect=0,
        production_replacement_authorized=False,existing_gpu_production_unchanged=True,
        cpu_control_unchanged=True,next_legal_action='run this verification only' if launch else 'report verification evidence only')
    excluded={'scientific_contract','state_id','updated_at','authoritative_route','research_branches'}
    assert all(updated[k]==state[k] for k in state.keys()-excluded)
    assert all(updated['research_branches'][k]==v for k,v in state['research_branches'].items() if k!=BRANCH)
    atomic_json(state_path,updated)
    validate()


def verify(active=False):
    manifest=check_inputs(OUTPUT)
    auth=json.loads(AUTH.read_text())
    basis={k:v for k,v in auth.items() if k!='fingerprint'}
    if digest(basis)!=auth['fingerprint'] or manifest['contract']!=auth['source_contract']:
        raise ValueError('validation contract mismatch')
    for name,expected in auth['source_files'].items():
        if Path(name).name!=name or file_hash(HERE/name)!=expected:
            raise ValueError('validation source changed: '+name)
    state=json.loads((ROOT/'LIA_CURRENT_STATE.json').read_text(encoding='utf-8-sig'))
    branch=state['research_branches'].get(BRANCH,{})
    if branch.get('authorization_sha256')!=file_hash(AUTH) or branch.get('status')!=('active' if active else 'authorized'):
        raise PermissionError('no matching scoped GPU validation authorization')
    if not active and branch.get('launch_authorized') is not True:
        raise PermissionError('launch consumed')
    return auth,manifest


def prepare():
    validate()
    manifest=check_inputs(OUTPUT)
    if TX.exists() or RUN.exists():
        raise FileExistsError('verification already registered; no duplicate')
    ordered=sorted(manifest['systems'],key=lambda r:(r['nx'],r['system']))
    systems=[ordered[i]['system'] for i in (0,63,126,190)]
    assert systems==[2387,584,2534,2692]
    cells=[asdict(Candidate(s,delta=d)) for s in systems for d in (-1.,0.,1.)]
    fixture=ROOT/'validation_data/results/paleo_surface_validation_v0/lia_gpu_active_prefix_cuda_20260905T083135Z/summary.json'
    assert json.loads(fixture.read_text())['status']=='fixture-and-synthetic-pass'
    source_files={p.name:file_hash(p) for p in (Path(__file__),WORKER,HERE/'torch_flowline_1d_active_prefix_v1.py')}
    auth=dict(schema='lia-gpu-active-prefix-long-verification-v2',user_authorization='Start the proposed real-sample continuous and disk-recovery comparison',
        technical_retry_ordinal=1,prior_failure='lia_gpu_active_prefix_long_verification_v1',source_contract=manifest['contract'],candidates=cells,selection='nx rank 0,63,126,190, then system ID; delta -1,0,+1',
        source_files=source_files,fixture=dict(path=relative(fixture),sha256=file_hash(fixture)),
        output_root=relative(RUN),max_year=330,checkpoint_year=300,comparison_interval_years=30,
        gpu_memory_fraction=.04,phase_timeout_seconds=3600,estimated_total_seconds=[600,1800],
        precision='float64',thickness_atol_m=1e-6,volume_rtol=1e-8,area_rtol=1e-10,length_exact=True,
        budget_rtol=1e-10,budget_atol_m3=1e-6,science_success_required=False,created_at_utc=now(),
        cpu_control_unchanged=True,existing_gpu_production_unchanged=True,production_replacement_authorized=False)
    auth['fingerprint']=digest(auth)
    TX.mkdir(parents=True)
    atomic_json(AUTH,auth)
    transition('authorized',AUTH,True)
    print(json.dumps(dict(status='registered',systems=systems,lanes=12,years=330)),flush=True)


def run():
    validate()
    auth,_=verify()
    from launch_lia_191_gpu_spinup_v10_239 import child_environment,SS4_PYTHON
    result=dict(status='technical-failure',contract=auth['fingerprint'],phases=[],production_unchanged=True,cpu_unchanged=True)
    with WriterLock(RUN):
        if (RUN/'launch.json').exists():
            raise FileExistsError('verification already launched')
        active=dict(pid=os.getpid(),started_at_utc=now(),contract=auth['fingerprint'],command=sys.argv)
        atomic_json(RUN/'launch.json',active)
        atomic_json(TX/'active.json',active)
        transition('active',TX/'active.json')
        try:
            for phase in ('continuous','resume'):
                command=[str(SS4_PYTHON),'-B',str(WORKER),phase]
                with (TX/(phase+'.stdout.log')).open('x') as out,(TX/(phase+'.stderr.log')).open('x') as err:
                    child=subprocess.Popen(command,cwd=ROOT,env=child_environment(),stdout=out,stderr=err)
                    atomic_json(TX/(phase+'-launch.json'),dict(pid=child.pid,command=command,started_at_utc=now()))
                    print(json.dumps(dict(phase=phase,pid=child.pid,status='started')),flush=True)
                    try:
                        code=child.wait(timeout=auth['phase_timeout_seconds'])
                    except subprocess.TimeoutExpired:
                        child.kill(); child.wait()
                        raise RuntimeError('verification-only child exceeded time budget')
                summary=RUN/phase/'summary.json'
                record=json.loads(summary.read_text()) if summary.exists() else None
                result['phases'].append(dict(phase=phase,exit_code=code,summary=record,
                    stderr_sha256=file_hash(TX/(phase+'.stderr.log'))))
                if code or not record or record['status']!='pass':
                    raise RuntimeError('verification phase failed: '+phase)
            result['status']='pass'
            result['long_run_comparison_passed']=True
            result['cross_process_disk_resume_passed']=True
        except Exception as error:
            result['error']=repr(error)
        result['finished_at_utc']=now()
        result['production_replacement_performed']=False
        atomic_json(TX/'terminal.json',result)
        transition('terminal',TX/'terminal.json')
    print(json.dumps(result,indent=2),flush=True)
    return 0 if result['status']=='pass' else 1


if __name__=='__main__':
    if sys.argv[1:]==['--prepare']:
        prepare()
    elif sys.argv[1:]==['--run']:
        raise SystemExit(run())
    else:
        raise SystemExit('Use --prepare or --run; no implicit launch')

