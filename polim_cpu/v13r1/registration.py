import copy,json,os
from pathlib import Path
from datetime import datetime,timezone
from ..v13 import registration as parent
from ..v3.runner import PROJECT,digest,write_json
from ..thermal_attribution_v1 import validate,ref
ROOT=Path(__file__).resolve().parent
NAME=parent.NAME+'_technical_retry1'
EVIDENCE=PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME
RUN=ROOT.parent/'runs/v13-surface-first-geometry-r1'
REPORT=ROOT.parent/'reports/surface-first-geometry-v1'

def implementation():
    return {**parent.implementation(),**{p.relative_to(PROJECT).as_posix():digest(p) for p in sorted(ROOT.glob('*.py'))}}

def active():
    validate();s=json.loads((PROJECT/'LIA_CURRENT_STATE.json').read_text(encoding='utf-8'));b=s['research_branches'][NAME]
    if not b['launch_authorized'] or s['current_focus']['state']=='decision-required':raise ValueError('Not authorized')
    p=PROJECT/b['plan']['path']
    if digest(p)!=b['plan']['sha256'].lower():raise ValueError('Plan changed')
    plan=json.loads(p.read_text(encoding='utf-8'))
    if plan['implementation']!=implementation():raise ValueError('Code changed')
    for p,h in plan['input_hashes'].items():
        if digest(PROJECT/p)!=h:raise ValueError('Input changed')
    return plan

def register():
    validate();sp=PROJECT/'LIA_CURRENT_STATE.json';before=sp.read_bytes();s=json.loads(before)
    if NAME in s['research_branches']:raise ValueError('Retry already exists')
    failure=json.loads((parent.RUN/'failure.json').read_text(encoding='utf-8'))
    if failure['status']!='technical-failure':raise ValueError('No technical failure')
    EVIDENCE.mkdir(parents=True,exist_ok=False);previous=json.loads((parent.EVIDENCE/'plan.frozen.json').read_text(encoding='utf-8'))
    files=[parent.RUN/'mesh.npz',parent.RUN/'geometry.npz',parent.RUN/'geometry-summary.json',parent.RUN/'failure.json',parent.RUN/'fixture.json',parent.RUN/'real/ledger-001.json',parent.A_SOURCE,*[parent.RUN/f'real/frame-{i:03d}/{name}' for i in [0,1] for name in ['state.npz','metadata.json','seal.json']]]
    plan={**previous,'schema':'polim-surface-first-technical-recovery-v1','implementation':implementation(),'input_hashes':{**previous['input_hashes'],**{p.relative_to(PROJECT).as_posix():digest(p) for p in files}},'technical_cause':'Original first hour creates3090newly wet nodes,minH5.868616332153578e-34m;2769nodes have0<H<1e-12m. Sigma geometry divides by these unresolvable heights;physical linear residual fails at second-hour solve. First hour and initial solve completed and sealed.',
          'technical_fix':'Only momentum local geometry: rawH<=64*float64eps*max(1,abs(B),abs(B+rawH)) treated as zero. Stored and transported H,A untouched;no mass removal or artificial thickness floor. Existing exactly-dry elimination then applies. Physics,forcing,geometry constraints and scientific criteria unchanged. This is coordinate-roundoff handling,not metre-scale dry tolerance.',
          'recovery':'Restore and verify v13 frame001;reuse/copy frame000/001,H,A,u and prior ledger without recomputation. Continue hours2..24 only,new solver fingerprint,Newton and hourly checkpoints. One technical retry only;no automatic second retry.',
          'fixture':'One algebraic equivalence check: ordinary wet synthetic assembly equals original;sub-roundoff film assembly equals exact dry reference,without changing input H. Reuse previous geometry/restart evidence.',
          'source_failed_contract':failure['contract'],'resume_frame':1}
    pp=EVIDENCE/'plan.frozen.json';write_json(pp,plan);now=datetime.now(timezone.utc).isoformat();ep=EVIDENCE/'development-registration.json';write_json(ep,dict(status='technical-recovery-authorized',plan=ref(pp),source_failure=ref(parent.RUN/'failure.json'),source_route=s['authoritative_route'],created_at=now))
    old={**s['research_branches'][parent.NAME],'status':'technical-failure-after-hour1','launch_authorized':False,'real_momentum_authorized':False,'real_glacier_evolution_authorized':False,'allowed_stages':[],'evidence':ref(parent.RUN/'terminal-evidence.json'),'next_legal_action':'Only registered technical retry1 resumes from saved hour1'}
    branch={**old,'status':'technical-recovery-authorized','launch_authorized':True,'real_momentum_authorized':True,'real_glacier_evolution_authorized':True,'allowed_stages':['one-algebraic-roundoff-fixture','resume-hour1-to-hour24','review-and-deliver'],'plan':ref(pp),'evidence':ref(ep),'next_legal_action':'Complete only unfinished hours2..24 with same physics and roundoff-dry momentum repair'}
    route=json.loads((PROJECT/s['authoritative_route']['path']).read_text(encoding='utf-8'));route['version']+='+polim-v13r1-development';route[parent.NAME]=old;route[NAME]=branch;rp=EVIDENCE/'route-development.json';write_json(rp,route)
    new=copy.deepcopy(s);new['research_branches'][parent.NAME]=old;new['research_branches'][NAME]=branch;new['authoritative_route']={**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']};new['state_id']+='+polim-v13r1-development';new['updated_at']=now
    if sp.read_bytes()!=before:raise ValueError('Concurrent update')
    tmp=sp.with_name(f'.LIA_CURRENT_STATE.surface-r1.{os.getpid()}.tmp');write_json(tmp,new);os.replace(tmp,sp)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {now} — One surface-first technical recovery\n\nInitial BP andhour1 complete. Sub-ULP wet films caused second-hour linear failure. Register only coordinate-roundoff momentum dry classification,conserved H unchanged,and restore hour1 without replay. One algebraic fixture;no scientific parameter change,Full-Stokes or second automatic retry. Evidence `{ref(pp)["path"]}`.\n')
    write_json(EVIDENCE/'development-validation.json',validate());print('technical-recovery-authorized')
