import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,hashlib,copy,re,datetime,subprocess,traceback
from pathlib import Path
from polim_cpu.geometry_probe_v2.common import ROOT,INPUT,read,write,arrays,digest,hashes,matches,ahash
from polim_cpu.geometry_probe_v2.driver import validate,environment
NAME='polim_v52_joint_block_v1'
E=ROOT/'validation_data/results/paleo_surface_validation_v0'/NAME
RUN=ROOT/'polim_cpu/runs/v52-joint-block-v1'
OLD=ROOT/'polim_cpu/runs/v48-cn-cuda-v1'
PREV=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v50r2_tangent_runtime_v1'
PREVRUN=ROOT/'polim_cpu/runs/v50r2-tangent-runtime-v1'
PLAN=E/'plan.frozen.json'
def ref(p):return dict(path=Path(p).relative_to(ROOT).as_posix(),sha256=digest(p).upper())

def gate(stage,integrity=False):
    start=time.perf_counter();v=validate();s=read(ROOT/'LIA_CURRENT_STATE.json');p=read(PLAN)
    assert s['research_branches'][NAME]['launch_authorized'] and s['research_branches'][NAME]['plan']==ref(PLAN)
    seconds_auth=time.perf_counter()-start;files=0
    if integrity:
        manifest={}
        for key in ['code_hashes','input_hashes','protected_hashes']:
            for path,sha in p[key].items():
                if path in manifest:assert manifest[path].lower()==sha.lower()
                manifest[path]=sha
        failures=[x for x,ok in matches(manifest).items() if not ok]
        if failures:raise RuntimeError('SHA256 mismatch: '+str(failures))
        files=len(manifest);assert environment()==p['environment']
    if RUN.exists():
        with (RUN/'management.jsonl').open('a') as f:f.write(json.dumps(dict(pid=os.getpid(),stage=stage,integrity=integrity,files=files,authorization_seconds=seconds_auth,seconds=time.perf_counter()-start,wall=time.time()))+'\n')
    return p

def transition(stage,evidence,note):
    validate();sp=ROOT/'LIA_CURRENT_STATE.json';fp=ROOT/'LIA_PROJECT_FACTS.md';sb=sp.read_bytes();fb=fp.read_bytes();s=json.loads(sb)
    assert s['current_focus']['state']!='decision-required' and digest(fp).upper()==s['scientific_contract']['sha256'].upper()
    active=stage=='authorized'
    if active:assert NAME not in s['research_branches']
    else:assert s['research_branches'][NAME]['launch_authorized']
    b=dict(status=stage,launch_authorized=active,allowed_stages=['N-quarter','D-quarter'] if active else [],
        plan=ref(PLAN),evidence=ref(evidence),accepted_count_effect=0,production_replacement_authorized=False,
        baseline_replay_authorized=False,automatic_retry=False,phase3_authorized=False)
    num=max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1;now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    addition=f'\n\n## {num}. v52 joint block/new-wet development: {stage}\n\n{note}\n\nEvidence: `{evidence.relative_to(ROOT).as_posix()}`, SHA256 `{digest(evidence)}`. Mainline, defaults, physics and 191-system ledger unchanged.\n'
    fh=hashlib.sha256(fb+addition.encode()).hexdigest().upper();route=read(ROOT/s['authoritative_route']['path']);suffix='+'+NAME+'-'+stage
    route['version']+=suffix;route[NAME]=b
    if 'scientific_contract' in route:route['scientific_contract']['sha256']=fh
    rp=E/('route-'+stage+'.json');write(rp,route);n=copy.deepcopy(s);n['research_branches'][NAME]=b
    n['authoritative_route']={**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']}
    n['state_id']+=suffix;n['scientific_contract']['sha256']=fh;n['updated_at']=now
    assert sp.read_bytes()==sb and fp.read_bytes()==fb
    tmp=E/('state-'+stage+'.tmp');write(tmp,n)
    with fp.open('ab') as f:f.write(addition.encode());f.flush();os.fsync(f.fileno())
    os.replace(tmp,sp)
    with (ROOT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {now} {NAME} {stage}\n\n{note}\n')
    v=validate();write(E/('controller-'+stage+'.json'),v);return v

