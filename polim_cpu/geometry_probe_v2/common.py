import hashlib,json,time,os,uuid
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v42_geometric_probe_preparation_v2'
RUN=ROOT/'polim_cpu/runs/v42-geometric-probe-v2'
BRANCH='polim_v42_geometric_probe_v2'
BASE=ROOT/'polim_cpu/runs/v40-predictor-repair-v1/1552-quarter'
INPUT=ROOT/'polim_cpu/runs/v32r4-historical-1470-1570-v1/input.npz'
VALIDATOR=ROOT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def save(p,**arrays):
    with Path(p).open('xb') as f:np.savez(f,**arrays)
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def hashes(paths):return {Path(p).relative_to(ROOT).as_posix():digest(p) for p in paths}
def matches(expected):return {k:(ROOT/k).is_file() and digest(ROOT/k)==v for k,v in expected.items()}

def restore_base():
    z=arrays(BASE/'outer-08/state.npz');meta=json.loads(z.pop('metadata').tobytes())
    for k in ('h','u'):
        if ahash(z[k])!=meta[k+'_sha256']:raise RuntimeError('Base array hash mismatch')
    evidence=read(ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v40_predictor_repair_v1/completion.json')
    row=evidence['new_original_CPU_checks'][0]
    if any(meta[k+'_sha256']!=row[k+'_sha256'] for k in ('h','u')):raise RuntimeError('Saved H/u pair does not match v40 audited pair')
    if meta['contract']!=read(BASE/'contract.json')['contract'] or not meta['not_accepted'] or meta['outer']!=8:
        raise RuntimeError('Wrong candidate identity')
    return z['h'],z['u'],meta


def authorization_allowed(state,reg,proposal_hash,registration_hash):
    b=state.get('research_branches',{}).get(BRANCH,{})
    return bool(b.get('launch_authorized') is True and b.get('status')=='authorized'
        and b.get('plan',{}).get('sha256','').lower()==registration_hash.lower()
        and b.get('plan',{}).get('path')==reg.get('self_path')
        and reg.get('launch_authorized') is True and reg.get('status')=='registered'
        and reg.get('proposal_sha256')==proposal_hash and reg.get('branch')==BRANCH
        and reg.get('output_root')==RUN.as_posix()
        and reg.get('full_BP_max')==2 and reg.get('tangent_max')==1 and reg.get('time_steps')==0)


class Budget:
    def __init__(self,root,seconds=300):
        self.root=Path(root);self.start=time.perf_counter();self.seconds=seconds
        self.counts={k:0 for k in ['full_BP','tangent','residual','geometry_build','matrix_build','AMG_build',
                                   'BP_Krylov','tangent_Krylov','write_arrays','write_records']}
        self.times={k:0. for k in ['full_BP','tangent','residual_and_matrix','geometry_build','AMG_build','transport','write']}
    def tick(self):
        if time.perf_counter()-self.start>self.seconds:raise TimeoutError('Total wall budget exhausted')
    def enter(self,kind,limit=None):
        self.tick()
        if limit is not None and self.counts[kind]>=limit:raise RuntimeError(kind+' budget exhausted')
        self.counts[kind]+=1
    def record(self):return dict(counts=self.counts,times=self.times,total_wall=time.perf_counter()-self.start,
        timing_note='full_BP and tangent include nested residual/matrix/AMG costs; do not sum overlapping fields. Original evaluate combines residual/tangent assembly; matrix build count is separate, timing combined.')
    def progress(self):
        tmp=self.root/('.progress-'+uuid.uuid4().hex+'.json')
        tmp.write_text(json.dumps(self.record()),encoding='utf-8');os.replace(tmp,self.root/'work-progress.json')
    def json(self,name,data):
        t=time.perf_counter();write(self.root/name,data);self.counts['write_records']+=1;self.times['write']+=time.perf_counter()-t;self.progress()
    def npz(self,name,**data):
        t=time.perf_counter();save(self.root/name,**data);self.counts['write_arrays']+=1;self.times['write']+=time.perf_counter()-t;self.progress()
