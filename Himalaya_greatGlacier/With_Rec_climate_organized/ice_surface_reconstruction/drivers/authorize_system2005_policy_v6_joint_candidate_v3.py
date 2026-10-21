"""Record v2 volume-only miss and authorize one response-corrected v3."""

from __future__ import annotations
import hashlib,json,os,shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[4]; RESULTS=ROOT/'validation_data/results/paleo_surface_validation_v0'
OUT=RESULTS/'lia_191_system2005_policy_v6_joint_candidate_v3_authorization'; ROUTE_DIR=RESULTS/'lia_191_current_route_v10_46'
FACTS,STATE,DECISIONS=ROOT/'LIA_PROJECT_FACTS.md',ROOT/'LIA_CURRENT_STATE.json',ROOT/'LIA_DECISION_LOG.md'
V2=RESULTS/'lia_191_system2005_policy_v6_joint_candidate_v2/summary.json'; DESIGN=RESULTS/'lia_191_system2005_policy_v6_uniform_thickness_v3_design_v2/summary.json'
CONFIG=ROOT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/system2005_policy_v6_joint_candidate_v3.json'
RUNNER=ROOT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/run_system2005_policy_v6_joint_candidate_v3.py'
EXPECTED='6852C49A2D9D75D3E7594A93A47D2A4064495D0CA9D47D7B84FEAC7A505CAA3B'

def sha(p):
 d=hashlib.sha256()
 with Path(p).open('rb') as h:
  for b in iter(lambda:h.read(1048576),b''): d.update(b)
 return d.hexdigest().upper()
def jb(v): return (json.dumps(v,indent=2,ensure_ascii=False)+'\n').encode()
def atomic(p,data):
 p=Path(p);t=p.with_name(p.name+'.v3auth.tmp')
 if t.exists(): raise RuntimeError('stale temp')
 with t.open('xb') as h: h.write(data);h.flush();os.fsync(h.fileno())
 os.replace(t,p)
def main():
 OUT.mkdir(parents=True,exist_ok=False);ROUTE_DIR.mkdir(parents=True,exist_ok=False)
 if sha(FACTS)!=EXPECTED: raise RuntimeError(f'facts changed:{sha(FACTS)}')
 old=json.loads(STATE.read_text(encoding='utf-8'))
 if old.get('state_id')!='2026-08-26-v10.45-joint-candidate-v2-authorized': raise RuntimeError('state changed')
 v2=json.loads(V2.read_text(encoding='utf-8'));design=json.loads(DESIGN.read_text(encoding='utf-8'))
 if v2.get('status')!='formal-50y-fail-mechanism-closed': raise RuntimeError('v2 not closed')
 if design.get('status')!='one-shot-v3-static-feasible': raise RuntimeError('v3 design not feasible')
 for p in (CONFIG,RUNNER):
  if not p.exists(): raise FileNotFoundError(p)
 for s,n in ((FACTS,'LIA_PROJECT_FACTS.pre.md'),(STATE,'LIA_CURRENT_STATE.pre.json'),(DECISIONS,'LIA_DECISION_LOG.pre.md')): shutil.copy2(s,OUT/n)
 section="""## 357. System-2005 Policy-V6 response-corrected uniform-thickness v3 (2026-08-26)

- Candidate v2 passed every static and 10-year gate. Its complete 50-year run
  retained `25.550 km` connected-root length and `5.900 km` free domain with
  clean numerical, nonnegative, and terminal diagnostics. Its only hard
  failure was volume change `+2.76996%`, exceeding the unchanged 2% gate.
- A single post-result state correction is explicitly disclosed. Holding
  `delta_t=-0.25 K` and every physical/numerical/scoring setting fixed, the v2
  response is used once to target `+1.30%` drift. The analytic scale is
  `1.034801176`, encoded as a uniform `4.876919 m` lift only on formally active
  nodes. No scalar search is authorized.
- The v3 static design passes: area change is `+0.4260%`, formal support and
  topology are unchanged, connected length remains `25.550 km`, within and
  junction reverse counts remain zero, and the 100 m hypsometry CDF difference
  is `0.5917%`.
- Exactly one staged v3 is authorized: static recheck, conditional 10-year
  preflight, and one conditional 50-year formal run. Historical execution
  remains forbidden. If v3 misses, this uniform-lift family closes.
- Evidence:
  `validation_data/results/paleo_surface_validation_v0/lia_191_system2005_policy_v6_joint_candidate_v2/summary.json`
  and
  `validation_data/results/paleo_surface_validation_v0/lia_191_system2005_policy_v6_uniform_thickness_v3_design_v2/summary.json`.
""".rstrip()
 facts=FACTS.read_text(encoding='utf-8')
 if '## 357.' in facts: raise RuntimeError('section exists')
 atomic(FACTS,(facts.rstrip()+'\n\n'+section+'\n').encode())
 route={'schema':'lia-191-current-route-v10-46','status':'policy-v6-response-corrected-joint-candidate-v3-authorized','population':191,'accepted':54,'remaining':137,'active':0,'queue':0,'launch_authorized':True,'full_historical_authorized':False,'supersedes':'lia_191_current_route_v10_45','system_id':2005,'candidate':{'source_year':9,'delta_t_c':-0.25,'uniform_active_thickness_lift_m':4.876919428200747,'target_volume_scale':1.0348011759452576},'authorized_stages':['static','conditional-10y','conditional-single-50y'],'candidate_count_max':1,'family_closes_if_fail':True,'historical_forbidden':True,'design':str(DESIGN.relative_to(ROOT)).replace('\\','/'),'config':str(CONFIG.relative_to(ROOT)).replace('\\','/')}
 rp=ROUTE_DIR/'summary.json';atomic(rp,jb(route))
 state={'schema':'lia-191-controller-state-v1','state_id':'2026-08-26-v10.46-joint-candidate-v3-authorized','updated_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),'scientific_contract':{'path':'LIA_PROJECT_FACTS.md','sha256':sha(FACTS)},'execution_contract':{'path':'LIA_EXECUTION_CONTRACT.md','version':'1.0'},'authoritative_route':{'version':'v10.46','path':str(rp.relative_to(ROOT)).replace('\\','/'),'sha256':sha(rp),'supersedes':'v10.45'},'population':191,'counts':{'accepted':54,'remaining':137,'active':0,'queued':0},'current_focus':{'system_id':2005,'mainline_family':'policy-v6-response-corrected-uniform-thickness-joint-shooting','state':'one-staged-joint-candidate-v3-authorized','candidate_source_year':9,'candidate_delta_t_c':-0.25,'candidate_uniform_thickness_lift_m':4.876919428200747,'minimum_length_for_20pct_separation_gate_m':25450.125918037134,'full_historical_authorized':False,'evidence':{'path':str(DESIGN.relative_to(ROOT)).replace('\\','/'),'sha256':sha(DESIGN)}},'research_branches':old.get('research_branches',{}),'next_action':{'type':'system2005-joint-candidate-v3-static-preflight-formal50','launch_authorized':True,'description':'Run one response-corrected v3; if it fails close uniform-lift family; no historical run.'}}
 atomic(STATE,jb(state));dec=DECISIONS.read_text(encoding='utf-8');note="""## 2026-08-26 — System 2005 response-corrected joint candidate v3

- V2 retained 25.550 km but missed volume at +2.770%.
- One disclosed analytic correction targets +1.30%; no search is allowed.
- Conditional 10-year and one 50-year v3 are authorized; historical is not.
""".strip();atomic(DECISIONS,(dec.rstrip()+'\n\n'+note+'\n').encode())
 summary={'schema':'lia-191-system2005-policy-v6-joint-candidate-v3-authorization','status':'staged-v3-authorized','facts_sha256':sha(FACTS),'route_sha256':sha(rp),'v2_sha256':sha(V2),'design_sha256':sha(DESIGN),'config_sha256':sha(CONFIG),'runner_sha256':sha(RUNNER),'historical_authorized':False}
 atomic(OUT/'summary.json',jb(summary));print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
