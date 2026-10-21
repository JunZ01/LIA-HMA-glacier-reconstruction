"""Close section-scale v1 and authorize uniform-thickness staged v2."""

from __future__ import annotations

import hashlib, json, os, shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
OUT = RESULTS / "lia_191_system2005_policy_v6_joint_candidate_v2_authorization"
ROUTE_DIR = RESULTS / "lia_191_current_route_v10_45"
FACTS, STATE, DECISIONS = ROOT / "LIA_PROJECT_FACTS.md", ROOT / "LIA_CURRENT_STATE.json", ROOT / "LIA_DECISION_LOG.md"
V1 = RESULTS / "lia_191_system2005_policy_v6_joint_candidate_v1/summary.json"
AUDIT = RESULTS / "lia_191_system2005_policy_v6_uniform_thickness_lift_audit_v2/summary.json"
CONFIG = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/system2005_policy_v6_joint_candidate_v2.json"
RUNNER = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/run_system2005_policy_v6_joint_candidate_v2.py"
EXPECTED = "1B1B73D426FDCA6EAB637DAAE90D53FDF7221F507A4F4EEEEE993F3EC13DC2BB"

def sha256(p):
    d=hashlib.sha256()
    with Path(p).open('rb') as h:
        for b in iter(lambda:h.read(1048576),b''): d.update(b)
    return d.hexdigest().upper()

def jb(v): return (json.dumps(v,indent=2,ensure_ascii=False)+'\n').encode('utf-8')

def atomic(p,data):
    p=Path(p); t=p.with_name(p.name+'.joint_v2_auth.tmp')
    if t.exists(): raise RuntimeError(f'stale temporary file:{t}')
    with t.open('xb') as h: h.write(data); h.flush(); os.fsync(h.fileno())
    os.replace(t,p)

def main():
    OUT.mkdir(parents=True,exist_ok=False); ROUTE_DIR.mkdir(parents=True,exist_ok=False)
    if sha256(FACTS)!=EXPECTED: raise RuntimeError(f'facts hash changed:{sha256(FACTS)}')
    old=json.loads(STATE.read_text(encoding='utf-8'))
    if old.get('state_id')!='2026-08-26-v10.44-joint-candidate-authorized': raise RuntimeError(f"unexpected state:{old.get('state_id')}")
    v1=json.loads(V1.read_text(encoding='utf-8')); audit=json.loads(AUDIT.read_text(encoding='utf-8'))
    if v1.get('status')!='static-fail-mechanism-closed': raise RuntimeError('v1 not closed at static gate')
    if audit.get('status')!='static-feasible': raise RuntimeError('v2 static audit not pass')
    for p in (CONFIG,RUNNER):
        if not p.exists(): raise FileNotFoundError(p)
    for s,n in ((FACTS,'LIA_PROJECT_FACTS.pre.md'),(STATE,'LIA_CURRENT_STATE.pre.json'),(DECISIONS,'LIA_DECISION_LOG.pre.md')): shutil.copy2(s,OUT/n)
    section="""## 356. System-2005 Policy-V6 uniform active-thickness candidate v2 (2026-08-26)

- The section-multiplier candidate v1 closed at its static gate without a
  dynamics run: although its volume and hypsometry gates passed, it increased
  ice area and created 18 within-branch and five junction reverse slopes.
- A distinct state representation adds one uniform `2.8052643 m` thickness
  increment only to nodes above the unchanged formal `1e-6 m` activity
  threshold. It reaches the same exact `1.02` volume scale while changing area
  by only `+0.2456%`; formal support, `25.550 km` connected length, free domain,
  all within/junction slopes, and topology remain unchanged. The 100 m
  hypsometry CDF difference is `0.4444%`.
- Version 1 of the lift audit is superseded only because it also lifted
  numerically positive nodes below the formal activity threshold. Version 2
  corrects that mask without changing the volume target, temperature,
  thresholds, or scientific mechanism.
- The v2 candidate retains `delta_t=-0.25 K` and is authorized for the same
  staged route: static recheck, conditional 10-year preflight, and at most one
  conditional 50-year formal steady run. Historical execution is forbidden.
- Evidence:
  `validation_data/results/paleo_surface_validation_v0/lia_191_system2005_policy_v6_joint_candidate_v1/summary.json`
  and
  `validation_data/results/paleo_surface_validation_v0/lia_191_system2005_policy_v6_uniform_thickness_lift_audit_v2/summary.json`.
""".rstrip()
    facts=FACTS.read_text(encoding='utf-8')
    if '## 356.' in facts: raise RuntimeError('Section356 exists')
    atomic(FACTS,(facts.rstrip()+'\n\n'+section+'\n').encode('utf-8'))
    route={'schema':'lia-191-current-route-v10-45','status':'policy-v6-uniform-thickness-joint-candidate-v2-authorized','population':191,'accepted':54,'remaining':137,'active':0,'queue':0,'launch_authorized':True,'full_historical_authorized':False,'supersedes':'lia_191_current_route_v10_44','system_id':2005,'candidate':{'source_year':9,'delta_t_c':-0.25,'uniform_active_thickness_lift_m':2.8052643058495814,'target_volume_scale':1.02},'authorized_stages':['static','conditional-10y','conditional-single-50y'],'candidate_count_max':1,'historical_forbidden':True,'audit':str(AUDIT.relative_to(ROOT)).replace('\\','/'),'config':str(CONFIG.relative_to(ROOT)).replace('\\','/')}
    rp=ROUTE_DIR/'summary.json'; atomic(rp,jb(route))
    state={'schema':'lia-191-controller-state-v1','state_id':'2026-08-26-v10.45-joint-candidate-v2-authorized','updated_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),'scientific_contract':{'path':'LIA_PROJECT_FACTS.md','sha256':sha256(FACTS)},'execution_contract':{'path':'LIA_EXECUTION_CONTRACT.md','version':'1.0'},'authoritative_route':{'version':'v10.45','path':str(rp.relative_to(ROOT)).replace('\\','/'),'sha256':sha256(rp),'supersedes':'v10.44'},'population':191,'counts':{'accepted':54,'remaining':137,'active':0,'queued':0},'current_focus':{'system_id':2005,'mainline_family':'policy-v6-uniform-thickness-temperature-joint-shooting','state':'one-staged-joint-candidate-v2-authorized','candidate_source_year':9,'candidate_delta_t_c':-0.25,'candidate_uniform_thickness_lift_m':2.8052643058495814,'minimum_length_for_20pct_separation_gate_m':25450.125918037134,'full_historical_authorized':False,'evidence':{'path':str(AUDIT.relative_to(ROOT)).replace('\\','/'),'sha256':sha256(AUDIT)}},'research_branches':old.get('research_branches',{}),'next_action':{'type':'system2005-joint-candidate-v2-static-preflight-formal50','launch_authorized':True,'description':'Run v2 static recheck, conditional 10y, then at most one conditional 50y; no historical run.'}}
    atomic(STATE,jb(state))
    decisions=DECISIONS.read_text(encoding='utf-8'); note="""## 2026-08-26 — System 2005 uniform-thickness joint candidate v2

- Section-scale v1 closed at static reverse-slope gates.
- Uniform active-thickness v2 passes every static feasibility gate.
- Conditional 10-year and one conditional 50-year run are authorized; historical is not.
""".strip(); atomic(DECISIONS,(decisions.rstrip()+'\n\n'+note+'\n').encode('utf-8'))
    summary={'schema':'lia-191-system2005-policy-v6-joint-candidate-v2-authorization','status':'staged-v2-authorized','facts_sha256':sha256(FACTS),'route_sha256':sha256(rp),'v1_sha256':sha256(V1),'audit_sha256':sha256(AUDIT),'config_sha256':sha256(CONFIG),'runner_sha256':sha256(RUNNER),'historical_authorized':False}
    atomic(OUT/'summary.json',jb(summary)); print(json.dumps(summary,indent=2,ensure_ascii=False)); return 0

if __name__=='__main__': raise SystemExit(main())
