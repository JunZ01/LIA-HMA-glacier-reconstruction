"""Read-only legacy alias/summary cache reconciliation. No model import."""
from . import lia_cpu_melt_factor_v10_279 as r
import subprocess
from pathlib import Path
def main():
    proposals=r.read(r.RES/'lia_cpu_melt_factor_design_v10_278_v1/spinup_matrix_proposed_7644.json');byid={}
    for x in proposals:byid.setdefault(x['lia_unique_id'],[]).append(x)
    files=subprocess.run(['rg','--files','--hidden','-g','!**/.git/**','-g','*summary*.json'],cwd=r.ROOT,capture_output=True,text=True,check=True).stdout.splitlines()
    matches=[];partial=[];count=0;bad=[];closed=[]
    alias={'delta_t_eq_degC':'delta_t_c','base_temp_bias_degC':'base_temp_bias_c','temp_melt_degC':'temp_melt_c'}
    for file in files:
        try:o=r.read(r.ROOT/file)
        except Exception as e:bad.append({'file':file,'error':repr(e)});continue
        if not isinstance(o,dict) or 'lia_unique_id' not in o or not isinstance(o.get('parameters'),dict):continue
        sid=int(o['lia_unique_id']);p=o['parameters'];params={k:p.get(k,p.get(alias.get(k,''))) for k in r.PARAMS};count+=1
        if sid in (1545,3584,2259,4959,2266) and o.get('layer')=='B':closed.append({'source':r.ref(r.ROOT/file),'lia_unique_id':sid,'spinup_window':o.get('spinup_window'),'spinup_control':o.get('spinup_control'),'parameters':p})
        for row in byid.get(sid,[]):
            known={k:v for k,v in params.items() if v is not None}
            if 'melt_f' not in known:continue
            if not all(abs(float(v)-row[k])<=max(abs(row[k])*1e-12,1e-30) for k,v in known.items()):continue
            record={'proposal_key':row['proposal_key'],'legacy_source':r.ref(r.ROOT/file),'missing_parameters':[k for k,v in params.items() if v is None],'spinup_window':o.get('spinup_window'),'spinup_control':o.get('spinup_control')}
            (partial if record['missing_parameters'] else matches).append(record)
    evidence={'status':'pass' if not(matches or partial or bad) else 'requires-reconciliation','summary_files_scanned':len(files),'legacy_parameter_records':count,'exact_physics_matches':matches,'possible_partial_matches':partial,'unreadable':bad,'named_closed_configuration_B_window_sources':closed,'interpretation':'Old B-window scalar families used the selected temperature offset during both spinup and historical forward. Current approved sequence changes the scientific question: spinup-only offset, fixed inherited historical bias q=0, consistent melt_f. No old closed scalar bracket is refined or reopened with its old historical temperature forcing; no old result is relabelled. Prior complete physical-tuple matches would block release regardless of version name.','source_code':[r.ref(r.DRIVERS/'lia_adaptive_spinup_chain.py'),r.ref(r.DRIVERS/'lia_first10_selected_chain.py')],'steps':0}
    r.write(r.TX/'legacy_cache_reconciliation.json',evidence)
    print({k:v for k,v in evidence.items() if k not in ('named_closed_configuration_B_window_sources','source_code','interpretation')},flush=True)
    if matches or partial or bad:raise SystemExit(2)
if __name__=='__main__':main()
