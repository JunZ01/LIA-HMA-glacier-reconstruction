"""Build immutable terminal control evidence, without editing canonical pointers."""
import copy
import json
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_lia_fixed_climate_forward_v10_276 import ROOT, TX, OUT, read, write, ref, sha, verify, now

state_path=ROOT/'LIA_CURRENT_STATE.json'
state=read(state_path)
state_hash=sha(state_path)
facts_hash=sha(ROOT/'LIA_PROJECT_FACTS.md')
verify(ROOT/state['scientific_contract']['path'],state['scientific_contract']['sha256'])
verify(ROOT/state['authoritative_route']['path'],state['authoritative_route']['sha256'])
for branch in state.get('research_branches',{}).values():
    verify(ROOT/branch['evidence']['path'],branch['evidence']['sha256'])
assert state['current_focus']['system_id']=='all1393-fixed-climate-historical-forward-cpu20'
assert state['counts']['accepted']==58 and state['counts']['remaining']==133
ps="Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'lia_fixed_climate_forward_v10_275\\.py' } | Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Depth 4"
process_snapshot=subprocess.run(['powershell','-NoProfile','-Command',ps],capture_output=True,text=True,check=True)
processes=json.loads(process_snapshot.stdout) if process_snapshot.stdout.strip() else []
assert not processes, 'A registered model process still exists; do not terminalize.'
process_evidence={'checked_at':now(),'registered_parent_pid':38052,'matching_model_processes':processes,
                  'read_only_query':ps,'query_stderr':process_snapshot.stderr,
                  'cpu_model_processes_alive':0,'workers_alive':0,'no_process_was_stopped':True}
write(OUT/'process_terminal_evidence.json',process_evidence)
analysis=read(OUT/'summary.json')
assert analysis['completed_cells']==1393 and analysis['technical_failures']==0
for item in analysis['outputs']:
    verify(ROOT/item['path'],item['sha256'])
figures=read(OUT/'figures_manifest.json')
assert figures['glacier_figures']==173 and figures['overview_pages']==11
for item in figures['outputs']:
    verify(ROOT/item['path'],item['sha256'])
logs=[]
classes=Counter()
unexpected=[]
paths=sorted((ROOT/'w275/l').glob('*.err.log'))+[ROOT/'w275/parent.err.log']
assert len(paths)==1394
for p in paths:
    text=p.read_text(encoding='utf-8',errors='replace')
    for line in text.splitlines():
        if 'header.dxf (GDAL_DATA is not defined)' in line:
            classes['GDAL optional header.dxf warning']+=1
        elif 'oggm.cfg:' in line:
            classes['OGGM configuration messages']+=1
        elif line.strip():
            unexpected.append({'log':str(p.relative_to(ROOT)),'line':line})
    logs.append({**ref(p),'bytes':p.stat().st_size})
write(OUT/'stderr_inventory.json',{'logs':logs,'line_classification':dict(classes),
                                  'unclassified_lines':unexpected,'parent_stderr_bytes':(ROOT/'w275/parent.err.log').stat().st_size})
parent=read(ROOT/'w275/r/parent.json')
elapsed=(datetime.fromisoformat(analysis['finished_at'])-datetime.fromisoformat(parent['started_at'])).total_seconds()
cells=pd.DataFrame(read(OUT/'scored_cells.json'))
glaciers=pd.DataFrame(read(OUT/'glacier_summary.json'))
vr=glaciers.end_volume_spread_fraction_of_initial.dropna()
write(OUT/'sensitivity_detail.json',{'source':ref(OUT/'glacier_summary_191.csv'),
     'systems_with_nonzero_initial_volume_spread':len(vr),
     'median_final_over_initial_absolute_volume_spread':float(vr.median()),
     'systems_larger_final_absolute_volume_spread':int((vr>1).sum()),
     'note':'Absolute between-parent volume spread, not relative volume error and not a climate-truth test. Length convergence does not imply complete-state convergence.'})
workbook=ROOT/'outputs/lia_cpu_forward_v10_276/冰川forward结果汇总.xlsx'
assert workbook.is_file()
checks=read(workbook.parent/'formula_check.json')
assert checks['actual']==checks['expected']
terminal={
 'schema':'lia-cpu-forward-formal-terminal-v10.276','created_at':now(),
 'status':'terminal-completed-and-scored','authorization':'User 开始做吧: read-only scoring, grouped curves, initial-state sensitivity and terminal registration.',
 'population':191,'completed_cells':1393,'forward_systems':173,'pending_cells':0,
 'technical_failures':0,'no_stable_systems_retained':18,'scientific_curve_nonpass_cells':1361,
 'provisional_curve_pass_cells':32,'provisional_curve_pass_systems':8,
 'provisional_system_ids':sorted(cells.loc[cells.provisional_curve_pass,'lia_unique_id'].unique().tolist()),
 'independent_reproduction':'not-run','accepted_ledger_effect':0,
 'started_at':parent['started_at'],'finished_at':analysis['finished_at'],'elapsed_wall_seconds':elapsed,
 'registered_parent':ref(ROOT/'w275/r/parent.json'),'source_transaction':ref(TX/'transaction_manifest.json'),
 'annual_rows_verified':739683,'checkpoint_commits_verified':26467,
 'full_checkpoint_state_restore_simulation':'not-rerun; this audit verifies source/result/checkpoint byte hashes, identity, parameters, topology and annual-state consistency',
 'process_evidence':ref(OUT/'process_terminal_evidence.json'),'analysis':ref(OUT/'summary.json'),
 'stderr_inventory':ref(OUT/'stderr_inventory.json'),'stderr_unclassified_line_count':len(unexpected),
 'figures':ref(OUT/'figures_manifest.json'),'workbook':ref(workbook),
 'workbook_checks':ref(workbook.parent/'formula_check.json'),'sensitivity_detail':ref(OUT/'sensitivity_detail.json'),
 'new_model_runs':0,'spinup_reruns':0,'launch_authorized':False,
 'prior_research_branches':copy.deepcopy(state['research_branches']),
 'prior_state':{'path':'LIA_CURRENT_STATE.json','sha256_at_read':state_hash},
 'prior_facts_sha256_at_read':facts_hash,
 'next_legal_action':'Review the 8 provisional systems and diagnose endpoint/direction/initial-state behavior from existing output; no automatic reproduction, climate correction, physics search, CPU rerun or GPU action.'}
write(OUT/'terminal_summary.json',terminal)
oldroute=read(ROOT/state['authoritative_route']['path'])
route=copy.deepcopy(oldroute)
route.update({'schema':'lia-current-route-v10.276','version':'v10.276-terminal-analysis',
 'status':'terminal-completed-and-scored','active':0,'queue':0,'launch_authorized':False,
 'full_historical_authorized':False,'evidence':ref(OUT/'terminal_summary.json'),
 'superseded_route':copy.deepcopy(state['authoritative_route']),
 'completed_forward_cells':1393,'provisional_curve_pass_cells':32,'provisional_curve_pass_systems':8,
 'no_stable_systems_retained':18,'new_model_runs':0,'accepted_ledger_effect':0,
 'population_ledger':ref(OUT/'glacier_summary_191.csv'),
 'population_ledger_role':'All191 ID coverage and current CPU stage; legacy accepted58 unchanged, no candidate promoted.',
 'next_action':terminal['next_legal_action']})
assert state_hash==sha(state_path) and facts_hash==sha(ROOT/'LIA_PROJECT_FACTS.md'), 'Concurrent canonical update; do not publish pointer.'
write(OUT/'route-terminal.json',route)
print(json.dumps({'terminal':ref(OUT/'terminal_summary.json'),'route':ref(OUT/'route-terminal.json'),
 'prior_state_sha256':state_hash,'prior_facts_sha256':facts_hash,'research_branches_preserved':True,
 'stderr_unclassified_lines':unexpected,'elapsed_minutes':elapsed/60,
 'volume_spread_median_ratio':float(vr.median()),'time':now()},indent=2))
