"""One registered repair; retain completed cases and all sealed years."""
from .large_step_phase2_common_v1 import *
import sys, subprocess, copy, re, ast, shutil
from datetime import datetime, timezone
from .thermal_attribution_v1 import validate, ref
from .v3.runner import seal, write_json
from .large_step_phase2_trial_v1 import hashes

OLD_E=E;OLD_RUN=RUN
NAME='polim_v38r1_large_step_phase2_v1'
E=PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME
RUN=R/'runs/v38r1-large-step-phase2-v1'


def transition(stage,evidence,fact):
    validate();sp=PROJECT/'LIA_CURRENT_STATE.json';raw=sp.read_bytes();s=json.loads(raw)
    fp=PROJECT/'LIA_PROJECT_FACTS.md';fb=fp.read_bytes();assert s['current_focus']['state']!='decision-required'
    assert st.digest(fp).upper()==s['scientific_contract']['sha256'].upper()
    authorized=stage=='authorized'
    if authorized:assert NAME not in s['research_branches']
    else:assert s['research_branches'][NAME]['launch_authorized']
    branch=dict(status=stage,launch_authorized=authorized,
        allowed_stages=['three-unfinished-1553-to-1554-continuations'] if authorized else [],
        technical_retry_ordinal=1,automatic_further_retry=False,plan=ref(E/'plan.frozen.json'),evidence=ref(evidence),
        accepted_count_effect=0,production_replacement_authorized=False,phase3_authorized=False,baseline_replay_authorized=False)
    number=max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1
    addition=f'\n\n## {number}. System2387 BP phase2 single technical repair {stage}\n\n- {fact} Evidence `{ref(evidence)["path"]}`.\n'
    fh=hashlib.sha256(fb+addition.encode()).hexdigest().upper();route=st.read(PROJECT/s['authoritative_route']['path'])
    suffix=f'+{NAME}-{stage}';route['version']+=suffix;route[NAME]=branch
    if 'scientific_contract' in route:route['scientific_contract']={**s['scientific_contract'],'sha256':fh}
    rp=E/f'route-{stage}.json';write_json(rp,route);n=copy.deepcopy(s);n['research_branches'][NAME]=branch
    n['authoritative_route']={**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']}
    n['state_id']+=suffix;n['updated_at']=datetime.now(timezone.utc).isoformat();n['scientific_contract']['sha256']=fh
    assert sp.read_bytes()==raw and fp.read_bytes()==fb
    temporary=sp.with_name(f'.LIA_CURRENT_STATE.phase2retry-{os.getpid()}.tmp');write_json(temporary,n)
    with fp.open('ab') as f:f.write(addition.encode());f.flush();os.fsync(f.fileno())
    os.replace(temporary,sp)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {n["updated_at"]} — BP phase2 repair {stage}\n\n{fact}\n')
    write_json(E/f'{stage}-validation.json',validate())


def prepare():
    validate();s=st.read(PROJECT/'LIA_CURRENT_STATE.json')
    assert not s['research_branches']['polim_v38_large_step_phase2_v1']['launch_authorized']
    assert not E.exists() and not RUN.exists()
    oldplan=st.read(OLD_E/'plan.frozen.json')
    for path,digest in oldplan['code_hashes'].items():assert st.digest(PROJECT/path)==digest
    ast.parse((R/'large_step_phase2_worker_v2.py').read_text())
    E.mkdir();RUN.mkdir();cases=[];adoptions=[];inputs=[OLD_E/'completion.json',OLD_E/'plan.frozen.json',OLD_E/'phase1-audit.json']
    failure=[]
    for label in ('quarter','half','annual'):
        case='1552-'+label;old=OLD_RUN/case;root=RUN/case;root.mkdir()
        result=st.read(old/'result-full.json');assert result['status']=='technical-failure' and 'assert global_mass' in result['traceback']
        prior=st.read(old/'completion.json');cfg=st.read(old/'config.json')
        saved,meta=st.restore(old/'frame-001553',st.read(old/'contract.json')['contract'])
        assert meta['year']==1553. and meta['dt_cap']==.0625
        attempts=sorted(old.glob('attempt-*.json'));last=st.read(attempts[-1]);lost_id=last['id']+1
        failure.append(dict(case=case,error=result['error'],failed_candidate_year=result['year'],
            last_recorded_year=last['end'],last_cumulative_mass=last['global_mass_relative'],
            saved_year=1553.,final_attempt_telemetry_missing=True,missing_BP_calls_bounds=[1,8]))
        cfg.update(branch_name=NAME,audit_evidence_root=str(OLD_E),technical_retry_ordinal=1,
            actual_resume_year=1553.,prior_root=str(old),no_automatic_further_retry=True)
        st.write(root/'config.json',cfg);cases.append(cfg)
        clean={k:v for k,v in meta.items() if k not in ('contract','format','h_sha256','u_sha256')}
        # Work budgets cover the old failed process, not only the saved scientific prefix.
        clean.update(prior_seconds=prior['timing']['total_including_launch_restore_and_prefix'],
            attempt_count=lost_id,bp_calls=prior['counts']['bp_calls']+8,rejected=prior['counts']['rejected_steps']+1,
            bp_budget_counter_is_upper_bound=True,retained_prefix_source=str(old/'frame-001553'))
        for year in (1552,1553):
            for suffix in ('npz','json'):
                src=old/f'forcing-{year}.{suffix}';shutil.copyfile(src,root/src.name);inputs.append(src)
        adoptions.append((root,saved,clean));inputs += [root/'config.json',old/'frame-001553/state.npz',old/'completion.json',old/'result-full.json',old/'contract.json']+attempts
    review=dict(status='identified-technical-failure',retry_ordinal=1,cases=failure,
        cause='The phase2 driver checked cumulative volume error only AFTER provisional H/u and ledger mutation. Per-step errors individually below1e-10 can accumulate beyond the same window ledger gate. The assertion bypassed the registered correction/rejection path and omitted the last attempt telemetry.',
        repair='Check cumulative mass on each exact BP candidate inside the same8-call outer budget before accepting or mutating the committed ledger. If it fails, continue Picard correction; reject/halve only after the existing budget. Independently verify before commit. Persist interrupted attempt telemetry and total counters on any new failure.',
        preserved='Original BP/transport/forcing/damping1.0 and all tolerance numbers unchanged. Earlier1482 cases completed and remain untouched. Each1552 case resumes its own sealed1553 H/u and1553forcing. Only the permitted unfinished-year suffix is recomputed.',
        telemetry_limit='Old failed candidate was not saved. Its BP count is1..8 and Newton/CG counts cannot be reconstructed exactly without forbidden replay. All original wall and BP timing remains measured and is included. Report lower bounds rather than invent counters.',
        automatic_further_retry=False)
    st.write(E/'failure-review.json',review)
    plan=dict(schema=NAME,technical_retry_ordinal=1,user_authorization='Current phase2 instruction permits one identified versioned technical repair under the execution contract.',
        original_plan=ref(OLD_E/'plan.frozen.json'),failure_review=ref(E/'failure-review.json'),cases=cases,
        gates=GATES,budgets=BUDGET,physics_and_damping_unchanged=True,
        algorithm_revision='Move the already enforced cumulative1e-10 mass gate into coupled acceptance before commit; count every necessary final BP correction within8. No threshold relaxation.',
        actual_intervals=[[1553.,1554.]]*3,original_windows=[[1552.,1554.]]*3,
        retained_cases=['1482-quarter','1482-half','1482-annual'],
        retained_years='Each individual1552-1553 prefix; restore own reduced1/16year cap and own cached1553 forcing.',
        budgets_scope='Original process cost and conservative missing8BP allowance count toward the same900s,1152BP,144attempt and16rejection budgets. No further automatic retry.',
        code_hashes=hashes(),input_hashes={p.relative_to(PROJECT).as_posix():st.digest(p) for p in inputs},
        output_root=str(RUN),phase3_authorized=False,baseline_replay_authorized=False)
    st.write(E/'plan.frozen.json',plan);fingerprint=seal(plan);st.write(E/f'fingerprint-{fingerprint}.json',dict(root=str(RUN)))
    for root,saved,meta in adoptions:
        contract=seal(dict(batch=fingerprint,case=root.name));st.write(root/'contract.json',dict(contract=contract,batch=fingerprint))
        st.save(root,'frame-001553',saved['h'],saved['u'],meta,contract)
        restored,rm=st.restore(root/'frame-001553',contract);assert np.array_equal(restored['h'],saved['h']) and np.array_equal(restored['u'],saved['u'])
        st.write(root/'adoption.json',dict(original=meta['retained_prefix_source'],year=1553.,h_sha256=st.array_hash(saved['h']),u_sha256=st.array_hash(saved['u']),forcing=meta['forcing'],dynamics_replayed=False))
    transition('authorized',E/'plan.frozen.json',
        'One identified phase2 technical repair: cumulative mass gate was checked after in-memory state/ledger mutation, allowing per-step accepted errors to accumulate and then abort without final-attempt telemetry. Apply the existing cumulative1e-10 gate inside the same8BP candidate budget before commit; unchanged physics/forcing/damping1.0/residual and error thresholds. Resume only three unfinished1552-window members from their individual sealed1553 checkpoints and original1553 SMB, retained1/16year caps; completed1482 cases and1552-1553 years never replayed. Original costs/counters plus conservative missing8BP allowance remain inside existing budgets. Old missing final-attempt counters disclosed as bounds, not invented. No further automatic retry,stage3,default or count change.')
    print(json.dumps(dict(status='registered-single-repair',cases=3,fingerprint=fingerprint)),flush=True)


def execute():
    validate();plan=st.read(E/'plan.frozen.json');assert hashes()==plan['code_hashes']
    assert st.read(PROJECT/'LIA_CURRENT_STATE.json')['research_branches'][NAME]['launch_authorized']
    for p,d in plan['input_hashes'].items():assert st.digest(PROJECT/p)==d,p
    st.write(RUN/'execution-once.json',dict(pid=os.getpid(),started=datetime.now(timezone.utc).isoformat()))
    results=[st.read(OLD_RUN/f'1482-{label}'/'completion.json') for label in ('quarter','half','annual')]
    from .large_step_phase2_report_v1 import case_report
    for cfg in plan['cases']:
        root=RUN/cfg['case_id'];old=OLD_RUN/cfg['case_id'];prior=st.read(old/'completion.json')
        site=PROJECT/'.runtime/torch-cu128-v1/site-packages';win=Path(os.environ.get('SystemRoot','C:/Windows'))
        env=os.environ.copy();env.pop('KMP_DUPLICATE_LIB_OK',None);env.pop('PYTHONHOME',None)
        env['PATH']=os.pathsep.join(map(str,[site/'torch/lib',win/'System32',win]));env['PYTHONPATH']=os.pathsep.join(map(str,[site,PROJECT]));env['POLIM_TORCH_SITE']=str(site)
        command=['D:/Anaconda/envs/SS4/python.exe','-m','polim_cpu.large_step_phase2_worker_v2',str(root),'1554.0','repair']
        remaining=BUDGET['case_seconds']-prior['timing']['total_including_launch_restore_and_prefix'];start=time.perf_counter()
        with (root/'worker-repair.log').open('x') as log:
            p=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000)
            st.write(root/'launch-repair.json',dict(pid=p.pid,command=command,start=datetime.now(timezone.utc).isoformat(),output_root=str(root),stop_seconds=remaining,estimate_minutes=[1,3],checkpoint='retained1553 and final1554'))
            print(json.dumps(dict(launch=cfg['case_id'],resume=1553.,pid=p.pid)),flush=True)
            try:p.wait(timeout=max(remaining,1.))
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True);p.wait(timeout=30)
        elapsed=time.perf_counter()-start;st.write(root/'process-repair.json',dict(elapsed_seconds=elapsed,returncode=p.returncode))
        if not (root/'result-repair.json').exists():st.write(root/'result-repair.json',dict(status='technical-failure',segment='repair',error='worker terminated; inspect log',wall_seconds=elapsed,initialization_seconds=0.,restore_seconds=0.,io_seconds=0.))
        resumed=case_report(root);result=copy.deepcopy(resumed);result['technical_retry_ordinal']=1
        result['source_results']=[ref(old/'completion.json'),ref(root/'completion.json')]
        result['timing']['total_including_launch_restore_and_prefix']+=prior['timing']['total_including_launch_restore_and_prefix']
        result['timing']['original_failed_process']=prior['timing']['total_including_launch_restore_and_prefix']
        result['timing']['historical_v32_ratio']=result['timing']['v32_step_seconds']/result['timing']['total_including_launch_restore_and_prefix']
        result['retained_first_year_steps']=16
        result['steps']={'0.0625':16+resumed['counts']['accepted_steps']}
        result['counts']=dict(bp_calls_recorded=prior['counts']['bp_calls']+resumed['counts']['bp_calls'],
            missing_original_BP_calls_bounds=[1,8],newton_recorded=prior['counts']['newton']+resumed['counts']['newton'],
            cg_recorded=prior['counts']['cg']+resumed['counts']['cg'],original_failed_final_Newton_CG_missing=True,
            logical_accepted_steps=16+resumed['counts']['accepted_steps'],
            rejected_steps=prior['counts']['rejected_steps']+resumed['counts']['rejected_steps'],
            original_cumulative_gate_aborts=1,abandoned_unsaved_accepted_steps=prior['counts']['accepted_steps']-16)
        result['original_failed_attempts_included']=True;result['performance_2x_v32_proxy']=bool(result['accuracy_passed'] and result['timing']['historical_v32_ratio']>=2.)
        st.write(E/f"{cfg['case_id']}-accounted.json",result);results.append(result)
        print(json.dumps(dict(case=cfg['case_id'],status=result['status'],accuracy=result['accuracy_passed'])),flush=True)
    eligible=[]
    for cap in (.25,.5,1.):
        selected=[r for r in results if r['max_dt']==cap]
        if all(r['accuracy_passed'] for r in selected):eligible.append((sum(r['timing']['total_including_launch_restore_and_prefix'] for r in selected),cap))
    recommended=min(eligible)[1] if eligible else None
    summary=dict(status='completed',cases=results,recommended_max_dt=recommended,launch_authorized=False,phase3_authorized=False,
        technical_retry_ordinal=1,automatic_further_retry=False,no_baseline_replay=True,accepted_count_effect=0,
        v35_performance='Not established; no same-window control. Original failed-process costs included; lost old final-attempt counters disclosed.')
    st.write(E/'completion.json',summary)
    lines=['# BP phase 2 — six combinations after one disclosed technical repair','',
        '| Window | Requested cap yr | Actual steps yr | Completed | Accuracy | Total s including failures | v32 historical ratio |',
        '|---|---:|---|---|---|---:|---:|']
    for r in results:
        lines.append(f"| {r['start']}-{r['end']} | {r['max_dt']} | {r['steps']} | {r['status']} | {r['accuracy_passed']} | {r['timing']['total_including_launch_restore_and_prefix']:.3f} | {r['timing']['historical_v32_ratio']:.2f} |")
    lines+=['',f'Recommended cap: {recommended if recommended else "none; no cap passes both windows"}.','',
        'One technical repair moved the existing cumulative mass gate into the coupled acceptance loop before commit. Every candidate still has at most8 BP calls. Each1552 case retained its own1553 checkpoint; only its unsaved unfinished-year suffix was recomputed. All1482 cases remain unchanged.','',
        'All failed-process and resumed-process wall times are included. Per-component original and resumed timings remain in linked source results; original last-aborted-attempt BP count is only known to be1..8 and its Newton/CG trace is unavailable. Recorded counts are lower bounds for those three cases. This telemetry limitation is not repaired by rerunning a completed prefix.','',
        'v32 ratios use historical same-window row times and are not paired hardware benchmarks. v32 row times omit final accepted-checkpoint IO. No matching v35 control exists; its2x performance target is not established.','',
        '| Case/year | Volume % | H RMS m | H max m | Surface velocity % | Export error / gate m3 | CPU BP |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        for a in r['endpoints']:
            lines.append(f"| {r['case_id']}/{a['year']} | {100*a['volume_relative']:.6f} | {a['h_rms']:.6f} | {a['h_max']:.6f} | {100*a['velocity_relative']:.6f} | {a['export_difference']:.3f} / {a['export_limit']:.3f} | {a['original_CPU_BP']:.3g} |")
    for r in results:lines+=['',f"## {r['case_id']}",'',f"Counts: `{r['counts']}`.",f"Accuracy failures: `{r['failed_accuracy_gates']}`. Terminal failures: `{r['failures']}`."]
    (E/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    transition('completed',E/'completion.json',
        f"Six phase2 combinations finalized after the sole identified technical repair; completed={sum(r['status']=='completed' for r in results)}, accuracy-qualified={sum(r['accuracy_passed'] for r in results)}, recommended-cap={recommended}. All original failures and retry costs included; missing old aborted-attempt BP1..8 and Newton/CG traces explicitly bounded/unavailable. Completed1482 combinations and sealed1552-1553 states preserved. No repeated baseline, matchingv35control or proven2x-v35 gain. Launch authorization closed; no further retry,stage3,default,physics,mainline or count change.")
    print(json.dumps(dict(status='closed',recommended_cap=recommended,evidence=str(E/'summary.md'))),flush=True)


if __name__=='__main__':
    if sys.argv[-1]=='prepare':prepare()
    elif sys.argv[-1]=='execute':execute()
    else:raise SystemExit('prepare | execute')
