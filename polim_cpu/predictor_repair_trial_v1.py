"""Single registered matched failed-step comparison for confirmed predictor omissions."""
from .large_step_phase2_common_v1 import *
import sys,copy,re,subprocess,ast
from datetime import datetime,timezone
from .thermal_attribution_v1 import validate,ref
from .v3.runner import seal,write_json
from .diffusion_predictor_verify_v2 import verify
from .large_step_phase2_trial_v1 import hashes

NAME='polim_v40_predictor_repair_v1'
E=PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME
RUN=R/'runs/v40-predictor-repair-v1/1552-quarter'
PRIOR=R/'runs/v38-large-step-phase2-v1/1552-quarter'


def transition(stage,evidence,fact):
    validate();sp=PROJECT/'LIA_CURRENT_STATE.json';raw=sp.read_bytes();s=json.loads(raw)
    fp=PROJECT/'LIA_PROJECT_FACTS.md';fb=fp.read_bytes();assert s['current_focus']['state']!='decision-required'
    assert st.digest(fp).upper()==s['scientific_contract']['sha256'].upper()
    authorized=stage=='authorized'
    if authorized:assert NAME not in s['research_branches']
    else:assert s['research_branches'][NAME]['launch_authorized']
    branch=dict(status=stage,launch_authorized=authorized,allowed_stages=['one-1552-quarter-candidate-predictor-comparison'] if authorized else [],
        plan=ref(E/'plan.frozen.json'),evidence=ref(evidence),accepted_count_effect=0,production_replacement_authorized=False,
        baseline_replay_authorized=False,automatic_retry=False,phase3_authorized=False)
    number=max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1
    addition=f'\n\n## {number}. System2387 diffusion predictor correction {stage}\n\n- {fact} Evidence `{ref(evidence)["path"]}`.\n'
    fh=hashlib.sha256(fb+addition.encode()).hexdigest().upper();route=st.read(PROJECT/s['authoritative_route']['path']);suffix=f'+{NAME}-{stage}'
    route['version']+=suffix;route[NAME]=branch
    if 'scientific_contract' in route:route['scientific_contract']={**s['scientific_contract'],'sha256':fh}
    rp=E/f'route-{stage}.json';write_json(rp,route);n=copy.deepcopy(s);n['research_branches'][NAME]=branch
    n['authoritative_route']={**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']}
    n['state_id']+=suffix;n['scientific_contract']['sha256']=fh;n['updated_at']=datetime.now(timezone.utc).isoformat()
    assert sp.read_bytes()==raw and fp.read_bytes()==fb
    tmp=sp.with_name(f'.LIA_CURRENT_STATE.predictor-{os.getpid()}.tmp');write_json(tmp,n)
    with fp.open('ab') as f:f.write(addition.encode());f.flush();os.fsync(f.fileno())
    os.replace(tmp,sp)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {n["updated_at"]} — predictor {stage}\n\n{fact}\n')
    write_json(E/f'{stage}-validation.json',validate())


def prepare():
    validate();assert not E.exists() and not RUN.parent.exists()
    prior=st.read(PRIOR/'attempt-0001.json')
    assert prior['start']==1552. and prior['dt']==.25 and prior['status']=='rejected' and prior['bp_calls']==8
    assert len(prior['history'])==8 and all(x['nonlinear_history'] for x in prior['history'])
    oldplan=st.read(PROJECT/'validation_data/results/paleo_surface_validation_v0/polim_v38_large_step_phase2_v1/plan.frozen.json')
    # The compared outer loop, BP and physical transport have not been edited.
    for p,h in oldplan['code_hashes'].items():assert st.digest(PROJECT/p)==h,p
    saved,meta=st.restore(PRIOR/'frame-000000',st.read(PRIOR/'contract.json')['contract']);baseline,bm,bpath=reference(1552)
    assert np.array_equal(saved['h'],baseline['h']) and np.array_equal(saved['u'],baseline['u'])
    assert st.digest(PRIOR/'forcing-1552.npz')==st.read(PRIOR/'forcing-1552.json')['sha256']
    for p in [R/'diffusion_predictor_v2.py',R/'predictor_repair_worker_v1.py']:ast.parse(p.read_text(encoding='utf-8'))
    fixture=verify();assert fixture['status']=='pass'
    E.mkdir();RUN.mkdir(parents=True);st.write(E/'algebraic-fixtures.json',fixture)
    cfg=dict(branch_name=NAME,source_root=str(SOURCE),prior_case=str(PRIOR),start=1552.,end=1552.25,dt=.25,
        h_sha256=st.array_hash(saved['h']),u_sha256=st.array_hash(saved['u']),forcing_sha256=st.digest(PRIOR/'forcing-1552.npz'),
        damping=1.,bp_calls_max=8,attempts_max=1,halving=False)
    st.write(RUN/'config.json',cfg)
    inputs=[PRIOR/'attempt-0001.json',PRIOR/'frame-000000/state.npz',PRIOR/'contract.json',PRIOR/'config.json',
        PRIOR/'forcing-1552.npz',PRIOR/'forcing-1552.json',SOURCE/'input.npz',SOURCE/'contract.json',SOURCE/'config.json',
        bpath/'state.npz',bpath/'metadata.json',bpath/'seal.json',RUN/'config.json',E/'algebraic-fixtures.json']
    plan=dict(schema=NAME,scope='Exactly one previously failed1552.0-1552.25 attempt; change predictor only and reuse original failure as control without replay.',
        user_authorization='User prioritizes confirmed predictor repair and proposes same-checkpoint/same-dt/eight-BP matched failed-candidate verification.',
        comparison=ref(PRIOR/'attempt-0001.json'),config=cfg,
        repair='g=abs(C)*H_donor/max(abs(ds_old),1m); implicit outward-rate donor diagonal. Frozen signed face remainder Q_old-g*ds_old anchors old formal flux on uphill/flat/limited faces. Predictor only; formal phase2 correction and original BP imported unchanged.',
        approximation_limits='Bare diffusion is not flux-equivalent on reversed slopes or cutoff faces. The signed remainder gives equality at OLD geometry only. Predicted new face flux can reverse as geometry changes; count and report this, never treat predictor as physical output. This tests the predictor repair bundle, not attribution of each individual term.',
        unchanged=['source H/u hashes','annual1552SMB cache','dt0.25year','omega1.0','8completeBP maximum including final correction','original BP tol1e-6','local1e-8','per-step mass1e-10','fixed-point1e-8','active-set equality','fixed bed/A/noslip/float64','formal shared-edge upwind/complementarity/outward boundary'],
        resources=dict(GPU_workers=1,allocator_fraction=.10,maximum_seconds=180,estimate_minutes=[1,3]),
        stopping='One candidate only. No halving, alternate predictor, budget extension or automatic retry. Accepted result means this attempt would avoid its prior rejection; no annual accuracy or1554H-error claim.',
        telemetry='Save predicted H/face coefficients and signed remainders, every BP candidate H/u and physical correction arrays; complete outer/Newton/CG history on rejection.',
        fixtures=ref(E/'algebraic-fixtures.json'),input_hashes={p.relative_to(PROJECT).as_posix():st.digest(p) for p in inputs},
        code_hashes=hashes(),baseline_replay_authorized=False,phase3_authorized=False,accepted_count_effect=0)
    st.write(E/'plan.frozen.json',plan);fingerprint=seal(plan);st.write(RUN/'contract.json',dict(contract=fingerprint))
    st.write(E/f'fingerprint-{fingerprint}.json',dict(root=str(RUN)))
    transition('authorized',E/'plan.frozen.json',
        'Confirmed local diffusion predictor omitted donorH in g and outward boundary terms; implement new predictor version while preserving archived old code. User-directed single controlled failed-candidate comparison: original1552.0checkpoint,1552annualSMB,dt0.25,omega1,8BP budget and all physical/acceptance equations unchanged. Add g=abs(C)Hdonor/max(abs(ds),1m), outward donor diagonal and frozen signed flux remainder to handle direction/cutoff at old geometry; predicted new flux remains approximate and reversals are reported. Reuse complete v38attempt0001 rejection; no old run replay. One180s-bounded attempt,10percentGPU, no halving/auto-retry/extension. Algebraic100m scaling,uphill/flat/cutoff andoutlet fixtures passed. This new identified predictor mechanism does not reopen prior mass-commit repair or annual accuracy promotion; no mainline/count/default change.')
    print(json.dumps(dict(status='registered',fingerprint=fingerprint)),flush=True)


def execute():
    validate();plan=st.read(E/'plan.frozen.json');assert hashes()==plan['code_hashes']
    assert st.read(PROJECT/'LIA_CURRENT_STATE.json')['research_branches'][NAME]['launch_authorized']
    for p,h in plan['input_hashes'].items():assert st.digest(PROJECT/p)==h,p
    assert not (RUN/'launch.json').exists()
    site=PROJECT/'.runtime/torch-cu128-v1/site-packages';win=Path(os.environ.get('SystemRoot','C:/Windows'))
    env=os.environ.copy();env.pop('KMP_DUPLICATE_LIB_OK',None);env.pop('PYTHONHOME',None)
    env['PATH']=os.pathsep.join(map(str,[site/'torch/lib',win/'System32',win]));env['PYTHONPATH']=os.pathsep.join(map(str,[site,PROJECT]));env['POLIM_TORCH_SITE']=str(site)
    command=['D:/Anaconda/envs/SS4/python.exe','-m','polim_cpu.predictor_repair_worker_v1',str(RUN)];start=time.perf_counter()
    with (RUN/'worker.log').open('x') as log:
        p=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000)
        st.write(RUN/'launch.json',dict(pid=p.pid,command=command,start=datetime.now(timezone.utc).isoformat(),output_root=str(RUN),
            max_seconds=180,estimate_minutes=[1,3],checkpoint='every BP candidate and accepted final state',stop_conditions='one attempt; maximum8BP calls; no halving'))
        try:p.wait(timeout=180)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True);p.wait(timeout=30)
    elapsed=time.perf_counter()-start
    result=st.read(RUN/'completion.json') if (RUN/'completion.json').exists() else dict(status='technical-failure',error=f'worker exited {p.returncode}')
    old=st.read(PRIOR/'attempt-0001.json');checkpoints=sorted(RUN.glob('outer-??'));checks=[]
    z=st.arrays(SOURCE/'input.npz');contract=st.read(RUN/'contract.json')['contract']
    if checkpoints:
        state,meta=st.restore(checkpoints[-1],contract);bp,_=cpu_bp(state['h'],state['u'],z)
        checks.append(dict(candidate=checkpoints[-1].name,original_CPU_BP=bp,h_sha256=st.array_hash(state['h']),u_sha256=st.array_hash(state['u'])))
    diagnostics=st.read(RUN/'predictor-diagnostics.json') if (RUN/'predictor-diagnostics.json').exists() else None
    comparison=dict(status='completed',new_status=result['status'],accepted_without_halving=result.get('accepted_without_halving',False),
        old_status=old['status'],old_bp_calls=8,new_bp_calls=result.get('bp_calls'),old_history=old['history'],new_result=result,
        old_attempt_seconds=old['timing']['total'],new_process_wall_seconds=elapsed,new_original_CPU_checks=checks,
        predictor=diagnostics,predictor_repair_bundle_only=True,old_failure_not_replayed=True,no_annual_accuracy_claim=True,
        launch_authorized=False,phase3_authorized=False)
    st.write(E/'completion.json',comparison)
    final=result.get('final') or {};oldfinal=old['history'][-1]
    lines=['# 扩散预测器修复：1552 年一个既有失败候选步的对照','',
        '原起点1552.0、季度步长0.25年、年度SMB缓存、阻尼1.0、最多8次BP和全部接受门槛不变。旧结果读取既有完整失败记录，未重跑。只替换预测器：补供冰厚度和出口项，并加入冻结有符号通量余项处理方向及分母截断。','',
        '| 指标 | 旧预测器 | 修复预测器 |','|---|---:|---:|',
        f"| 结果 | rejected | {result['status']} |",f"| BP调用 | 8 | {result.get('bp_calls')} |",
        f"| 牛顿更新 | {sum(x['newton'] for x in old['history'])} | {result.get('newton')} |",f"| CG迭代 | {sum(x['cg'] for x in old['history'])} | {result.get('cg')} |"]
    for key in ('bp','mass_relative','thickness_equation_relative','fixed_point_relative'):
        lines.append(f"| 第末次 {key} | {oldfinal[key]:.9g} | {final.get(key)} |")
    lines+=['',f"本次是否在原步长下通过：**{comparison['accepted_without_halving']}**。只执行了一次候选步，未减半、未扩展预算或再试其他版本。",'',
        '100 m供冰厚度的代数检查确认系数为旧值100倍。裸扩散仍只在顺坡且未截断的原几何上匹配通量；冻结余项使预测模型在原几何匹配有符号正式通量，不能把它当作新几何的精确BP通量。出口用原冻结向外速率乘预测冰厚。预测器的障碍乘子不是实际消融账本，最终物理源项仍由原正式校正器确定。','',
        f"方向与截断诊断：`{diagnostics}`。",'',
        f"旧尝试记录耗时 {old['timing']['total']:.3f} s；新尝试记录耗时 {result.get('timing',{}).get('total')} s；新进程完整耗时 {elapsed:.3f} s。诊断落盘增加了观测开销，历史计时不是硬件配对性能证明。",'',
        '这是整个预测器修复组合的受控对照，不能分别归因到供冰厚度、出口或通量余项中的某一项。也不能据此解释1554年2 m厚度门槛失败；消融时序仍为其他位置的待查线索。']
    (E/'summary.zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    transition('completed',E/'completion.json',
        f"Single registered predictor-only1552quarter candidate finished status={result['status']}, BPcalls={result.get('bp_calls')}, accepted-without-halving={comparison['accepted_without_halving']}. Original failed attempt retained without replay; sameH/u/SMB/dt/omega/budget and exactBP/formal transport gates. Predictor direction/cutoff/outlet diagnostics and all candidate states saved; no automatic second trial,annual-accuracy attribution,phase3,default,count or physics change. Authorization closed.")
    print(json.dumps(dict(status='closed',result=result['status'],bp_calls=result.get('bp_calls'),accepted_without_halving=comparison['accepted_without_halving'])),flush=True)


if __name__=='__main__':
    if sys.argv[-1]=='prepare':prepare()
    elif sys.argv[-1]=='execute':execute()
    else:raise SystemExit('prepare | execute')
