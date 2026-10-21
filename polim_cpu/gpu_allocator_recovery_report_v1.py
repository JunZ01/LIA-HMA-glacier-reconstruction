from .gpu_allocator_recovery_trial_v1 import *
def transition(stage, evidence, facts=''):
    validate(); sp = PROJECT/'LIA_CURRENT_STATE.json'; raw = sp.read_bytes(); s = json.loads(raw)
    assert s['current_focus']['state'] != 'decision-required'
    authorized = stage == 'authorized'
    if authorized: assert NAME not in s['research_branches']
    else: assert s['research_branches'][NAME]['status'] in ['completed','failed']
    b = dict(status=stage,launch_authorized=authorized,real_momentum_authorized=authorized,
             real_glacier_evolution_authorized=authorized,full_stokes_authorized=False,
             allowed_stages=['one-technical-fixture','50m-saved-pending1111-recovery-to100','conditional-working-solver-promotion'] if authorized else [],
             plan=ref(E/'plan.frozen.json'),evidence=ref(evidence),accepted_count_effect=0,
             production_replacement_authorized=False,working_BP_solver_selected=(stage=='working-selected'),working_settings=ref(E/'working-settings.json') if stage=='working-selected' else None,next_legal_action='Resume50m pending1111 afterchunkedmemoryrepair; retain100mpass' if authorized else 'Use measured comparison; no automatic extension')
    fp = PROJECT/'LIA_PROJECT_FACTS.md'; fb = fp.read_bytes(); fh = digest(fp).upper()
    assert fh == s['scientific_contract']['sha256'].upper()
    if facts:
        number = max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1
        addition = f'\n\n## {number}. System2387 GPU handover memory recovery\n\n{facts}\n'
        fh = hashlib.sha256(fb+addition.encode('utf-8')).hexdigest().upper()
    route = read(PROJECT/s['authoritative_route']['path']); suffix = f'+polim-v31r3-{stage}'
    route['version'] += suffix; route[NAME] = b
    if 'scientific_contract' in route: route['scientific_contract'] = {**s['scientific_contract'],'sha256':fh}
    rp = E/f'route-{stage}.json'; write_json(rp,route)
    n = copy.deepcopy(s); n['research_branches'][NAME] = b
    n['authoritative_route'] = {**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']}
    n['state_id'] += suffix; n['updated_at'] = datetime.now(timezone.utc).isoformat()
    n['scientific_contract']['sha256'] = fh
    if sp.read_bytes() != raw or fp.read_bytes() != fb: raise RuntimeError('Concurrent state update')
    temp = sp.with_name(f'.LIA_CURRENT_STATE.acceleration.{os.getpid()}.tmp'); write_json(temp,n)
    if facts:
        with fp.open('ab') as f: f.write(addition.encode('utf-8')); f.flush(); os.fsync(f.fileno())
    os.replace(temp,sp)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:
        f.write(f'\n\n## {n["updated_at"]} - BP acceleration {stage}\n\nUser agreed to50/100m longhandover withaccuracy andrestart checks,then selectas workingBPsolver onlyifbothpass. NoOGGM191mainline/countchange. Same30day,float64,finalBP1e-6;annualAMGrefresh fromcurrentstate. `{ref(evidence)["path"]}`.\n')
    write_json(E/f'{stage}-validation.json',validate())

def main():
    validate();out=read(E/'reverification.json');passed=out['status']=='completed' and all(c['passed'] for c in out['cases'].values());REPORT.mkdir(exist_ok=False)
    fixture=read(RUN/'grid-50m/technical-fixture.json');summaries={}
    for dx,c in out['cases'].items():
        if not c['passed']:
            f=read(RUN/'grid-50m/result-recovery.json');summaries[dx]=dict(passed=False,error=f.get('error'),last_accepted_step=f.get('accepted_steps'),pending_year=f.get('year'));continue
        phases=c['phases'];rows=[r for p in phases for r in p['rows']];checks=c['checks']
        summaries[dx]=dict(passed=True,completed_year=100,accepted_steps=phases[-1]['accepted_steps'],wall_seconds=c['wall_seconds'],checkpoint_seconds=c['checkpoint_seconds'],CPU_historical_seconds=c['historical_CPU_seconds'],recorded_hierarchy_seconds=c['hierarchy_seconds'],
            H_max_m=max(x['H_max_m'] for x in checks),surface_velocity_relative_L2=max(x['velocity_relative_L2'] for x in checks),original_BP_residual=max(x['BP_residual'] for x in checks),mass_relative=max(x['mass_relative'] for x in rows),
            Newton=sum(p['Newton'] for p in phases),CG=sum(p['CG'] for p in phases),recorded_peak_MiB=max(p.get('peak_memory_bytes') or 0 for p in phases)/2**20,
            annual_states_verified=len(checks),checkpoints_verified=c['verified_checkpoints'],restart_and_status_tests_passed=True,phases=[{k:v for k,v in p.items() if k not in ['rows','traceback','error']} for p in phases])
    summary=dict(status='passed' if passed else 'not-selected',grids=summaries,working_selected=passed,technical_fixture=fixture,additional_recovery_authorized=True,source=ref(E/'reverification.json'),no_completed_dynamics_replayed=True)
    write_json(REPORT/'summary.json',summary)
    if passed:
        pointer=R/'current-working-settings.json';previous_raw=pointer.read_bytes();previous=json.loads(previous_raw);write_json(E/'previous-working-pointer.json',previous)
        settings=read(PROJECT/previous['settings']['path'])
        settings.update(schema='polim-working-settings-gpu-v31r3',momentum_solver='CUDA cached exact CSR BP + Graph PCG + reusable geometry/diagonal buffers and inactive cache release',
            solver_module='polim_cpu.gpu_allocator_safe_v1',hierarchy_module='polim_cpu.gpu_long_hierarchy_v1',storage_module='polim_cpu.gpu_long_storage_v1',solver_implementation=hashes(),
            solver_precision='float64',momentum_tolerance=1e-6,inner_linear_target=.03,CG_graph_batch=4,tangent_element_chunk=2048,geometry_buffer_reused=True,diagonal_buffer_reused=True,inactive_cache_release=True,
            runtime=dict(cuda_python='D:/Anaconda/envs/SS4/python.exe',torch_site=str(PROJECT/'.runtime/torch-cu128-v1/site-packages'),initializer='polim_cpu.v3.gpu.initialize',isolated_process=True,allocator_fraction=.10),
            preconditioner_update='Annual current-state AMG rebuild on CPU; all BP iteration and transport on GPU.',CPU_reference_solver='polim_cpu.bp_accelerated_v1',automatic_CPU_fallback=False,
            solver_validation=ref(REPORT/'summary.json'),long_term_scope='DefaultworkingBPsolver after50/100m year1-100 handover andoriginalCPUannualchecks. Fixedbed/A/zeroSMB validation;climate/thermalcoupling continuesasfuturework.',
            supersedes=previous['settings'],launch_authorized=False,user_decision='Selectaftercombinedlonghandoverpasses;additionalyear39recoveryexplicitlyauthorized允许.')
        write_json(E/'working-settings.json',settings)
        new_pointer={**previous,'schema':'polim-working-settings-pointer-v2','momentum_solver':'gpu-reusable-graph-v31r3','settings':ref(E/'working-settings.json')}
    rows='';accuracy=''
    for dx,c in summaries.items():
        if not c['passed']:
            rows+=f'<tr><td>{dx}米</td><td colspan="4">未完成，默认不切换</td></tr>';continue
        rows+=f"<tr><td>{dx}米</td><td>{c['wall_seconds']/60:.2f}分钟</td><td>{c['CPU_historical_seconds']/60:.2f}分钟</td><td>{c['checkpoint_seconds']/60:.2f}分钟</td><td>{c['accepted_steps']}</td></tr>"
        accuracy+=f"<tr><td>{dx}米</td><td>{c['H_max_m']:.3e}米</td><td>{c['surface_velocity_relative_L2']:.3e}</td><td>{c['original_BP_residual']:.3e}</td><td>{c['mass_relative']:.3e}</td></tr>"
    title='GPU长期交接通过：已设为后续工作求解器' if passed else '本次续跑未通过，保留原工作设置'
    lead='50米和100米均已完成至第100年，并通过各自100个年度状态的原CPU精度核对。后续默认使用GPU求解器，保持50米网格、30天上限。' if passed else '本次额外授权的续跑未通过，已保存最新状态；未切换默认设置。'
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GPU工作版本交接</title>
<style>body{{margin:0;background:#edf3f5;color:#243d4a;font:17px "Microsoft YaHei",sans-serif}}main{{max-width:1160px;margin:auto;padding:30px}}p,li{{line-height:1.85}}.lead{{background:#e1f2ee;border-left:5px solid #228f85;padding:20px}}table{{width:100%;background:white;border-collapse:collapse}}th,td{{padding:12px;border-bottom:1px solid #dbe5e9;text-align:left}}.note{{font-size:14px;color:#526976}}a{{color:#147f90}}</style>
<main><h1>{title}</h1><p class="lead">{lead}</p>
<table><tr><th>网格</th><th>GPU总推进耗时</th><th>原CPU历史耗时</th><th>GPU保存耗时</th><th>接受步数</th></tr>{rows}</table>
<p>从保存的第1年状态出发，各新推进99年到第100年。50米总时间包含所有已完成分段及中途三次显存分配失败的耗时；未重算已完成时间步。两组计时均包含逐步保存、载入、GPU求解、输运、年度辅助矩阵构建和状态发布，未计外部解释器启动、初始CPU数据导出和完成后的独立核对。旧CPU的求解器与保存策略不同，历史时间只作实际工作成本参考。</p>
<h2>最终精度</h2><table><tr><th>网格</th><th>最大冰厚差</th><th>表面速度相对L2差</th><th>原CPU BP最大残差</th><th>质量相对误差</th></tr>{accuracy}</table>
<p>预定门槛：冰厚差≤0.001米、速度相对差≤0.001、BP残差≤10⁻⁶、质量相对误差≤10⁻¹⁰。每种网格分别对照同网格CPU结果，不混入50米与100米本身的分辨率差异。{'两组共200个年度状态核对通过。' if passed else '100米核对已通过，50米尚未完成交接。'}</p>
<h2>显存修复与恢复过程</h2><p>50米第16年首次停止时，大型刚度临时数组超过设定分配预算；改成每批2048个单元的精确组装后，从保存状态继续。第39年又发现几何基函数数组每步重复申请约198MiB；按用户追加授权，改为始终复用同一块固定大小的数组。随后推进到第86年，在几何对角项的临时数组申请处再次触及预算；最终把对角项也按2048单元分块计算并复用输出数组，同时在每步和线性化开始前释放闲置缓存。活跃的CUDA Graph数组保留。</p>
<p>内存修复仅调整数组生命周期和运算分块。释放闲置缓存的依据见<a href="https://docs.pytorch.org/docs/main/generated/torch.cuda.memory.empty_cache.html">PyTorch官方说明</a>；它可减少碎片，不增加显存预算。</p><p>新内存实现的两种网格算子对照均通过，矩阵向量积相对差约2×10⁻¹⁵，并在每一步检查几何数组地址保持不变。物理公式、双精度、收敛精度和GPU分配预算均不变。恢复时直接采用已完成输运的第1111步，不重算前1110步，也不重跑100米轨迹。100米长期证据来自保留的原GPU运行，内存修复另做了该网格的算子等价核对。</p>
<p>两组第2年均正常退出并由新GPU进程读取状态接续。状态发布失败注入测试也已通过：写入备用快照后继续推进。每个待求解与已接受状态都原子保存，牛顿迭代每5次保存。三个失败阶段未完整保留显存峰值和辅助构建分项时间，摘要中相应数值仅代表已记录部分；总推进和保存时间均保留。</p>
<h2>以后使用的工作版本</h2><p>{'默认动力求解器已切换为GPU缓存精确刚度矩阵＋CUDA Graph，配合分块组装、几何及对角数组复用、闲置缓存释放和每年一次的当前状态辅助矩阵更新。CPU AMG保留为对照；没有自动CPU回退。' if passed else '当前默认设置保持不变。'} 50米网格和30天上限保持不变，CFL仍可缩短时间步。</p>
<p>本次长期验证为固定冰床、固定材料参数、零表面质量平衡。后续气候驱动和热力耦合继续在这套求解器上完善；本次没有新增相关模拟，也没有修改独立OGGM批量主线或已接受数量。</p>
<p><a href="summary.json">完整摘要</a> · <a href="../gpu-long-handover-v1/index-v2.html">此前交接与故障记录</a> · <a href="../../index.html">项目入口</a></p></main></html>'''
    (REPORT/'index.html').write_text(page,encoding='utf-8')
    evidence=dict(status=summary['status'],summary=ref(REPORT/'summary.json'),report=ref(REPORT/'index.html'),working_selected=passed)
    if passed:evidence['working_settings']=ref(E/'working-settings.json')
    write_json(E/'review.json',evidence)
    facts=f"- Completedsimulationandcorrectedread-onlyverification: priorcompletionfailedonlybecausepending-adoption.jsonmatchedthecheckpointglob;no simulationrepeated. Allocatorrecovery {'passed; selectedasworkingBPsolver' if passed else 'failed; defaultretained'}. Statistics {json.dumps(summaries,separators=(',',':'))}. BPphysics/float64/final1e-6/50m/30daysunchanged;annualAMGcurrent-stateCPUsetup,GPUcachedCSR/Graph,2048elementtangentchunks,geometryB/diagonalreused,inactivecacheclearedoutsideGraphcapture. Bothgridnewoperatorfixturepassed;retained100mtrajectory. Restore50maccepted1110andpending1111,noprefixreplay. CPUAMGreference,noautomaticfallback. Fixedbed/A/zeroSMB validation only;noOGGM191countchangeornewclimate/thermalrun. Evidence `{ref(E/'review.json')['path']}`."
    transition('working-selected' if passed else 'reviewed',E/'review.json',facts)
    if passed:
        assert pointer.read_bytes()==previous_raw
        temp=pointer.with_name(f'.working-{os.getpid()}.json');write_json(temp,new_pointer);os.replace(temp,pointer)
        write_json(E/'working-pointer-delivery.json',dict(pointer=ref(pointer),previous=ref(E/'previous-working-pointer.json')));write_json(E/'selection-validation.json',validate())
    p=R/'index.html';text=p.read_text(encoding='utf-8');tag='<h1>2387 冰川模型</h1>';text=text.replace(tag,tag+f'<a class="current" href="reports/gpu-working-v31r3/index.html"><b>{title}</b><p>50米与100米长期交接、精度核对及断点恢复。</p></a>',1);p.write_text(text,encoding='utf-8')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()

