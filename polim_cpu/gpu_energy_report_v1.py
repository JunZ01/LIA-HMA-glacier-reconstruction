from .gpu_energy_trial_v1 import *
def transition(stage, evidence, facts=''):
    validate(); sp = PROJECT/'LIA_CURRENT_STATE.json'; raw = sp.read_bytes(); s = json.loads(raw)
    assert s['current_focus']['state'] != 'decision-required'
    authorized = stage == 'authorized'
    if authorized: assert NAME not in s['research_branches']
    else: assert s['research_branches'][NAME]['status'] == 'completed'
    b = dict(status=stage,launch_authorized=authorized,real_momentum_authorized=authorized,
             real_glacier_evolution_authorized=authorized,full_stokes_authorized=False,
             allowed_stages=['combined-preflight','GPU-direct-and-neural-year99-100'] if authorized else [],
             plan=ref(E/'plan.frozen.json'),evidence=ref(evidence),accepted_count_effect=0,
             production_replacement_authorized=False,next_legal_action='Run bounded100m CUDA direct and neural pilot' if authorized else 'Retain GPU research result; no automatic extension')
    fp = PROJECT/'LIA_PROJECT_FACTS.md'; fb = fp.read_bytes(); fh = digest(fp).upper()
    assert fh == s['scientific_contract']['sha256'].upper()
    if facts:
        number = max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1
        addition = f'\n\n## {number}. System2387 CUDA BP energy test\n\n{facts}\n'
        fh = hashlib.sha256(fb+addition.encode('utf-8')).hexdigest().upper()
    route = read(PROJECT/s['authoritative_route']['path']); suffix = f'+polim-v29-{stage}'
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
        f.write(f'\n\n## {n["updated_at"]} - BP acceleration {stage}\n\nUser authorized GPU parallel BP energy and neural comparison. Same100m99-100,30daycap,float64,BP1e-6;previous offline network reused,4online updates;GPU Newton-PCG correction. CPU hierarchy prepared once; no production replacement. `{ref(evidence)["path"]}`.\n')
    write_json(E/f'{stage}-validation.json',validate())

def main():
    validate();result=read(E/'completion.json');gpu=result['gpu'];checks=result['verification'];report=R/'reports/gpu-energy-v1';report.mkdir(exist_ok=False)
    base=result['historical_CPU_seconds'];direct=gpu['cases']['direct'];neural=gpu['cases']['neural']
    setup=result['cpu_setup']['seconds']+gpu['setup_seconds'];preflight=read(RUN/'preflight.json')
    summary=dict(result=result,preflight=preflight,setup_seconds=setup,CPU_historical=True,
        GPU_direct_elapsed_reduction_percent=100*(1-direct['wall_seconds']/base),
        GPU_neural_overhead_percent=100*(neural['wall_seconds']/direct['wall_seconds']-1),
        peak_MiB=gpu['peak_memory_bytes']/2**20,
        runtime_includes_checkpoints=True,CPU_GPU_checkpoint_cadences_differ=True,
        no_automatic_promotion=True)
    write_json(report/'summary.json',summary)
    rows=''
    for label,v in [('GPU直接BP',direct),('GPU神经网络＋BP校正',neural)]:
        selected=sum(r['prediction'].get('chosen','previous')!='previous' for r in v['rows'])
        rows+=f"<tr><td>{label}</td><td>{v['wall_seconds']:.2f}秒</td><td>{v['compute_seconds']:.2f}秒</td><td>{v['checkpoint_seconds']:.2f}秒</td><td>{sum(r['momentum']['iterations'] for r in v['rows'])}</td><td>{selected}/13</td></tr>"
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GPU BP与神经网络实测</title>
<style>body{{margin:0;background:#edf3f5;color:#243d4a;font:17px "Microsoft YaHei",sans-serif}}main{{max-width:1120px;margin:auto;padding:30px}}p,li{{line-height:1.85}}.lead{{background:#e1f2ee;border-left:5px solid #228f85;padding:20px}}table{{width:100%;background:white;border-collapse:collapse}}th,td{{padding:12px;border-bottom:1px solid #dbe5e9;text-align:left}}.note{{font-size:14px;color:#526976}}a{{color:#147f90}}</style>
<main><h1>GPU求解已跑通，当前小网络尚未减少计算量</h1>
<p class="lead">100米网格、第99—100年，两组各13步均完成，并通过原CPU BP算子的精度核对。GPU直接求解7.60秒；加入GPU神经网络后9.65秒。网络被采用7次，但两组仍各需25次牛顿迭代。</p>
<table><tr><th>方案</th><th>完整续算</th><th>扣除存盘</th><th>存盘</th><th>牛顿更新</th><th>采用网络</th></tr>{rows}<tr><td>CPU AMG历史基准</td><td>{base:.2f}秒</td><td colspan="2">历史保存频率不同，未重跑</td><td>25</td><td>—</td></tr></table>
<h2>这次实际改了什么</h2><p>BP能量、力学残差、牛顿与共轭梯度迭代、网络前向与反向传播、冰厚输运均在GPU上执行，几何和状态保留在显存。CPU仅在开始时准备一次多重网格层级，随后传入GPU复用；每次迭代直接计算刚度矩阵与向量的乘积，省去反复组装完整矩阵。显存峰值{summary['peak_MiB']:.1f} MiB（PyTorch分配，不含驱动和其他任务）。</p>
<p>GPU网络复用此前只用早期状态训练的1124参数模型，每步在线最多4次L-BFGS更新，损失仍为同一BP能量。输出按局部刚度修正尺度缩放，底部速度为零。预测通过残差与能量筛选后，再由GPU BP校正至1e−6相对残差。Torch在线优化器与前版SciPy实现有差异，本次不作为只改变硬件的消融试验。</p>
<h2>结果是否一致</h2><table><tr><th>与CPU参考的最大差异</th><th>直接GPU</th><th>GPU网络</th></tr>
<tr><td>冰厚</td><td>{max(c['H_max_m'] for c in checks['direct']['checks']):.3e} 米</td><td>{max(c['H_max_m'] for c in checks['neural']['checks']):.3e} 米</td></tr>
<tr><td>表面速度相对L2差</td><td>{max(c['velocity_relative_L2'] for c in checks['direct']['checks']):.3e}</td><td>{max(c['velocity_relative_L2'] for c in checks['neural']['checks']):.3e}</td></tr>
<tr><td>原CPU算子核对的最大BP残差</td><td>{max(c['residual'] for c in checks['direct']['checks']):.3e}</td><td>{max(c['residual'] for c in checks['neural']['checks']):.3e}</td></tr></table>
<p>两组共28个完整状态已校验。冰床、流变参数、30天上限、CFL 0.35和零表面质量平衡保持一致，使用双精度。首步输运冰厚与保留参考逐元素相同；CPU/GPU能量、力、梯度、刚度对角项、矩阵向量积和网络反向传播通过对照，GPU保存恢复逐元素相等。</p>
<h2>如何理解加速</h2><p>本次GPU直接续算耗时比历史CPU低{summary['GPU_direct_elapsed_reduction_percent']:.1f}%。其中也包含层级复用与保存频率的改变，不能全部归因于GPU硬件。GPU网络比同轮直接GPU慢{summary['GPU_neural_overhead_percent']:.1f}%，说明这个小网络目前仍是额外工作：预测更常被采用，却没有省去BP牛顿更新。</p>
<p>CPU准备{result['cpu_setup']['seconds']:.2f}秒，GPU载入与初始化{gpu['setup_seconds']:.2f}秒，新算子合并检查{preflight['seconds']:.2f}秒，均未计入上表续算时间。准备＋直接续算为{setup+direct['wall_seconds']:.2f}秒；网络复用了既有离线训练成果，其历史训练和解释器启动也未计入本次续算。两组在同一进程依次运行，其他GPU任务仍可能影响计时。</p>
<p>下一项最有价值的改进是减少线性迭代和小GPU运算的调度开销；网络若要获得明显收益，需要让预测直接达到所需精度，或明显减少BP校正次数。本次结果支持保留GPU求解分支继续优化，默认工作模型和百年结果不变。</p>
<p class="note">这是固定材料、零SMB的一年性能窗口；不直接外推有气候驱动的千年耗时。每个待求解和已接受时间步原子保存，牛顿每5次保存；旧CPU版本每次牛顿保存。无可变状态文件依赖。运行完成后未追加长试验。</p>
<p><a href="summary.json">完整实测数据</a> · <a href="../physics-neural-local-scale-v1/index.html">前版输出缩放试验</a> · <a href="../../index.html">项目入口</a></p></main></html>'''
    (report/'index.html').write_text(page,encoding='utf-8')
    index=R/'index.html';text=index.read_text(encoding='utf-8');tag='<h1>2387 冰川模型</h1>'
    text=text.replace(tag,tag+'<a class="current" href="reports/gpu-energy-v1/index.html"><b>新增：GPU BP与神经网络实测</b><p>两组一年完成：GPU直接7.60秒，GPU网络9.65秒；精度通过。</p></a>',1);index.write_text(text,encoding='utf-8')
    evidence=dict(status='reviewed',report=ref(report/'index.html'),summary=ref(report/'summary.json'),source=ref(E/'completion.json'),accuracy_passed=True,default_unchanged=True)
    write_json(E/'review.json',evidence)
    facts=f"- Both CUDA100m99-100 cases completed13steps/14states each. OriginalCPU operator and28state verification passed. DirectGPU wall7.600420600s,compute6.054142700s,checkpoint1.546277900s;neuralGPU9.648843500s,compute8.007539400s,checkpoint1.641304100s. Both25Newton;networkselected7/13,none avoidedcorrection;CG1070/1078. HistoricalCPU9.481634300s;different checkpointcadence/hierarchy,not hardware-onlycausalcomparison. CPUsetup0.550388600s,GPUsetup0.671954000s,preflight1.318726700s excludedfromcontinuation;previousofflineweightsreused,no trainingcharge. GPUallocatedpeak203508736bytes. MaxHdiff4.06644e-11/2.09416e-10m,velocityL29.17069e-9/1.95099e-8,originalBP4.19002e-8/5.56834e-8. Fixedbed/A,zeroSMB,float64,BP1e-6 unchanged. GPUbranchfeasible;currentnetworknotfaster. DefaultAMG/countsunchanged,noautomaticextension. Evidence `{ref(E/'review.json')['path']}`."
    transition('reviewed',E/'review.json',facts)
    print(str(report/'index.html'))

if __name__=='__main__':main()


