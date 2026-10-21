from .gpu_cached_trial_v1 import *
def transition(stage, evidence, facts=''):
    validate(); sp = PROJECT/'LIA_CURRENT_STATE.json'; raw = sp.read_bytes(); s = json.loads(raw)
    assert s['current_focus']['state'] != 'decision-required'
    authorized = stage == 'authorized'
    if authorized: assert NAME not in s['research_branches']
    else: assert s['research_branches'][NAME]['status'] == 'completed'
    b = dict(status=stage,launch_authorized=authorized,real_momentum_authorized=authorized,
             real_glacier_evolution_authorized=authorized,full_stokes_authorized=False,
             allowed_stages=['combined-preflight','GPU-cached-and-graph-year99-100'] if authorized else [],
             plan=ref(E/'plan.frozen.json'),evidence=ref(evidence),accepted_count_effect=0,
             production_replacement_authorized=False,next_legal_action='Run bounded100m CUDA cached and graph pilot' if authorized else 'Use measured comparison; no automatic extension')
    fp = PROJECT/'LIA_PROJECT_FACTS.md'; fb = fp.read_bytes(); fh = digest(fp).upper()
    assert fh == s['scientific_contract']['sha256'].upper()
    if facts:
        number = max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1
        addition = f'\n\n## {number}. System2387 CUDA tangent cache and Graph test\n\n{facts}\n'
        fh = hashlib.sha256(fb+addition.encode('utf-8')).hexdigest().upper()
    route = read(PROJECT/s['authoritative_route']['path']); suffix = f'+polim-v30-{stage}'
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
        f.write(f'\n\n## {n["updated_at"]} - BP acceleration {stage}\n\nUser authorized cachedGPUtangent and CUDA Graph comparison. Same100m99-100,30daycap,float64,finalBP1e-6;innerlinear.03,4iterationGraphbatches;no productionreplacement. `{ref(evidence)["path"]}`.\n')
    write_json(E/f'{stage}-validation.json',validate())

def main():
    validate();result=read(E/'completion.json');assert result['status']=='completed'
    old=read(R/'runs/v29-gpu-energy-v1/gpu-result.json')['cases']['direct'];gpu=result['gpu'];cases=gpu['cases'];checks=result['verification'];best=cases['graph']
    report=R/'reports/gpu-cached-v1';report.mkdir(exist_ok=False)
    cg=lambda c:sum(i.get('cg_iterations',0) for r in c['rows'] for i in r['momentum']['history'])
    newton=lambda c:sum(r['momentum']['iterations'] for r in c['rows'])
    summary=dict(result=result,prior_gpu=old,new_kernels=read(RUN/'new-kernels.json'),preflight=read(RUN/'preflight.json'),
        speedup_vs_prior=old['wall_seconds']/best['wall_seconds'],compute_speedup=old['compute_seconds']/best['compute_seconds'],
        CG_old=cg(old),CG_new=cg(best),same_checkpoint_cadence=True,default_unchanged=True)
    write_json(report/'summary.json',summary)
    rows=''
    for label,c in [('上一版GPU（历史）',old),('缓存刚度矩阵＋提前结束内层迭代',cases['cached']),('再加CUDA Graph',best)]:
        rows+=f"<tr><td>{label}</td><td>{c['wall_seconds']:.2f}秒</td><td>{c['compute_seconds']:.2f}秒</td><td>{c['checkpoint_seconds']:.2f}秒</td><td>{cg(c)}</td><td>{newton(c)}</td></tr>"
    hmax=max(c['H_max_m'] for c in checks['graph']['checks']);umax=max(c['velocity_relative_L2'] for c in checks['graph']['checks']);resmax=max(c['residual'] for c in checks['graph']['checks'])
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GPU求解优化实测</title>
<style>body{{margin:0;background:#edf3f5;color:#243d4a;font:17px "Microsoft YaHei",sans-serif}}main{{max-width:1150px;margin:auto;padding:30px}}p,li{{line-height:1.85}}.lead{{background:#e1f2ee;border-left:5px solid #228f85;padding:20px}}table{{width:100%;background:white;border-collapse:collapse}}th,td{{padding:12px;border-bottom:1px solid #dbe5e9;text-align:left}}.note{{font-size:14px;color:#526976}}a{{color:#147f90}}</style>
<main><h1>GPU一年续算：7.60秒 → 2.61秒</h1><p class="lead">100米网格，第99—100年，两组优化方案均完成13步并通过精度核对。最快方案含保存耗时2.61秒，其中计算0.96秒、保存1.66秒。相比上一版GPU完整续算快{summary['speedup_vs_prior']:.2f}倍；扣除保存后的计算部分快{summary['compute_speedup']:.2f}倍。</p>
<table><tr><th>方案</th><th>完整续算</th><th>扣除保存</th><th>保存</th><th>线性迭代</th><th>牛顿更新</th></tr>{rows}</table>
<h2>加速来自哪些改变</h2><ol><li><b>缓存当前刚度矩阵。</b>此前每次线性迭代都重复执行积分点上的矩阵向量计算；现在每次牛顿更新只在GPU上组装一次当前精确切线矩阵，之后用稀疏矩阵乘法复用。每次牛顿更新仍重新计算矩阵。</li>
<li><b>避免把中间线性解算得过细。</b>内层线性相对残差目标改为0.03，牛顿迭代继续校正到最终BP相对残差不超过10⁻⁶。最终要求没有放宽。CUDA Graph组线性迭代由1070次降到520次。</li>
<li><b>批量提交GPU运算。</b>CUDA Graph记录四次共轭梯度迭代的运算序列，随后重复提交；每四次检查是否达到内层目标，减少Python调用及CPU/GPU同步。小规模最粗层使用预先计算的逆矩阵乘法，避免反复三角求解。</li></ol>
<p>缓存方案与Graph方案的停止检查频率不同，因此线性迭代为491与520次、牛顿更新为26与25次。Graph组虽多做少量线性迭代，计算部分仍更快。上一版与本版使用相同的保存频率；保存耗时的波动单列，未算作计算加速。</p>
<h2>精度核对</h2><table><tr><th>最快方案与原CPU参考的差异</th><th>实测</th><th>预定要求</th></tr><tr><td>最大冰厚差</td><td>{hmax:.3e}米</td><td>≤0.001米</td></tr><tr><td>表面速度相对L2差</td><td>{umax:.3e}</td><td>≤0.001</td></tr><tr><td>最大原BP相对残差</td><td>{resmax:.3e}</td><td>≤10⁻⁶</td></tr><tr><td>最大质量相对误差</td><td>{max(r['mass_relative'] for r in best['rows']):.3e}</td><td>≤10⁻¹⁰</td></tr></table>
<p>两组共28个状态通过原CPU BP算子核对。GPU缓存矩阵与原CPU矩阵向量积相对差约1.43×10⁻¹⁵；固定八次线性迭代的普通执行和Graph执行结果一致；保存恢复检查通过。相同冰床、材料、双精度、零表面质量平衡、30天上限和CFL0.35。</p>
<h2>准备成本与目前的主要耗时</h2><p>本轮CPU建立索引和导出{result['cpu_setup']['seconds']:.3f}秒，GPU载入{gpu['setup_seconds']:.3f}秒，Graph记录{gpu['graph_capture_seconds']:.3f}秒；这些一次性成本未计入续算。多重网格层级复用已保存数据，上一版准备成本0.550秒也应计入从零开始使用的预算。新算子合并检查{summary['preflight']['seconds']:.3f}秒。上述合计仍不包括解释器启动及完成后的CPU核对。</p>
<p>PyTorch分配显存峰值{gpu['peak_memory_bytes']/2**20:.1f}MiB，不含驱动和其他任务。当前保存约占最快方案总时间的{best['checkpoint_seconds']/best['wall_seconds']*100:.1f}%。更长的正式运行可研究减少保存开销，同时保留可恢复性；这次没有削减保存来制造提速。</p>
<p>保留缓存矩阵＋CUDA Graph作为下一阶段GPU候选。此次属于一年固定材料、零SMB窗口，尚未替代默认CPU模型；50米、长期气候驱动的表现需在对应任务中确认。未继续增加网络训练预算。</p>
<p class="note">历史耗时与当前GPU负载可能不同，不能将速度比理解为所有冰川和所有年份的固定倍数。本次比较的是算法与调度的合并优化。CUDA Graph实现依据<a href="https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/">PyTorch官方说明</a>：保持输入输出内存地址固定、预热后记录、重复执行相同运算序列。</p>
<p><a href="summary.json">完整实测数据</a> · <a href="../gpu-energy-v1/index.html">上一版GPU对照</a> · <a href="../../index.html">项目入口</a></p></main></html>'''
    (report/'index.html').write_text(page,encoding='utf-8')
    p=R/'index.html';s=p.read_text(encoding='utf-8');tag='<h1>2387 冰川模型</h1>';s=s.replace(tag,tag+'<a class="current" href="reports/gpu-cached-v1/index.html"><b>新增：GPU求解优化</b><p>一年7.60秒降至2.61秒；线性迭代1070降至520，精度通过。</p></a>',1);p.write_text(s,encoding='utf-8')
    evidence=dict(status='reviewed',source=ref(E/'completion.json'),report=ref(report/'index.html'),summary=ref(report/'summary.json'),passed=True,default_unchanged=True)
    write_json(E/'review.json',evidence)
    facts=f"- Two100m99-100CUDA cases completed13steps each,28states verifiedwith originalCPU BP. Exactcurrenttangent cachedCSR eachNewton;innerlineartarget.03,finalBP1e-6 unchanged. SameAMGhierarchy,coarseinverse matvec;Graph4CGiterations/replay. Cached eager wall3.359123700s,compute1.486859900s,IO1.872263800s,CG491,Newton26. Graph wall2.613628800s,compute0.956901900s,IO1.656726900s,CG520,Newton25. PriorGPU7.600420600s,compute6.054142700s,CG1070;samecheckpointcadence,hardwareloadnotpaired. GraphmaxHdiff8.23820e-9m,velocityL24.41367e-8,originalBP6.21495e-7,mass1.47942e-16;gatespassed. CPUexport0.103614700s,GPUsetup0.314659700s,Graphcapture0.049222800s,preflight0.365680100s excluded;priorhierarchyreused(originalsetup0.550388600s). GPUpeak292114432bytes. Fixed8CGgraph/eager identical;cachedHVP1.42884e-15relative. Samephysics/drytest/float64/noSMB,30daycap. GPUcandidate retained,default/countsunchanged,no automaticlongextension. Evidence `{ref(E/'review.json')['path']}`."
    transition('reviewed',E/'review.json',facts);print(str(report/'index.html'))

if __name__=='__main__':main()


