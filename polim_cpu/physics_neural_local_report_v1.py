"""Controlled output-scaling ablation and matched-budget continuation report."""
from . import physics_neural_local_trial_v1 as t


def report():
    t.validate();s=t.read(t.RUN/'result.json');c=s['continuation'];tr=s['training'];a=s['ablation'];old=t.read(t.R/'runs/v27-physics-neural-v1/result.json')
    initial_new=t.arrays(t.RUN/'training/initial-network.npz');initial_old=t.arrays(t.R/'runs/v27-physics-neural-v1/training/initial-network.npz')
    assert all(t.np.array_equal(initial_new[k],initial_old[k]) for k in ['theta','mean','std'])
    root=t.REPORT;root.mkdir(exist_ok=False)
    q=dict(status='completed',decision='Keep local-scale research implementation; retain v23 AMG-CG as default. No automatic further training or century run.',
        only_model_change='Output scale becomes local RMS(-r/diagK),removing globalfloor; inputfeatures,initialweights,normalization,architecture,physicaloperator and40offline/4online budgets unchanged.',
        same_initial_weights_and_feature_normalization=True,ablation=a,
        fixed_weight_residual_reduction=a['records']['legacy']['relative_residual']/a['records']['local']['relative_residual'],
        parameters=tr['parameters'],training_seconds=tr['total_seconds'],optimizer_seconds=tr['optimizer']['seconds'],
        continuation_seconds=c['wall_seconds'],first_use_seconds=s['first_use_seconds'],total_process_seconds=s['total_seconds'],
        historical_AMG_seconds=c['historical_baseline_seconds'],previous_network_seconds=old['continuation']['wall_seconds'],
        previous_selected=old['continuation']['neural_selected'],neural_selected=c['neural_selected'],steps=c['steps'],
        new_Newton=c['new_Newton'],baseline_Newton=c['baseline_Newton'],BP_skipped=c['BP_skipped'],verified_states=c['verified_states'],
        maximum_H_difference_m=max(x['H_max_m'] for x in c['comparison']),maximum_velocity_relative_L2=max(x['velocity_relative_L2'] for x in c['comparison']),
        maximum_original_BP_residual=max(x['original_BP_residual'] for x in c['comparison']),maximum_mass_relative=max(x['mass_relative'] for x in c['rows']),
        online_seconds=sum(r['online']['seconds'] for r in c['rows']),engineering_equivalence=c['engineering_equivalence'],acceleration_gate=c['acceleration_gate'],
        timing='HistoricalAMG/v27 walltimes not paired. v28 reuses hash-bound unchanged operator/storage preflight while testing newscalegradient/reload. First-use includespreparation/training/newpreflight+continuation; fixed-weightablation andpostcheck shown in totalprocess.',
        outputs={p.relative_to(t.RUN).as_posix():t.digest(p) for p in t.RUN.rglob('*') if p.is_file()})
    t.write_json(root/'summary.json',q)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    plt.rcParams['font.family']=FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name();plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    values=[a['records']['legacy']['relative_residual'],a['records']['local']['relative_residual']]
    bars=axs[0].bar(['旧全场下限','新局部缩放'],values,color=['#bf8655','#228f85']);axs[0].set_yscale('log');axs[0].bar_label(bars,fmt='%.3g',padding=4)
    axs[0].axhline(c['rows'][0]['old_initial_residual'],color='#647a91',ls='--',label='沿用上一步速度');axs[0].legend()
    axs[0].set(ylabel='BP相对残差（对数）',title='固定权重，只改缩放：残差降低1056倍',ylim=(1e-4,1.))
    values=[q['historical_AMG_seconds'],q['previous_network_seconds'],q['continuation_seconds']]
    bars=axs[1].bar(['原AMG','旧缩放网络','新缩放网络'],values,color=['#647a91','#bf8655','#228f85']);axs[1].bar_label(bars,fmt='%.2f秒',padding=4)
    axs[1].set(ylabel='一年推进耗时（秒）',title='完整推进仍未减少BP迭代：25 → 25',ylim=(0,max(values)*1.2))
    for ax in axs:ax.spines[['top','right']].set_visible(False)
    fig.savefig(root/'comparison.png',dpi=150);plt.close(fig)
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>局部输出缩放对照</title>
<style>body{{margin:0;background:#edf3f5;color:#243d4a;font:17px "Microsoft YaHei",sans-serif}}main{{max-width:1180px;margin:auto;padding:30px}}p,li{{line-height:1.85}}.lead{{background:#e1f2ee;border-left:5px solid #228f85;padding:20px}}table{{width:100%;background:white;border-collapse:collapse}}th,td{{padding:12px;border-bottom:1px solid #dbe5e9;text-align:left}}img{{width:100%;margin:25px 0}}.note{{font-size:14px;color:#526976}}a{{color:#147f90}}</style>
<main><h1>局部输出缩放：缺陷得到确认，整体尚未加速</h1><p class="lead">保持旧权重、输入和几何完全相同，仅去掉神经输出的全场幅度下限，第一步残差降低{q['fixed_weight_residual_reduction']:.0f}倍。这支持了此前的缩放问题判断。按原预算重新训练并推进一年后，网络采用2/13次，BP仍需25次牛顿迭代，因此继续保留AMG-CG。</p>
<h2>先固定权重，单独验证原因</h2><table><tr><th>第一步固定权重对照</th><th>旧缩放</th><th>新局部缩放</th></tr>
<tr><td>全场BP相对残差</td><td>{a['records']['legacy']['relative_residual']:.9f}</td><td>{a['records']['local']['relative_residual']:.9f}</td></tr>
<tr><td>H&lt;1米节点残差范数 / 全场力范数</td><td>{a['records']['legacy']['H_below1_residual_L2']:.9f}</td><td>{a['records']['local']['H_below1_residual_L2']:.3e}</td></tr>
<tr><td>H&lt;1米节点占残差平方和比例</td><td>{100*a['records']['legacy']['H_below1_fraction_residual_squared']:.6f}%</td><td>{100*a['records']['local']['H_below1_fraction_residual_squared']:.6f}%</td></tr></table>
<p>这一步没有重新训练网络，也没有推进冰川。只将输出幅度从“局部修正与全场下限取较大值”改为“局部修正本身的均方根”。局部修正由BP残差除以该节点的切线刚度对角项得到，因此随局部几何和力学敏感性变化。输入特征仍使用原归一化，避免同时改变输入和输出。没有删去薄冰节点，也没有改冰厚阈值或BP容差。</p>
<img src="comparison.png" alt="固定权重残差对照与完整推进耗时">
<h2>再按相同预算重新训练、推进一年</h2><table><tr><th>指标</th><th>实测</th></tr>
<tr><td>网格与窗口</td><td>100米，第99—100年，13步；30天上限、CFL0.35</td></tr>
<tr><td>网络与训练预算</td><td>1124参数，初始权重及输入归一化与v27逐元素相等；离线40次，每步在线最多4次L-BFGS更新</td></tr>
<tr><td>新预测被采用</td><td>{q['neural_selected']}/13次（原版0/13次）；均仍需BP校正</td></tr>
<tr><td>BP牛顿迭代</td><td>{q['new_Newton']}次，原AMG {q['baseline_Newton']}次</td></tr>
<tr><td>完整推进</td><td>{q['continuation_seconds']:.2f}秒；原AMG历史基准{q['historical_AMG_seconds']:.2f}秒；旧网络历史结果{q['previous_network_seconds']:.2f}秒</td></tr>
<tr><td>训练、准备和新检查</td><td>{q['training_seconds']:.2f}秒；加推进首次使用{q['first_use_seconds']:.2f}秒</td></tr>
<tr><td>最大冰厚差 / 速度相对L2差</td><td>{q['maximum_H_difference_m']:.3e}米 / {q['maximum_velocity_relative_L2']:.3e}</td></tr>
<tr><td>最大质量相对误差 / 原BP残差</td><td>{q['maximum_mass_relative']:.3e} / {q['maximum_original_BP_residual']:.3e}</td></tr></table>
<p>第一项固定权重对照证明了输出缩放的影响；第二项重新训练对照衡量实际运行收益。两者不是同一个预测器，因此不能将第一项的1056倍残差改善解释成推进加速倍数。重新训练仍优化总BP能量，在有限预算内只让两步预测略好于原初始解，没有减少牛顿更新次数。</p>
<p>保留这次局部缩放改动作为研究版本，默认继续使用已有AMG-CG。本轮不扩展网络、不增加训练预算、不自动运行百年试验。</p>
<p class="note">14个状态、在线权重及输出哈希已验证；新输出映射的能量梯度差分和权重恢复检查通过。原BP/存储/状态写入预检复用已封存证据。所有推进耗时含在线训练、残差检查、输运、BP和存盘；首次使用包含准备与训练，完整进程另含固定权重诊断及完成后比对，共{q['total_process_seconds']:.2f}秒。历史耗时受负载及存盘波动影响，不能把相对v27的全部耗时变化归因于缩放。</p>
<p><a href="summary.json">完整结果</a> · <a href="../physics-neural-v1/index.html">上一版物理网络</a> · <a href="../solver-acceleration-v1/index.html">保留的AMG-CG方案</a></p></main></html>'''
    (root/'index.html').write_text(html,encoding='utf-8');t.write_json(root/'delivery.json',dict(files={p.name:t.digest(p) for p in root.iterdir() if p.is_file()}))


def finish():
    t.validate();q=t.read(t.REPORT/'summary.json');assert q['engineering_equivalence'] and q['verified_states']==14
    for p,h in q['outputs'].items():assert t.digest(t.RUN/p)==h
    for p,h in t.read(t.REPORT/'delivery.json')['files'].items():assert t.digest(t.REPORT/p)==h
    t.write_json(t.E/'completion.json',dict(status='completed',summary=t.ref(t.REPORT/'summary.json'),result=t.ref(t.RUN/'result.json'),delivery=t.ref(t.REPORT/'delivery.json'),decision=q['decision'],accepted_count_effect=0))
    a=q['ablation']['records']
    facts=f'''- User authorized targeted outputscale test. Only networkoutputscale changed frommax(localRMS(-r/diagK),0.1globalRMS) tolocalRMS withnoglobalfloor. Inputs/normalization,initialweights,1124parameterarchitecture,40offline/4online budgets andBPphysics/tolerance unchanged. NoHthresholdchange,nothinnode deletion. Newgradient/reloadchecks passed;unchangedoperator/storagepreflightreused.
- Controlled v27firststep fixedweights/features/geometry ablation:relativeBP residual{a['legacy']['relative_residual']:.12g} to{a['local']['relative_residual']:.12g},reduction{q['fixed_weight_residual_reduction']:.6f}times. ThinH<1m residualnorm/globalforcenorm{a['legacy']['H_below1_residual_L2']:.12g} to{a['local']['H_below1_residual_L2']:.12g}. Confirmsglobaloutputfloor defect underthisfixedcase without retrainingoroldtrajectoryreplay.
- Samebudgetretraining and100m99-10013stepcontinuation completed. Neuralaccepted{q['neural_selected']}/13 versusv27zero;Newton{q['new_Newton']}versusAMG25,BPskip0. MaxHdiff{q['maximum_H_difference_m']:.12g}m,velocityL2{q['maximum_velocity_relative_L2']:.12g},massrelative{q['maximum_mass_relative']:.12g},originalBPresidual{q['maximum_original_BP_residual']:.12g};14statesandnetworkweightshashchecked.
- Training/preparation/newpreflight{q['training_seconds']:.9f}s,continuation{q['continuation_seconds']:.9f}s,firstuse{q['first_use_seconds']:.9f}s;historicalAMG{q['historical_AMG_seconds']:.9f}s,v27{q['previous_network_seconds']:.9f}s. NoNewtonreductionornooverallaccelerationversusAMG;historicalwall comparisonsnotpaired. Retainlocal-scalereseachcode butdefaultAMGunchanged,no automaticextension/countchange. Evidence `{t.ref(t.E/'completion.json')['path']}`.'''
    t.transition('completed',t.E/'completion.json',facts)
    home=t.R/'index.html';old=home.read_text(encoding='utf-8');anchor='<h1>2387 冰川模型</h1>';assert anchor in old
    (t.REPORT/'previous-project-index.html').write_text(old,encoding='utf-8')
    home.write_text(old.replace(anchor,anchor+'<a class="current" href="reports/physics-neural-local-scale-v1/index.html"><b>新增：局部输出缩放对照</b><p>固定权重残差降低1056倍；一年推进完成，整体尚未加速。</p></a>',1),encoding='utf-8')


if __name__=='__main__':
    if t.sys.argv[1]=='report':report()
    elif t.sys.argv[1]=='finish':finish()
