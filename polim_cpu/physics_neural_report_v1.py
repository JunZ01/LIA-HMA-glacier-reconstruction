"""Report and close the bounded physics-trained graph-network experiment."""
from . import physics_neural_trial_v1 as t


def report():
    t.validate();s=t.read(t.RUN/'result.json');c=s['continuation'];train=s['training'];d=t.read(t.RUN/'postrun-diagnostic.json')
    root=t.REPORT;root.mkdir(exist_ok=False)
    initial,_,_=t.source(100,99);store=t.Store(t.RUN/'continuation',c['contract'],initial['A'])
    base=t.read(t.BASELINE/'result.json');bs=t.Store(t.BASELINE,base['contract'],initial['A']);exact=True
    for i in range(14):
        a,u,_=store.restore(f'frame-{i:03d}');b,v,_=bs.restore(f'frame-{i:03d}')
        exact=exact and t.np.array_equal(a['h'],b['h']) and t.np.array_equal(u,v)
    summary=dict(status='completed',decision='Do not enable this network; retain optimized v23 AMG-CG. No automatic extension.',
        physics_training=True,physics_loss='Unchanged discrete BP energy; manual exact chain-rule through mesh graph network.',
        network_parameters=train['parameters'],train_years=t.TRAIN_YEARS,validation_years=t.VALID_YEARS,test_years=[99,100],
        online_training='At most4 L-BFGS updates perstep usingcurrent geometry/energy; no referencevelocity labels. Heldout dynamics are not frozen-weight inference.',
        offline_iterations=train['optimizer']['iterations'],offline_objective_calls=train['optimizer']['objective_calls'],
        offline_optimizer_converged=train['optimizer']['success'],offline_optimizer_message=train['optimizer']['message'],
        training_total_seconds=train['total_seconds'],offline_optimizer_seconds=train['optimizer']['seconds'],
        continuation_seconds=c['wall_seconds'],first_use_seconds=s['first_use_seconds'],historical_AMG_seconds=c['historical_baseline_seconds'],
        postcheck_seconds=c['verification_seconds'],total_process_seconds=s['total_seconds'],
        online_seconds=sum(r['online']['seconds'] for r in c['rows']),online_objective_calls=sum(r['online']['objective_calls'] for r in c['rows']),
        features_and_initial_screen_seconds=sum(r['feature_and_initial_screen_seconds'] for r in c['rows']),final_screen_seconds=sum(r['final_screen_seconds'] for r in c['rows']),
        baseline_BP_seconds=c['baseline_BP_seconds'],new_BP_seconds=c['new_BP_seconds'],baseline_Newton=c['baseline_Newton'],new_Newton=c['new_Newton'],
        neural_selected=c['neural_selected'],BP_skipped=c['BP_skipped'],steps=c['steps'],verified_states=c['verified_states'],exact_H_and_full_velocity=bool(exact),
        maximum_H_difference_m=max(x['H_max_m'] for x in c['comparison']),maximum_velocity_relative_L2=max(x['velocity_relative_L2'] for x in c['comparison']),
        maximum_original_BP_residual=max(x['original_BP_residual'] for x in c['comparison']),maximum_mass_relative=max(x['mass_relative'] for x in c['rows']),
        engineering_equivalence=c['engineering_equivalence'],acceleration_gate=c['acceleration_gate'],first_step_diagnostic=d,
        limitation='Finite40offline/4online iteration budget reached,not fully converged neural optimization. Energy decreased but thin-node momentum residual remainslarge. This one1124parameterarchitecture/outputscaling does not test allphysicsinformednetworks.',
        timing='HistoricalAMG comparison,not simultaneous; includesneuralstatecheckpointIO. Trainingtotal andfirstuse includecombinedpreflight/preparation; continuationincludesonlineupdates/screens/BP/transport/allcheckpoints. Postrundiagnostic separate.',
        outputs={p.relative_to(t.RUN).as_posix():t.digest(p) for p in t.RUN.rglob('*') if p.is_file()})
    t.write_json(root/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    plt.rcParams['font.family']=FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name();plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    values=[c['historical_baseline_seconds'],c['wall_seconds'],s['first_use_seconds']]
    bars=axes[0].bar(['原AMG\n历史基准','物理网络＋BP\n一年推进','首次使用\n含训练与合并预检'],values,color=['#13877c','#cf8b48','#577d99'])
    axes[0].bar_label(bars,fmt='%.2f秒',padding=4);axes[0].set(ylabel='秒',ylim=(0,max(values)*1.2),title='总耗时增加，BP迭代25 → 25')
    axes[1].semilogy(range(1,14),[r['old_initial_residual'] for r in c['rows']],'-o',label='沿用上一时刻速度',color='#13877c')
    axes[1].semilogy(range(1,14),[r['updated_network_residual'] for r in c['rows']],'-o',label='物理训练网络预测',color='#cf8b48')
    axes[1].axhline(1e-6,ls='--',color='#667',label='原BP收敛标准');axes[1].set(xlabel='推进步',ylabel='BP相对残差',title='能量下降未转化为较小方程残差');axes[1].legend()
    for a in axes:a.spines[['top','right']].set_visible(False)
    fig.savefig(root/'comparison.png',dpi=150);plt.close(fig)
    q=summary
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>物理约束小网络测试</title>
<style>body{{margin:0;background:#eef3f6;color:#233c4b;font:17px "Microsoft YaHei",sans-serif}}main{{max-width:1200px;margin:auto;padding:30px}}p,li{{line-height:1.8}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{padding:12px;border-bottom:1px solid #dbe3e8;text-align:left}}.lead{{background:#fff3e3;border-left:5px solid #cf8b48;padding:20px}}.note{{font-size:14px;color:#576b78}}img{{width:100%;margin:22px 0}}a{{color:#087d91}}</style>
<main><h1>物理约束小网络：100米一年测试完成</h1><p class="lead">这次将BP能量直接用于训练，网络有1124个参数。训练和13次推进全部完成，原BP与守恒检查通过；网络预测13次均回退，BP迭代仍为25次，总耗时增加。本版不启用，保留已有AMG-CG。</p>
<table><tr><th>项目</th><th>结果</th></tr><tr><td>训练物理</td><td>原BP离散能量、原Glen材料和积分；底部速度恒为零；图网络显式反向传播</td></tr>
<tr><td>网络与输入</td><td>两层16通道图网络，1124参数；网格邻接交换信息；几何、坡度、材料、旧速度与局部对角近似修正</td></tr>
<tr><td>训练范围</td><td>第1、5、10、20、30、40、50、60、70、79年各一个输运后状态；85、90年只做诊断；99—100年连续推进留出</td></tr>
<tr><td>优化预算</td><td>离线40次L-BFGS迭代；每步在线最多4次，仅使用当前几何与BP能量，不读取参考速度标签。达到预算上限，未宣称网络优化已收敛。</td></tr>
<tr><td>训练与准备总计</td><td>{q['training_total_seconds']:.2f}秒，其中优化及网络存盘{q['offline_optimizer_seconds']:.2f}秒；包括合并预检</td></tr>
<tr><td>一年推进 / 首次使用总计</td><td>{q['continuation_seconds']:.2f}秒 / {q['first_use_seconds']:.2f}秒；历史AMG基准{q['historical_AMG_seconds']:.2f}秒</td></tr>
<tr><td>推进中的在线训练</td><td>{q['online_seconds']:.2f}秒，{q['online_objective_calls']}次能量与梯度计算，含逐迭代权重存盘</td></tr>
<tr><td>结果与BP计算量</td><td>网络采用0/13，跳过BP 0/13；牛顿25 → 25；14个完整冰厚及速度数组与基准逐元素相等</td></tr></table>
<img src="comparison.png" alt="物理网络成本与BP残差对照">
<h2>为何有物理训练仍未加速</h2><p>第一步中，网络让BP能量下降了约{abs(d['energy_change']):.0f}，但相对方程残差由{d['old_residual']:.3g}升至{d['neural_residual']:.3g}。训练追求的总能量下降，不等于有限迭代后每个节点都已满足应力平衡。</p>
<p>对第一步残差逐节点分解，<b>{100*d['residual_share_by_H']['1']:.2f}%的残差平方和位于冰厚不足1米的节点</b>。残差最大的节点冰厚仅{d['top_nodes'][0]['H']:.2e}米，网络施加的速度变化虽只有约10⁻⁸—10⁻⁷米/年，仍产生了较大局部残差。本版输出缩放使用全场修正幅度下限，使极薄节点也接收非零神经修正；这是需要优先处理的具体设计问题。本轮未修改薄冰阈值、物理方程或收敛标准来隐藏它。</p>
<p>局部对角近似本身的第一步残差为{d['cheap_diagonal_residual']:.3g}，原始初始解为{d['old_residual']:.3g}。增加网络后的大残差来自本版预测场，而非BP缓存算子替换：使用原始BP算子重新计算，结果完全一致。训练后网络梯度的方向差分相对误差为{d['trained_network_gradient_error']:.2e}。</p>
<h2>本轮结论</h2><p>在预先规定的训练和在线更新预算内，这个小图网络没有形成足够准确的初始解，也没有减少BP工作量。本次结果限定于这套网络结构、输出缩放和训练预算，不能推广为物理约束神经网络不可行。若继续研究，应先改进极薄冰节点的输出参数化及能量训练的尺度处理，再谈扩大网络或训练时长。本轮到此结束，不自动展开调参或百年试算。</p>
<p class="note">100米网格、30天上限、CFL0.35、固定床面/材料、零表面质量收支、底部无滑动保持一致。13次新BP校正、14个状态哈希重读及全部在线权重检查；最大原BP残差{q['maximum_original_BP_residual']:.2e}，最大质量相对误差{q['maximum_mass_relative']:.2e}。与基准完全一致是因为预测全部回退，不能据此认为网络预测准确。</p>
<p class="note">原AMG时间来自已保存v23，不是同时段配对测速。所有新推进计时含特征、在线优化、残差筛选、BP、输运和检查点；首次使用还含离线准备、合并预检、训练及验证。完整进程{q['total_process_seconds']:.2f}秒，后续定位诊断另行计时。不能用网络推断时间代替完整成本。</p>
<p><a href="summary.json">详细结果与哈希</a> · <a href="../neural-predictor-v1/index.html">此前监督学习测试</a> · <a href="../solver-acceleration-v1/index.html">保留的AMG-CG方案</a></p></main></html>'''
    (root/'index.html').write_text(html,encoding='utf-8');t.write_json(root/'delivery.json',dict(files={p.name:t.digest(p) for p in root.iterdir() if p.is_file()}))


def finish():
    t.validate();q=t.read(t.REPORT/'summary.json');assert q['engineering_equivalence'] and q['verified_states']==14
    for p,h in q['outputs'].items():assert t.digest(t.RUN/p)==h
    for p,h in t.read(t.REPORT/'delivery.json')['files'].items():assert t.digest(t.REPORT/p)==h
    t.write_json(t.E/'completion.json',dict(status='completed',result=t.ref(t.RUN/'result.json'),summary=t.ref(t.REPORT/'summary.json'),delivery=t.ref(t.REPORT/'delivery.json'),decision=q['decision'],accepted_count_effect=0))
    facts=f'''- User authorized one100m spatial-network BP-energy training and99-100 continuation test. New1124parameter,two16channelgraphlayers,NumPyfloat64,hardbasalzero/drymask. InputsincludeexactBP residual/localtangentdiagonal. Train10earlygeometries(years1,5,10,20,30,40,50,60,70,79),40L-BFGSsteps;85/90diagnostic only;online4steps perteststepusingcurrentenergy,noreferencevelocitylabels. Trainingbudget reached,notfullyconverged optimization. SameBPquadrature/A/REG,30daycap/CFL0.35,zeroSMB/fixedbed/material,tolerance1e-6.
- Operator/storage/restart/status fixture,diagonal and energygradient checks passed;13newBPsolves,14states andonlineweights hashchecked. All13neuralcandidatesrejected;Newton25versus25,BPskip0;completeH/u arrays identicaltobaseline. Maxoriginalresidual{q['maximum_original_BP_residual']:.12g},massrelative{q['maximum_mass_relative']:.12g}.
- Training/preparation/combinedpreflight{q['training_total_seconds']:.9f}s,continuation{q['continuation_seconds']:.9f}s,firstuse{q['first_use_seconds']:.9f}s,retainedhistoricalAMG{q['historical_AMG_seconds']:.9f}s. No acceleration;historicaltiming not pairedcausalbenchmark. Retainv23AMG,do not enablethispredictor,default/counts unchanged,noautomaticextension.
- Firststep read-onlydiagnostic:energy decreasedbutrelativeBP residual0.260792008;99.828325percent ofresidualsquaredsum onH<1m nodes. MaxresidualnodeH1.715435e-6m. Tinyneuralperturbations onextremelythin nodes are poorly scaledbythisoutputmapping's globalcorrectionscale floor. Originaloperator givesexactly sameresidual;trainednetworkdirectionalgradienterror1.501423e-4. This identifiesa localized outputparameterization problem;doesnotdisproveotherphysics-trainednetworks. No thresholdorphysics change. Evidence `{t.ref(t.E/'completion.json')['path']}`.'''
    t.transition('completed',t.E/'completion.json',facts)
    home=t.R/'index.html';old=home.read_text(encoding='utf-8');anchor='<h1>2387 冰川模型</h1>';assert anchor in old
    (t.REPORT/'previous-project-index.html').write_text(old,encoding='utf-8')
    home.write_text(old.replace(anchor,anchor+'<a class="current" href="reports/physics-neural-v1/index.html"><b>新增：物理约束小网络测试</b><p>100米一年完成；定位极薄冰节点预测残差，保留AMG-CG。</p></a>',1),encoding='utf-8')


if __name__=='__main__':
    if t.sys.argv[1]=='report':report()
    elif t.sys.argv[1]=='finish':finish()
