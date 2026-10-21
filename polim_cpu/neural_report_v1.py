"""Read-only diagnostics and immutable completion of the finite neural pilot."""
from . import neural_numpy_v1 as experiment
experiment.configure()
n=experiment.n

def report():
    n.validate();s=n.read(n.RUN/'result.json');c=s['continuation'];root=n.REPORT;root.mkdir(exist_ok=False)
    store=n.Store(n.RUN/'continuation',c['contract'],n.source(100,99)[0]['A']);predict=n.Predictor();b=predict.b;diagnostics=[]
    for i in range(1,14):
        old,u,_=store.restore(f'frame-{i-1:03d}');st,v,_=store.restore(f'frame-{i:03d}');dt=st['time']-old['time']
        rate=(v[:,1:]-u[:,1:]).ravel()/dt;proj=b['ym']+((rate-b['ym'])@b['yb'])@b['yb'].T
        guessed=predict(st['h'],old['h'],u,dt);pred_rate=(guessed-u[:,1:]).ravel()/dt
        diagnostics.append(dict(step=i,rate_projection_relative_L2=float(n.np.linalg.norm(proj-rate)/n.np.linalg.norm(rate)),
            neural_rate_relative_L2=float(n.np.linalg.norm(pred_rate-rate)/n.np.linalg.norm(rate))))
    train_times=[n.load_frame(i)[0]['time'] for i in n.TRAIN];val_times=[n.load_frame(i)[0]['time'] for i in n.VALID]
    summary=dict(status='completed',decision='retain-v23-AMG-CG; do not enable this neural predictor',grid_m=100,test_years=[99,100],
        train_samples=len(n.TRAIN),train_time_range=[min(train_times),max(train_times)],validation_samples=len(n.VALID),validation_time_range=[min(val_times),max(val_times)],
        network_parameters=s['training']['parameters'],epochs=1200,training_seconds=s['training']['training_seconds'],
        data_POD_seconds=s['dataset']['seconds'],training_and_weights_io_seconds=s['training_process']['seconds'],
        predictor_seconds=sum(r['inference_seconds'] for r in c['rows']),screen_seconds=sum(r['screen_seconds'] for r in c['rows']),
        continuation_seconds=c['wall_seconds'],first_use_total_seconds=s['amortized_test_seconds'],historical_baseline_seconds=c['historical_baseline_seconds'],
        end_to_end_with_fixture_and_postcheck_seconds=s['total_elapsed_seconds']+s['dataset']['seconds'],
        baseline_Newton=c['baseline_Newton'],new_Newton=c['new_Newton'],neural_selected=c['neural_selected'],BP_skipped=c['BP_skipped'],steps=c['steps'],
        maximum_H_difference_m=max(x['H_max_m'] for x in c['comparison']),maximum_velocity_relative_L2=max(x['velocity_relative_L2'] for x in c['comparison']),
        maximum_original_BP_residual=max(x['original_BP_residual'] for x in c['comparison']),maximum_mass_relative=max(r['mass_relative'] for r in c['rows']),
        residual_ratio_range=[min(r['neural_initial_residual']/r['old_initial_residual'] for r in c['rows']),max(r['neural_initial_residual']/r['old_initial_residual'] for r in c['rows'])],
        engineering_equivalence=c['engineering_equivalence'],diagnostics=diagnostics,
        verification='13 accepted states independently checked against original BP; all14 states hash-reloaded; directional NN gradient difference and exact weight reload passed.',
        scope='Single fixed geometry family/bed/material zeroSMB trajectory, supervised POD+MLP warmstart; not IGM energy-trained CNN. No general conclusion against physics-trained neural solvers.',
        timing_note='Historical v23 baseline, not controlled simultaneous benchmark. Lower continuation wall does not establish neural speedup: every candidate rejected, Newton unchanged. Data/training amortization is first-use only; setup/debugging operator time excluded, numerical first-use and fixture/postcheck totals explicit.',
        all_outputs={p.relative_to(n.RUN).as_posix():n.digest(p) for p in n.RUN.rglob('*') if p.is_file()})
    n.write_json(root/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    plt.rcParams['font.family']=FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name();plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    vals=[summary['historical_baseline_seconds'],summary['continuation_seconds'],summary['first_use_total_seconds']]
    bars=ax[0].bar(['原AMG\n历史基准','网络＋BP\n仅推进','首次使用\n含数据与训练'],vals,color=['#248d86','#6489a7','#bd7747']);ax[0].bar_label(bars,fmt='%.2f秒',padding=4)
    ax[0].set(ylabel='秒',title='没有减少BP工作量：25次 → 25次',ylim=(0,max(vals)*1.2))
    ax[1].semilogy(range(1,14),[r['old_initial_residual'] for r in c['rows']],'-o',label='沿用上一步速度',color='#248d86')
    ax[1].semilogy(range(1,14),[r['neural_initial_residual'] for r in c['rows']],'-o',label='网络预测',color='#bd7747')
    ax[1].axhline(1e-6,color='#555',ls='--',label='BP收敛标准');ax[1].set(xlabel='推进步',ylabel='BP相对残差',title='13次预测均未改善初始解');ax[1].legend()
    for a in ax:a.spines[['top','right']].set_visible(False)
    fig.savefig(root/'comparison.png',dpi=150);plt.close(fig)
    q=summary
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>100米神经网络加速测试</title>
<style>body{{background:#edf2f5;color:#213c4a;font:17px "Microsoft YaHei",sans-serif;margin:0}}main{{max-width:1150px;margin:auto;padding:32px}}p,li{{line-height:1.8}}.lead{{background:#fff4e5;padding:22px;border-left:5px solid #bd7747}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{padding:13px;text-align:left;border-bottom:1px solid #ddd}}img{{width:100%;margin:25px 0}}a{{color:#147d87}}.small{{font-size:14px;color:#566875}}</style>
<main><h1>100米神经网络测试：这版暂不启用</h1><p class="lead">第99—100年、13次推进全部完成。网络预测均未改善BP初始残差，全部采用原初始解，牛顿迭代仍为25次。保留已有优化版AMG-CG。</p>
<table><tr><th>项目</th><th>实测结果</th></tr><tr><td>网络</td><td>37输入 → 64 → 64 → 12输出，{q['network_parameters']}参数；NumPy双精度Adam</td></tr>
<tr><td>数据划分</td><td>训练206状态，第{min(train_times):.2f}—{max(train_times):.2f}年；验证26状态，第{min(val_times):.2f}—{max(val_times):.2f}年；测试第99—100年</td></tr>
<tr><td>训练</td><td>1200轮，{q['training_seconds']:.3f}秒；准备数据和POD {q['data_POD_seconds']:.3f}秒</td></tr>
<tr><td>13次网络预测 / 残差筛选</td><td>{q['predictor_seconds']:.4f}秒 / {q['screen_seconds']:.3f}秒</td></tr>
<tr><td>完整推进 / 首次使用总计</td><td>{q['continuation_seconds']:.2f}秒 / {q['first_use_total_seconds']:.2f}秒；历史AMG基准{q['historical_baseline_seconds']:.2f}秒</td></tr>
<tr><td>BP计算量</td><td>牛顿25 → 25；网络采用0/13；跳过BP 0/13</td></tr>
<tr><td>与基准差异</td><td>最大冰厚差0米，速度差0；质量相对误差{q['maximum_mass_relative']:.2e}</td></tr></table>
<img src="comparison.png" alt="完整成本和BP初始残差比较">
<h2>这次测试说明什么</h2><p>网络推断只需毫秒，训练也很快；困难在于预测精度。此模型将全冰川速度增量压缩成12个空间模式，再用小网络预测系数。在未参与训练的末期年份，网络预测的BP残差约为旧初始解的{q['residual_ratio_range'][0]:.1f}—{q['residual_ratio_range'][1]:.1f}倍，因此没有减少求解工作量。</p>
<p>这项结果针对本次“监督学习＋低阶空间压缩＋严格BP校正”原型。它没有检验IGM那种直接用冰流能量训练的空间网络。若继续深度学习路线，应研究物理损失和空间结构，而不能从本次毫秒级推断时间推算千年加速倍数。</p>
<p class="small">原始条件保留：固定床面和材料、零表面质量收支、底部无滑动、30天上限、CFL0.35、原BP收敛标准10⁻⁶。最终13个状态逐一检查原BP残差，最大{q['maximum_original_BP_residual']:.2e}；14个保存状态重读验证。网络反向传播方向导数误差{s['training']['directional_gradient_relative_error']:.2e}，权重重读一致。测试结果一致是因为所有神经预测均回退，不代表网络本身精确。</p>
<p class="small">时间来自不同运行时段，受系统负载和存盘波动影响。本次仅推进9.02秒略低于历史9.48秒，不能归因于神经网络。首次使用{q['first_use_total_seconds']:.2f}秒含数据准备、训练、预测、筛选、BP和推进存盘；再计入合并预检和完成后比对共{q['end_to_end_with_fixture_and_postcheck_seconds']:.2f}秒。前期环境排错时间另属开发成本，未混入这些数值运行计时。既有PyTorch环境在训练前中止，最终使用NumPy，未安装依赖或使用不安全OpenMP开关。</p>
<p><a href="summary.json">完整结果与输出哈希</a> · <a href="../solver-acceleration-v1/index.html">保留的AMG-CG方案</a> · <a href="https://doi.org/10.1017/jog.2023.73">IGM物理训练方法参考</a></p></main></html>'''
    (root/'index.html').write_text(html,encoding='utf-8')
    n.write_json(root/'delivery.json',dict(files={p.name:n.digest(p) for p in root.iterdir() if p.is_file()}))

def finish():
    n.validate();q=n.read(n.REPORT/'summary.json');assert q['engineering_equivalence'] and q['steps']==13
    for p,h in q['all_outputs'].items():assert n.digest(n.RUN/p)==h
    for p,h in n.read(n.REPORT/'delivery.json')['files'].items():assert n.digest(n.REPORT/p)==h
    n.write_json(n.E/'completion.json',dict(status='completed',result=n.ref(n.RUN/'result.json'),summary=n.ref(n.REPORT/'summary.json'),delivery=n.ref(n.REPORT/'delivery.json'),decision=q['decision'],accepted_count_effect=0))
    facts=f'''- User authorized neural acceleration test and continuation after errors. Existing torch environment attempts stopped before training/BP; preserved inputs reused. Disclosed final NumPy float64 POD12+MLP37-64-64-12,7372parameters,Adam1200epochs,seed2387. Train206 saved100m states beforeyear80,validate26 beforeyear90; continuous99-100 heldout,zeroSMB/fixedbed/A/noslip/30daycap/CFL0.35. No test fitting,originalBP tolerance1e-6 retained.
- Completed13steps. Neural candidates selected0/13,all had larger residual than previous velocity. Newton25 versus retainedAMG25,noBPskips. H andvelocity bitwise identicalto baseline;maximum originalBP residual{q['maximum_original_BP_residual']:.12g},massrelative{q['maximum_mass_relative']:.12g}. All14states reloaded;backprop directional finite-difference and exactweights reload passed.
- Dataset/POD{q['data_POD_seconds']:.9f}s,trainingandweightsIO{q['training_and_weights_io_seconds']:.9f}s,inference{q['predictor_seconds']:.9f}s,screen{q['screen_seconds']:.9f}s,continuation{q['continuation_seconds']:.9f}s,first-usecost{q['first_use_total_seconds']:.9f}s;historicalAMG{q['historical_baseline_seconds']:.9f}s. Wall-load variations are not neural acceleration; no reductioninBPwork. Retainv23AMG;do not enable this predictor. ResultdoesnotrejectIGMenergytrainedspatialnetworks. No further automatic experiment,no default/count change. Evidence `{n.ref(n.E/'completion.json')['path']}`.'''
    n.transition('completed',n.E/'completion.json',facts)
    home=n.R/'index.html';text=home.read_text(encoding='utf-8');anchor='<h1>2387 冰川模型</h1>';assert anchor in text
    (n.REPORT/'previous-project-index.html').write_text(text,encoding='utf-8')
    home.write_text(text.replace(anchor,anchor+'<a class="current" href="reports/neural-predictor-v1/index.html"><b>新增：神经网络加速测试</b><p>100米一年对照完成；本版未减少BP迭代，保留AMG-CG。</p></a>',1),encoding='utf-8')

if __name__=='__main__':
    if n.sys.argv[1]=='report':report()
    elif n.sys.argv[1]=='finish':finish()
