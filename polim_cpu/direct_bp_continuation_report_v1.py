"""Full nonlinear direct BP comparison and bounded experiment closure."""
import numpy as np
from .direct_bp_continuation_v1 import (RUN,REPORT,E,R,BASELINE,Store,read,arrays,ref,
                                       hashes,write_json,digest,transition,validate)


def report():
    validate();out=read(RUN/'result.json');assert out['implementation']==hashes()
    assert out['status']=='completed' and len(out['cases'])==4
    REPORT.mkdir(parents=True,exist_ok=False);cases=[];verified=0
    for c in sorted(out['cases'],key=lambda a:(a['grid_m'],a['start_year'])):
        root=RUN/c['case'];st=Store(root,c['contract'],arrays(root/'static-A.npz')['A'])
        for p in root.glob('frame-*'):
            s,u,_=st.restore(p.name);assert np.isfinite(s['h']).all() and np.min(s['h'])>=0 and np.isfinite(u).all();verified+=1
        old=read(BASELINE/c['case']/'result.json')
        newtimers={k:sum(row['momentum']['timers'][k] for row in c['rows']) for k in c['rows'][0]['momentum']['timers']}
        oldtimers={k:sum(row['momentum']['timers'][k] for row in old['rows']) for k in old['rows'][0]['momentum']['timers']}
        a={k:c[k] for k in ['case','grid_m','start_year','end_year','wall_seconds','historical_baseline_seconds',
            'new_BP_seconds','old_BP_seconds','engineering_equivalence','direct_factorizations',
            'maximum_mass_relative','final_original_BP_residual','peak_sampled_RSS_MiB','steps']}
        a.update(AMG_newton_updates=sum(r['momentum']['iterations'] for r in old['rows']),
                 LU_newton_updates=sum(r['momentum']['iterations'] for r in c['rows']),
                 AMG_preconditioner_builds=sum(r['momentum']['amg_builds'] for r in old['rows']),
                 new_timers=newtimers,old_timers=oldtimers,
                 elapsed_increase_percent=(c['wall_seconds']/c['historical_baseline_seconds']-1)*100,
                 maximum_H_difference_m=max(v['H_max_m'] for v in c['comparison']),
                 maximum_H_RMS_m=max(v['H_RMS_m'] for v in c['comparison']),
                 maximum_velocity_relative_L2=max(v['velocity_relative_L2'] for v in c['comparison']),
                 source=ref(root/'result.json'),baseline=ref(BASELINE/c['case']/'result.json'))
        cases.append(a)
    summary=dict(status='completed',all_equivalence_pass=all(c['engineering_equivalence'] for c in cases),
                 cases=cases,verified_states=verified,BP_solves=sum(c['steps'] for c in cases),
                 decision='Retain optimizedv23AMG-CG for subsequent runs;do not adopt SuperLU as default after no full-window speed benefit.',
                 default_solver_changed=False,source=ref(RUN/'result.json'))
    assert summary['all_equivalence_pass']
    write_json(REPORT/'summary.json',summary)
    write_json(REPORT/'solver-decision.json',dict(selection='retain-AMG-CG',kernel=ref(R/'bp_accelerated_v1.py'),
                 reason='AllfourfullSuperLUwindows tooklongerthanretainedsame-settingsv23;noNewtoniterationreduction;AMGreusespreconditionerwithinBP.',
                 grid_or_time_settings_changed=False,additional_tests_required=False,summary=ref(REPORT/'summary.json')))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(1,2,figsize=(12.5,4.7),constrained_layout=True);x=np.arange(4)
    labels=[f"{c['grid_m']}米\n第{c['start_year']}→{c['end_year']}年" for c in cases]
    for i,(field,label,color) in enumerate([('historical_baseline_seconds','优化版AMG-CG','#148d86'),('wall_seconds','SuperLU完整BP','#d69474')]):
        bars=axs[0].bar(x+(i-.5)*.36,[c[field] for c in cases],.36,label=label,color=color);axs[0].bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
    axs[0].set(xticks=x,xticklabels=labels,ylabel='推进一年耗时（秒）',title='完整推进总耗时（包含存盘）');axs[0].legend()
    for i,(field,label,color) in enumerate([('AMG_preconditioner_builds','AMG构建次数','#148d86'),('direct_factorizations','LU分解次数','#d69474')]):
        bars=axs[1].bar(x+(i-.5)*.36,[c[field] for c in cases],.36,label=label,color=color);axs[1].bar_label(bars,padding=3)
    axs[1].set(xticks=x,xticklabels=labels,ylabel='次数',title='每个一年窗口的辅助结构构建');axs[1].legend()
    for ax in axs:ax.spines[['top','right']].set_visible(False)
    fig.savefig(REPORT/'comparison.png',dpi=150);plt.close(fig)
    table='';detail=''
    for c in cases:
        table+=f'<tr><td>{c["grid_m"]}米</td><td>{c["start_year"]}→{c["end_year"]}年</td><td>{c["historical_baseline_seconds"]:.2f}秒</td><td>{c["wall_seconds"]:.2f}秒</td><td>+{c["elapsed_increase_percent"]:.1f}%</td><td>{c["maximum_H_difference_m"]:.2e}米</td><td>{c["maximum_velocity_relative_L2"]:.2e}</td></tr>'
        ot=c['old_timers'];nt=c['new_timers']
        detail+=f'<tr><td>{c["grid_m"]}米·第{c["start_year"]}年起</td><td>{ot["amg"]+ot["cg"]:.3f}秒</td><td>{nt["factor"]+nt["direct_solve"]:.3f}秒</td><td>{ot["checkpoint"]:.3f} / {nt["checkpoint"]:.3f}秒</td><td>{c["AMG_newton_updates"]} / {c["LU_newton_updates"]}</td></tr>'
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SuperLU完整BP推进对照</title>
<style>body{{margin:0;background:#eef3f7;color:#20394b;font:17px "Microsoft YaHei",sans-serif}}main{{max-width:1350px;margin:auto;padding:30px}}p,li{{line-height:1.85}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{padding:12px;border-bottom:1px solid #dce5eb;text-align:left}}.lead{{padding:20px;background:#e0f3ee;border-left:5px solid #148d86}}img{{width:100%;margin:22px 0;background:white;border-radius:12px}}.note{{font-size:15px;color:#526779}}a{{color:#087f9e}}</style>
<main><h1>SuperLU完整BP推进：保留优化版AMG-CG</h1><p class="lead">四组一年推进全部完成，冰厚、速度、质量守恒和原BP方程检查通过；SuperLU没有在完整推进中带来加速。本次比较到此结束，后续保留已有AMG-CG方案。</p>
<table><tr><th>网格</th><th>窗口</th><th>优化版AMG-CG</th><th>SuperLU</th><th>耗时增加</th><th>最大冰厚差</th><th>最大速度相对L2差</th></tr>{table}</table>
<p class="note">相同30天上限、CFL0.35、CPU8/BLAS1配置、几何缓存、非线性容差10⁻⁶和逐步检查点。AMG时间来自已保存v23窗口，未重新跑原轨迹；硬件负载和存盘延迟不完全相同，所以表中百分比是历史耗时对比。新计时包含所有推进、BP和检查点，编译、合并预检及完成后的结果对比单独计时。</p>
<img src="comparison.png" alt="完整推进耗时和AMG构建次数与LU分解次数比较">
<h2>为何单矩阵较快，完整推进却没有更快</h2><p>之前的矩阵测试比较了从头构建AMG与从头做LU分解。完整BP中，AMG可以在牛顿迭代之间复用；LU则在每次切线矩阵更新后重新分解。四组的牛顿更新次数相同，LU更精确的线性解没有减少外层迭代。每个窗口AMG只构建13次，LU需分解25或26次。</p>
<table><tr><th>窗口</th><th>AMG构建＋CG</th><th>LU缩放/分解＋求解</th><th>BP内部存盘：AMG / LU</th><th>牛顿更新：AMG / LU</th></tr>{detail}</table>
<p class="note">LU分项还包含对角缩放与矩阵格式转换，AMG分项不含这部分，二者分项边界并非完全一致；完整BP和总推进耗时另行记录。存盘延迟也贡献了总耗时变化，不能将全部百分比归因于求解器本身。</p>
<h2>完成检查与决定</h2><p>52次新BP求解、56个接受状态重新读取校验；四组结果均满足预先登记的工程差异标准。仅替换线性求解方式，BP物理、固定床面/材料、零表面质量收支及边界处理与原试算一致。决定保留优化版AMG-CG，网格和时间设置不变。</p>
<p><a href="summary.json">完整数据</a> · <a href="solver-decision.json">求解器选择记录</a> · <a href="../linear-solver-comparison-v1/index.html">此前同矩阵测试</a> · <a href="../solver-acceleration-v1/index.html">优化版AMG推进结果</a></p></main></html>'''
    (REPORT/'index.html').write_text(html,encoding='utf-8')
    write_json(REPORT/'delivery.json',dict(files={p.name:digest(p) for p in REPORT.iterdir() if p.is_file()}))


def finish():
    validate();s=read(REPORT/'summary.json');assert s['all_equivalence_pass'] and s['verified_states']==56
    for name,h in read(REPORT/'delivery.json')['files'].items():assert digest(REPORT/name)==h
    home=R/'index.html';old=home.read_text(encoding='utf-8');anchor='<h1>2387 冰川模型</h1>';assert anchor in old
    (REPORT/'previous-project-index.html').write_text(old,encoding='utf-8')
    link='<a class="current" href="reports/direct-bp-continuation-v1/index.html"><b>新增：完整BP求解器对照</b><p>SuperLU四组一年推进已完成，保留优化版AMG-CG。</p></a>'
    home.write_text(old.replace(anchor,anchor+link,1),encoding='utf-8')
    write_json(E/'completion.json',dict(status='completed',summary=ref(REPORT/'summary.json'),delivery=ref(REPORT/'delivery.json'),
                 decision=ref(REPORT/'solver-decision.json'),result=ref(RUN/'result.json'),accepted_count_effect=0))
    facts='- UserauthorizedcompleteSuperLU-BP50/100mone-yearwindows1-2and99-100. Samev23nonlinearworkflow/cache/rawcheckpointing,30daycap/CFL0.35,CPU8/BLAS1,fixedbed/A/zeroSMB;onlylinearsolverSuperLUMMD_AT_PLUS_A,directtruephysicalresidual1e-8,outer1e-6. Retainedv23AMGtrajectoriesreadwithoutreplay.\n'
    for c in s['cases']:
        facts+=f'- {c["case"]}: AMGretainedwall{c["historical_baseline_seconds"]:.9f}s,LUwall{c["wall_seconds"]:.9f}s;Hmaxdiff{c["maximum_H_difference_m"]:.12g}m,uL2max{c["maximum_velocity_relative_L2"]:.12g},massrelative{c["maximum_mass_relative"]:.12g};AMG/LUnewtonupdates{c["AMG_newton_updates"]}/{c["LU_newton_updates"]},AMGbuilds{c["AMG_preconditioner_builds"]},LUfactorizations{c["direct_factorizations"]}.\n'
    facts+='- All4engineeringcomparisons/trueBPchecks passed,52newBPsolves,56stateshashrestored. FullLUwindowsnotfaster;AMGreusebenefitandunchangedNewtoncountsremovefirst-matrixadvantage. Historicalmachine-load/IOdifferencesalsoaffectwallratios;notallslowdownattributedtosolver. Retainoptimizedv23AMG-CG,noadditionalsolvercomparisonrequired,no90daystep,noclimatechange,nodefaultswitchorformalcountchange. Evidence `'+ref(E/'completion.json')['path']+'`.'
    transition('completed',E/'completion.json',facts)


if __name__=='__main__':
    import sys
    if sys.argv[1]=='report':report()
    elif sys.argv[1]=='finish':finish()
