"""Finalize the bounded BP acceleration experiment, preserving frozen runs."""
import json
import numpy as np
from .solver_acceleration_v1 import (RUN,REPORT,E,R,NAME,Store,read,ref,hashes,
                                    write_json,digest,transition,validate)


def report():
    validate(); out=read(RUN/'result.json'); plan=read(E/'plan.frozen.json')
    assert out['implementation']==hashes()==plan['implementation']
    REPORT.mkdir(parents=True,exist_ok=False)
    cases=[]; verified=0
    for r in out['cases']:
        assert r['status']=='completed',r
        root=RUN/r['case']; static=np.load(root/'static-A.npz')['A']; st=Store(root,r['contract'],static)
        for path in sorted(root.glob('frame-*')):
            state,u,_=st.restore(path.name)
            assert np.isfinite(state['h']).all() and np.min(state['h'])>=0 and np.isfinite(u).all()
            verified+=1
        fixture=read(root/'preflight/verification.json'); assert fixture['status']=='pass'
        timers={k:sum(row['momentum']['timers'][k] for row in r['rows']) for k in r['rows'][0]['momentum']['timers']}
        c={k:r[k] for k in ['case','grid_m','start_year','end_year','wall_seconds','cpu_seconds',
            'historical_baseline_seconds','historical_speedup','new_BP_seconds','old_BP_seconds',
            'new_AMG_builds','old_AMG_builds','new_tangent_assemblies','engineering_equivalence',
            'final_original_BP_residual','maximum_mass_relative','steps']}
        c.update(maximum_H_difference_m=max(d['H_max_m'] for d in r['comparison']),
                 maximum_H_RMS_m=max(d['H_RMS_m'] for d in r['comparison']),
                 maximum_velocity_relative_L2=max(d['velocity_relative_L2'] for d in r['comparison']),
                 maximum_volume_difference_m3=max(abs(d['volume_difference_m3']) for d in r['comparison']),
                 timers=timers,fixture=fixture,source=ref(root/'result.json'))
        cases.append(c)
    cases.sort(key=lambda c:(c['grid_m'],c['start_year']))
    summary=dict(status=out['status'],engineering_equivalence=all(c['engineering_equivalence'] for c in cases),
                 cases=cases,verified_accepted_states=verified,new_BP_solves=sum(c['steps'] for c in cases),
                 old_AMG_builds=sum(c['old_AMG_builds'] for c in cases),
                 new_AMG_builds=sum(c['new_AMG_builds'] for c in cases),
                 old_tangent_assemblies=sum(c['old_AMG_builds']+c['steps'] for c in cases),
                 new_tangent_assemblies=sum(c['new_tangent_assemblies'] for c in cases),
                 timing_note='Historical v22 timing intervals,not simultaneous paired benchmarks. New run sequential CPU8; old century grids ran concurrently CPU8 each. No old trajectory recomputed. JIT/fixture outside per-window time.',
                 source=ref(RUN/'result.json'),working_settings_changed=False)
    write_json(REPORT/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
    labels=[f"{c['grid_m']}米\n第{c['start_year']}→{c['end_year']}年" for c in cases];x=np.arange(4)
    a=axs[0].bar(x-.18,[c['historical_baseline_seconds'] for c in cases],.36,label='原运行记录',color='#afbfce')
    b=axs[0].bar(x+.18,[c['wall_seconds'] for c in cases],.36,label='优化版',color='#148d86')
    for bars in [a,b]:axs[0].bar_label(bars,fmt='%.1f',fontsize=10,padding=3)
    axs[0].set(xticks=x,xticklabels=labels,ylabel='推进一年耗时（秒）',title='四个一年窗口：实际耗时');axs[0].legend()
    bottom=np.zeros(4)
    names=[('prepare','几何准备'),('assembly','残差/矩阵组装'),('amg','AMG构建'),('cg','线性求解'),('linesearch','线搜索'),('checkpoint','BP内部存盘')]
    for key,label in names:
        val=np.array([c['timers'][key] for c in cases]);axs[1].bar(x,val,bottom=bottom,label=label);bottom+=val
    remainder=np.array([c['wall_seconds'] for c in cases])-bottom
    axs[1].bar(x,remainder,bottom=bottom,label='输运/外部存盘等',color='#d5dfe7')
    axs[1].set(xticks=x,xticklabels=labels,ylabel='秒',title='优化版分项计时');axs[1].legend(fontsize=8,ncol=2)
    for ax in axs:ax.spines[['top','right']].set_visible(False)
    fig.savefig(REPORT/'performance.png',dpi=150);plt.close(fig)
    table=''
    for c in cases:
        table+=f'<tr><td>{c["grid_m"]}米</td><td>{c["start_year"]}→{c["end_year"]}年</td><td>{c["historical_baseline_seconds"]:.2f}秒</td><td>{c["wall_seconds"]:.2f}秒</td><td>{c["historical_speedup"]:.2f}×</td><td>{c["maximum_H_difference_m"]:.2e}米</td><td>{c["maximum_velocity_relative_L2"]:.2e}</td></tr>'
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>BP求解加速：50米与100米实测</title>
<style>body{{margin:0;background:#eef3f7;color:#20394b;font:17px "Microsoft YaHei",sans-serif}}main{{max-width:1250px;margin:auto;padding:30px}}p,li{{line-height:1.85}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:13px;border-bottom:1px solid #dce5eb;text-align:left}}.lead{{padding:20px;background:#e0f3ee;border-left:5px solid #148d86}}img{{width:100%;background:white;border-radius:12px;margin:22px 0}}a{{color:#087f9e}}.note{{font-size:15px;color:#526779}}</style>
<main><h1>BP求解加速：50米与100米实测</h1><p class="lead">四组一年推进全部完成，数值对照通过。保持30天上限、原BP方程及收敛要求。与原记录相比，本次耗时约为原来的47%—66%。</p>
<table><thead><tr><th>网格</th><th>推进窗口</th><th>原记录</th><th>优化版</th><th>原/新耗时</th><th>最大冰厚差</th><th>最大速度相对L2差</th></tr></thead><tbody>{table}</tbody></table>
<p class="note">差异统计覆盖每个窗口的全部13步，并与相同网格、相同时间的原结果比较。耗时为历史记录对比：原50/100米两组并行运行，本次四组依次运行，机器负载不完全相同；编译和合并验证单独计时。</p>
<img src="performance.png" alt="四组实际推进耗时和优化版分项计时">
<h2>具体减少了哪些工作</h2><ul><li>水平几何缓存跨时间步复用；积分基函数、驱动力和材料系数在每次BP求解开始时更新，在内部迭代中复用。</li><li>预条件器构建由{summary['old_AMG_builds']}次降至{summary['new_AMG_builds']}次，仍对当前真实矩阵求解并检查残差。</li><li>切线矩阵组装由{summary['old_tangent_assemblies']}次降至{summary['new_tangent_assemblies']}次，省去已经收敛时的重复组装。</li><li>固定A只存一次，动态检查点改用未压缩NPZ以减少压缩开销；仍保存每步及每次牛顿迭代。此版不宣称文件数量或磁盘占用必然下降。</li></ul>
<h2>合并检查结果</h2><p>四组初始状态的新旧残差、驱动力、能量和切线矩阵比较差异均为0；湿/干节点几何处理一致。{verified}个接受状态重新读取并验证哈希、非负冰厚和有限速度；质量守恒及末期原始BP内核残差检查通过。检查点精确恢复和状态文件临时写入失败后继续工作均通过。</p>
<p>这次是第1→2年及第99→100年的短窗对照，使用既有百年状态，原模拟未重跑。固定床面、固定材料、零表面质量收支与原百年试算一致。季度更新BP和预测—校正时间推进尚未启用。</p>
<p><a href="summary.json">完整比较数据</a> · <a href="../bp-solver-acceleration-review-v1/index.md">调研与方案</a> · <a href="../century-50-100-v1/index.html">原百年结果</a></p></main></html>'''
    (REPORT/'index.html').write_text(text,encoding='utf-8')
    write_json(REPORT/'delivery.json',dict(files={p.name:digest(p) for p in REPORT.iterdir() if p.is_file()}))
    return summary


def finish():
    validate();summary=read(REPORT/'summary.json')
    for name,h in read(REPORT/'delivery.json')['files'].items():assert digest(REPORT/name)==h
    home=R/'index.html';old=home.read_text(encoding='utf-8');anchor='<h1>2387 冰川模型</h1>'
    assert anchor in old
    (REPORT/'previous-project-index.html').write_text(old,encoding='utf-8')
    link='<a class="current" href="reports/solver-acceleration-v1/index.html"><b>新增：BP求解加速实测</b><p>50米与100米早期、后期一年推进：耗时与数值对照。</p></a>'
    home.write_text(old.replace(anchor,anchor+link,1),encoding='utf-8')
    write_json(REPORT/'navigation.json',dict(home_sha256=digest(home),backup_sha256=digest(REPORT/'previous-project-index.html')))
    completion=dict(status=summary['status'],engineering_equivalence=summary['engineering_equivalence'],
                    result=ref(RUN/'result.json'),summary=ref(REPORT/'summary.json'),delivery=ref(REPORT/'delivery.json'),
                    verified_states=summary['verified_accepted_states'],accepted_count_effect=0)
    write_json(E/'completion.json',completion)
    facts='- User authorized50m/100m acceleration trials after literature review. Four retained checkpoints advanced1-2 and99-100;30daycap/CFL0.35,BP each step,samefixedbed/A/zeroSMB. New quadrature/horizontalcache,within-solveAMGreuse,convergedtangentavoidance,rawdynamiccheckpoints/staticAonce. Frozenv22preserved.\n'
    for c in summary['cases']:
        facts+=f'- {c["grid_m"]}m years{c["start_year"]}-{c["end_year"]}: newwall{c["wall_seconds"]:.9f}s,retainedbaseline{c["historical_baseline_seconds"]:.9f}s,ratio{c["historical_speedup"]:.6f};maxHdiff{c["maximum_H_difference_m"]:.12g}m,maxuL2{c["maximum_velocity_relative_L2"]:.12g},massrelative{c["maximum_mass_relative"]:.12g},trueBPfinalresidual{c["final_original_BP_residual"]:.12g}.\n'
    facts+=f'- All4operatorfixturesexact,restore/statusfailurechecks pass;{summary["verified_accepted_states"]}acceptedstates hashrestored.52newBPsolves;AMGbuilds102to52,tangents154to102. All preregisteredengineeringequivalencechecks pass. Historical wall ratios are not same-load pairedbenchmarks;JIT/preflightoutsidecasewall. Noautomatic100yearreplay,working-settingchange,climatecoupling,promotionorcountchange. Evidence `{ref(E/"completion.json")["path"]}`.'
    transition('completed',E/'completion.json',facts)


if __name__=='__main__':
    import sys
    if sys.argv[1]=='report':report()
    elif sys.argv[1]=='finish':finish()
