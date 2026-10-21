"""Summarize and seal v24 matrix benchmarks without changing the BP default."""
import json
import numpy as np
from scipy.sparse import csr_matrix
from .linear_solver_comparison_v1 import (RUN,REPORT,E,R,METHODS,read,arrays,digest,
                                         write_json,ref,hashes,transition,validate)


def report():
    validate();data=read(RUN/'result.json');assert data['implementation']==hashes()
    assert data['status']=='completed' and len(data['samples'])==48
    REPORT.mkdir(parents=True,exist_ok=False);groups=[];verified=0
    for m in sorted(data['matrices'],key=lambda x:(x['grid_m'],x['parent_year'])):
        root=RUN/m['case'];assert digest(root/'matrix.npz')==m['matrix']['sha256'].lower()
        z=arrays(root/'matrix.npz');K=csr_matrix((z['data'],z['indices'],z['indptr']),shape=tuple(z['shape']));r=z['r']
        reference=arrays(root/'lu-symmetric-order-r1/solution.npz')['delta'];methods={}
        for method in METHODS:
            samples=[]
            for rep in [1,2,3]:
                folder=root/f'{method}-r{rep}';s=read(folder/'result.json')
                assert digest(folder/'result.json')==read(folder/'seal.json')['result_sha256']
                assert digest(folder/'solution.npz')==s['solution_sha256']
                assert s['matrix_sha256']==digest(root/'matrix.npz') and s['implementation']==hashes()
                x=arrays(folder/'solution.npz')['delta'];ratio=float(np.linalg.norm(K@x+r)/np.linalg.norm(r))
                assert s['passed'] and ratio<=s['residual_target']
                samples.append({**s,'verified_residual_ratio':ratio,
                     'correction_relative_L2_vs_direct':float(np.linalg.norm(x-reference)/np.linalg.norm(reference))})
                verified+=1
            methods[method]=dict(median_seconds=float(np.median([s['total_seconds'] for s in samples])),
               min_seconds=min(s['total_seconds'] for s in samples),max_seconds=max(s['total_seconds'] for s in samples),
               median_build_seconds=float(np.median([s['build_seconds'] for s in samples])),
               median_solve_seconds=float(np.median([s['solve_seconds'] for s in samples])),
               median_setup_seconds=float(np.median([s['setup_seconds'] for s in samples])),
               maximum_residual=max(s['verified_residual_ratio'] for s in samples),
               maximum_peak_RSS_MiB=max(s['peak_sampled_rss_bytes'] for s in samples)/2**20,
               median_peak_increment_MiB=float(np.median([s['peak_increment_bytes'] for s in samples]))/2**20,
               factor_nnz=samples[0]['factor_nnz'],iterations=[s['iterations'] for s in samples],
               max_correction_difference=max(s['correction_relative_L2_vs_direct'] for s in samples))
        speed=methods['amg-production']['median_seconds']/methods['lu-symmetric-order']['median_seconds']
        strict=methods['amg-strict']['median_seconds']/methods['lu-symmetric-order']['median_seconds']
        groups.append(dict(matrix=m,methods=methods,direct_speedup_vs_production=speed,direct_speedup_vs_equal_accuracy=strict))
    summary=dict(status='completed',all_residual_checks_pass=True,verified_samples=verified,cases=groups,
                 default_solver_changed=False,glacier_years_advanced=0,
                 conclusion='SuperLU MMD_AT_PLUS_A faster onallfourfirstNewtonmatrices;defaultCOLAMDslowerthanAMG. NotCHOLMOD. Fullnonlinearsolveandtimeadvancementnotbenchmarked.',
                 source=ref(RUN/'result.json'),resources=read(E/'plan.frozen.json')['resources'])
    write_json(REPORT/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    labels=[f'{g["matrix"]["grid_m"]}米\n早期' if g['matrix']['parent_year']==1 else f'{g["matrix"]["grid_m"]}米\n后期' for g in groups]
    names=['AMG-CG当前容差','AMG-CG严格容差','SuperLU默认排序','SuperLU对称结构排序']
    colors=['#6388a5','#adc0d0','#d69474','#148d86'];fig,axs=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True);x=np.arange(4)
    for i,(method,name,color) in enumerate(zip(METHODS,names,colors)):
        values=np.array([g['methods'][method]['median_seconds']*1000 for g in groups])
        lo=values-np.array([g['methods'][method]['min_seconds']*1000 for g in groups]);hi=np.array([g['methods'][method]['max_seconds']*1000 for g in groups])-values
        axs[0].bar(x+(i-1.5)*.2,values,.2,label=name,color=color,yerr=np.vstack([lo,hi]),capsize=2)
        axs[1].bar(x+(i-1.5)*.2,[g['methods'][method]['maximum_peak_RSS_MiB'] for g in groups],.2,color=color,label=name)
    axs[0].set(title='一次线性求解总耗时（三次中位数）',ylabel='毫秒',xticks=x,xticklabels=labels)
    axs[1].set(title='测量期间进程RSS峰值（5毫秒采样）',ylabel='MiB，包含矩阵与运行时',xticks=x,xticklabels=labels)
    axs[0].legend(fontsize=9);axs[1].legend(fontsize=9)
    for ax in axs:ax.spines[['top','right']].set_visible(False)
    fig.savefig(REPORT/'comparison.png',dpi=150);plt.close(fig)
    table='';residual=''
    for g in groups:
        m=g['matrix'];a=g['methods'];phase='早期' if m['parent_year']==1 else '后期'
        table+=f'<tr><td>{m["grid_m"]}米·{phase}</td><td>{m["unknowns"]:,}</td>'+''.join(f'<td>{a[k]["median_seconds"]*1000:.1f}</td>' for k in METHODS)+f'<td>{g["direct_speedup_vs_production"]:.2f}×</td></tr>'
        residual+=f'<tr><td>{m["grid_m"]}米·{phase}</td>'+''.join(f'<td>{a[k]["maximum_residual"]:.2e}</td>' for k in METHODS)+'</tr>'
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>BP矩阵：AMG-CG与直接求解对照</title>
<style>body{{margin:0;background:#eef3f7;color:#20394b;font:17px "Microsoft YaHei",sans-serif}}main{{max-width:1400px;margin:auto;padding:30px}}p,li{{line-height:1.8}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #dce5eb}}.lead{{background:#e0f3ee;border-left:5px solid #148d86;padding:20px}}img{{width:100%;margin:22px 0;background:white;border-radius:10px}}.note{{font-size:15px;color:#526779}}a{{color:#087f9e}}</style>
<main><h1>BP矩阵：AMG-CG与直接求解对照</h1><p class="lead">四组矩阵、四种配置、各三次，共48次测量全部完成并通过残差检查。SuperLU采用MMD_AT_PLUS_A排序时，四组均比当前AMG-CG更快；默认COLAMD排序均更慢。</p>
<table><tr><th>矩阵</th><th>未知量</th><th>AMG当前容差/ms</th><th>AMG严格容差/ms</th><th>LU默认排序/ms</th><th>LU对称结构排序/ms</th><th>当前AMG/最佳LU</th></tr>{table}</table>
<p class="note">所有方法使用完全相同的float64矩阵、右端项和对角缩放，独立进程顺序执行，排序轮换。总耗时含缩放转换、预条件器构建或LU分解、求解及真实残差检查；不含共同的矩阵组装、进程启动、输入读取和结果保存。误差条为三次测量的最小值至最大值。BLAS单线程。</p>
<img src="comparison.png" alt="四组同矩阵求解的耗时和采样内存峰值">
<h2>精度检查</h2><p>使用原矩阵计算‖Kδ+r‖/‖r‖。AMG当前配置沿用0.1的生产线性残差上限，严格AMG与直接法均要求不超过10⁻⁸；直接法实际达到约10⁻¹⁵。全部48份解重新读取并校验哈希和残差。</p>
<table><tr><th>矩阵</th><th>AMG当前</th><th>AMG严格</th><th>LU默认</th><th>LU对称结构排序</th></tr>{residual}</table>
<h2>应如何采用</h2><p>最值得接入下一次完整BP对照的是SuperLU的MMD_AT_PLUS_A排序，尤其是50米网格。对称结构排序是LU的排序方式，此次没有测试CHOLMOD，也没有把LU称作Cholesky。内存使用高于AMG，但本次四组远低于3GiB预算。</p>
<p>这些是各窗口第一个牛顿线性系统的比较，每种方法都从头构建辅助结构。现有优化版会在同一次BP求解内复用AMG，直接法每次矩阵改变可能需要重新分解，因此单次线性求解的速度比不能直接作为整年推进的加速比。当前默认求解器未切换，没有推进新的冰川年份。</p>
<p><a href="summary.json">完整数据及内存、分解规模</a> · <a href="../solver-acceleration-v1/index.html">此前50/100米一年推进</a></p></main></html>'''
    (REPORT/'index.html').write_text(html,encoding='utf-8')
    write_json(REPORT/'delivery.json',dict(files={p.name:digest(p) for p in REPORT.iterdir() if p.is_file()}))


def finish():
    validate();s=read(REPORT/'summary.json');assert s['verified_samples']==48 and s['all_residual_checks_pass']
    for name,h in read(REPORT/'delivery.json')['files'].items():assert digest(REPORT/name)==h
    p=R/'index.html';old=p.read_text(encoding='utf-8');anchor='<h1>2387 冰川模型</h1>';assert anchor in old
    (REPORT/'previous-project-index.html').write_text(old,encoding='utf-8')
    link='<a class="current" href="reports/linear-solver-comparison-v1/index.html"><b>新增：求解器同矩阵实测</b><p>AMG-CG与SuperLU，两种网格、早晚状态的速度和内存对照。</p></a>'
    p.write_text(old.replace(anchor,anchor+link,1),encoding='utf-8')
    write_json(E/'completion.json',dict(status='completed',summary=ref(REPORT/'summary.json'),delivery=ref(REPORT/'delivery.json'),
                                      result=ref(RUN/'result.json'),verified_samples=48,glacier_years_advanced=0,accepted_count_effect=0))
    facts='- Userauthorizedmatchedlinearbenchmark;fourretainedv23firstNewtonmatrices50/100mearly/late. SameK/r/float64/diagonalscaling,oneBLASthread,freshserialprocesses,threepreregisteredreplicatespermethod;48total. AvailableSciPy1.13.1SuperLUtested,COLAMDandMMD_AT_PLUS_A;CHOLMOD/PARDISOnotinstalled,nopackageinstallation. AMGcurrentphysicalresidualtarget0.1 andstrict1e-8;directtarget1e-8.\n'
    for c in s['cases']:
        facts+=f'- {c["matrix"]["case"]}: n{c["matrix"]["unknowns"]}; medianseconds '+','.join(f'{k}={v["median_seconds"]:.9f}' for k,v in c['methods'].items())+f';bestLUvsproductionAMGspeedratio{c["direct_speedup_vs_production"]:.6f}.\n'
    facts+='- All48solutionshashreloadedandtruephysicalresidualverified. MMD_AT_PLUS_ALUfasteronallfourmatrices;defaultCOLAMDslower. PeakRSSsampled5ms,includesruntime/matrix. Totalmeasuredtimeincludesscaling/build/solve/residual,excludesassembly/imports/IO. Firstlinearizationonly;AMGreuseacrossNewtoniterationsnotmeasured. NofullBPsolve,timeadvance,defaultsolverchange,productionpromotionorcountchange. Evidence `'+ref(E/'completion.json')['path']+'`.'
    transition('completed',E/'completion.json',facts)


if __name__=='__main__':
    import sys
    if sys.argv[1]=='report':report()
    elif sys.argv[1]=='finish':finish()
