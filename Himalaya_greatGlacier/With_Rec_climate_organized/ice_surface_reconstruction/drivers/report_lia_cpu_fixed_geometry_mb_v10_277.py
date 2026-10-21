"""Render existing offline results only; no new numerical model evaluation."""
from pathlib import Path
import html
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from .diagnose_lia_cpu_fixed_geometry_mb_v10_277 import ROOT,OUT,read,write,ref,verify,now

s=read(OUT/'summary.json')
for p in s['members']:verify(p)
rows=read(OUT/'rescored_cells_1393.json')
pop=read(OUT/'population_191_v2.json')
diagnostic=read(OUT/'fixed_geometry_glaciers_191.json')
figdir=OUT/'figures'
if figdir.exists():raise RuntimeError('immutable report figures already exist')
figdir.mkdir()
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
fig,axes=plt.subplots(1,2,figsize=(12.8,5.4),layout='constrained')
cases=['A','B','C_first30','C_second30','C_final30']
labels=['A 原spin-up\n气候＋原偏差','B 同一气候\n去掉spin-up偏差',
        'C 历史气候\n1470–1499','C 历史气候\n1500–1529','C 历史气候\n1970–1999']
values=[s['equal_glacier_weight_median_of_within_glacier_medians'][c]/1000 for c in cases]
axes[0].bar(range(5),values,color=['#6e8796','#bd7751','#376c87','#60879c','#9baebb'])
for i,v in enumerate(values):axes[0].text(i,v+.013,f'{v:.3f}',ha='center',fontsize=10)
axes[0].set_xticks(range(5),labels,fontsize=8.5)
axes[0].set_ylim(0,max(values)*1.20)
axes[0].set_ylabel('固定初始表面净收支（m 水当量 / 年）')
axes[0].set_title('同一初态，改变气候设置后的收支')
effects=[s['equal_glacier_weight_median_effects']['withdrawal_B_minus_A_kg_m2_yr']/1000,
         s['equal_glacier_weight_median_effects']['first30_climate_C_minus_B_kg_m2_yr']/1000]
axes[1].barh(['去掉spin-up温偏\n配对差 B−A','改用前30年历史气候\n配对差 C−B'],effects,color=['#bd7751','#376c87'])
for i,v in enumerate(effects):axes[1].text(v+.01,i,f'+{v:.3f}',va='center')
axes[1].set_xlim(0,max(effects)*1.28)
axes[1].set_xlabel('固定表面收支变化（m 水当量 / 年）')
axes[1].set_title('输入切换的影响分别计算\n各项中位数不能直接相加为总效应')
fig.suptitle('1393个初态的离线诊断：每条冰川等权\n先取每条冰川内部中位数，再对173条冰川取中位数',fontsize=13)
fig.savefig(figdir/'fixed_geometry_balance.png',dpi=140);plt.close(fig)
passed=[p for p in pop if p['provisional_curve_pass_v2_count']>0]
norm=Normalize(-1,1);cmap=plt.get_cmap('coolwarm')
fig,axes=plt.subplots(3,3,figsize=(15,10.5),layout='constrained')
for ax,g in zip(axes.flat,passed):
    group=[r for r in rows if r['lia_unique_id']==g['lia_unique_id']]
    for r in group:
        a=pd.read_csv(ROOT/r['source_annual']);yes=r['provisional_curve_pass_v2']
        ax.plot(a.year,a.length_m/1000,color=cmap(norm(r['delta_t_eq_degC'])) if yes else '#bcbcbc',
                alpha=.9 if yes else .5,lw=1.1 if yes else .7,zorder=3 if yes else 1)
    ax.axhline(g['lia_reference_m']/1000,color='#333333',ls='--',lw=.8)
    ax.errorbar([2000],[g['rgi_reference_m']/1000],yerr=[g['rgi_reference_m']/10000],fmt='*',color='#176c42',ms=9,capsize=3)
    ax.set_title(f"冰川 {g['lia_unique_id']}：{g['provisional_curve_pass_v2_count']} 个暂通过 / {len(group)} 个初态",fontsize=10.5)
    ax.set_xlim(1470,2010);ax.set_ylabel('长度 (km)');ax.set_xlabel('年份');ax.grid(axis='y',alpha=.13)
fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=axes,label='spin-up ΔT_eq (℃)，不带入 forward',shrink=.65,pad=.01)
fig.suptitle('CPU 方向解释更新后：77个组合，9条冰川暂通过\n彩色为新版暂通过，灰色为同冰川其他初态；虚线为LIA参考，绿星为RGI末端及±10%',fontsize=13)
fig.savefig(figdir/'provisional_9_glaciers.png',dpi=135);plt.close(fig)

def fmt(v,n=3):return '' if v is None else f'{v:.{n}f}'
candidate_rows=''.join('<tr><td>'+str(g['lia_unique_id'])+'</td><td>'+str(g['provisional_curve_pass_v2_count'])+'</td><td>'+', '.join(f'{x:+.1f}' for x in g['provisional_delta_t_eq_v2_degC'])+'</td></tr>' for g in passed)
diagmap={r['lia_unique_id']:r for r in diagnostic}
pop_rows=[]
for g in pop:
    d=diagmap[g['lia_unique_id']];means=d['median_fixed_geometry_net_mb'];eff=d['median_effects']
    vals=[str(g['lia_unique_id']),str(g['stable_parent_count']),str(g['provisional_curve_pass_v2_count'])]
    vals+=[fmt(means[c]/1000 if means[c] is not None else None) for c in ['A','B','C_first30','C_final30']]
    vals+=[fmt(eff[k]/1000 if eff[k] is not None else None) for k in ['withdrawal_B_minus_A_kg_m2_yr','first30_climate_C_minus_B_kg_m2_yr']]
    vals+=['无稳态初态，保留后续处理' if g['stable_parent_count']==0 else '诊断完成']
    pop_rows.append(f'<tr data-id="{g["lia_unique_id"]}">'+''.join('<td>'+html.escape(v)+'</td>' for v in vals)+'</tr>')
body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CPU离线评价与初态收支</title>
<style>body{{max-width:1160px;margin:30px auto;padding:0 22px;font:16px/1.7 system-ui;color:#283642;background:#fafbfc}}h1{{font-size:28px}}h2{{font-size:21px;margin-top:32px}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;background:white;margin:16px 0}}th,td{{padding:9px 12px;border-bottom:1px solid #dce2e7;text-align:right}}th{{background:#e7edf1}}td:first-child,th:first-child{{text-align:left}}a{{color:#286283}}input{{padding:9px;font:inherit}}.scroll{{overflow:auto}}.note{{color:#536271}}</style>
<h1>CPU 离线评价与初态收支</h1>
<p>2026-09-07。1393个初态全部完成诊断，覆盖173条冰川；另18条无稳态初态者保留在191条名单。没有重跑spin-up或forward，没有新物理参数搜索。</p>
<h2>1. 方向解释更新后的候选</h2>
<p>1470年保留为数值起点，不作为观测长度或净变化方向锚点。只将旧净方向硬门改为诊断；峰值10%、末端10%、振幅、滞后、持续阶段和异常检查全部不变。旧32组合/8条结果保留，新版为<strong>77个组合、9条冰川暂通过</strong>。新增45个组合，其中843为新增覆盖冰川。这是经批准的事后评价政策修订，不是原始预注册结果；尚未独立复现，正式接受数不增加。</p>
<table><thead><tr><th>冰川ID</th><th>新版暂通过组合数</th><th>spin-up ΔT_eq（℃）</th></tr></thead><tbody>{candidate_rows}</tbody></table>
<a href="figures/provisional_9_glaciers.png"><img src="figures/provisional_9_glaciers.png" alt="九条冰川候选曲线"></a>
<h2>2. 固定同一形状后的积雪与融化收支</h2>
<p>A是原1441–1470循环气候加基准温偏b及spin-up偏差s；B只去掉s；C改为历史气候、仍使用b且q=0。三个情况始终使用同一份年900初态的表面高程与已有冰覆盖面积，没有让几何随时间变化。</p>
<a href="figures/fixed_geometry_balance.png"><img src="figures/fixed_geometry_balance.png" alt="三种气候设置下固定几何收支"></a>
<p>按每条冰川等权统计，去掉spin-up偏差使固定表面净收支的配对变化中位数增加<strong>{effects[0]:.3f} m水当量/年</strong>；再换为前30年历史气候，增加<strong>{effects[1]:.3f} m水当量/年</strong>。因此，当前初态上的增冰倾向对偏差撤除很敏感。各冰川平均后的历史前30年温度相对原循环气候加s的差值中位数为{s['equal_glacier_weight_median_effects']['first30_ref_temp_C_minus_A_degC']:.2f}℃。</p>
<p>实际历史曲线中净增长的1206个组合里，1205个在初始几何上的前30年历史气候收支为正；1192个在去掉s后收支增加。这是关联证据，不是对530年演变的严格因果分解。</p>
<p class="note">收支数值是固定初始表面上的模型计算，不是观测，也不是新的冰厚增长预测。单位m水当量/年不能直接当作冰厚m/年。潜在融化未加有限冰量限制；没有计算冰流输送、形状反馈或新增冰覆盖面积。原周期稳定的冰川在某个冻结年末几何上不必恰有零平均收支。各情景中位数与配对差中位数不可直接相加。</p>
<h2>3. 全部191条冰川</h2><p>以下净收支及差值均为各冰川内部全部稳态初态的中位数，单位m水当量/年；空白表示没有稳态初态，不是零。</p><input id="filter" placeholder="按冰川ID筛选"><div class="scroll"><table id="population"><thead><tr><th>冰川ID</th><th>初态数</th><th>暂通过数</th><th>A</th><th>B</th><th>C前30年</th><th>C末30年</th><th>B−A</th><th>C前30年−B</th><th>状态</th></tr></thead><tbody>{''.join(pop_rows)}</tbody></table></div>
<h2>4. 结果的边界与后续</h2><p>保留77个多解候选和全部剩余冰川。当前证据支持优先审查稳态初态与历史强迫的衔接，再确定后续melt_f等粗参数矩阵；不能直接断言重建气候有错，也不能用调参自动抵消初始化偏差。未启动任何后续动力学实验。</p>
<p><a href="rescored_cells_1393.json">1393个组合新版完整指标</a>　<a href="fixed_geometry_glaciers_191.json">191条冰川收支汇总</a>　<a href="summary.json">正式诊断摘要</a></p>
<script>document.getElementById('filter').addEventListener('input',e=>{{for(const r of document.querySelectorAll('#population tbody tr'))r.hidden=!r.dataset.id.includes(e.target.value.trim())}});</script></html>'''
with (OUT/'index.html').open('x',encoding='utf-8') as f:f.write(body)
write(OUT/'report_manifest.json',{'created_at':now(),'source_summary':ref(OUT/'summary.json'),
    'figures':[ref(p) for p in sorted(figdir.glob('*.png'))],'report':ref(OUT/'index.html'),
    'population_count':len(pop),'provisional_curves':77,'provisional_glaciers':9,'new_model_evaluations':0})
print(json.dumps({'report':ref(OUT/'index.html'),'figures':2,'complete':True}))
