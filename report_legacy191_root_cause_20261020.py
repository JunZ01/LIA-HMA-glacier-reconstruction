from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('E:/3spyderProject/1_LIA_himalaya_oggm')
OUT=Path('H:/ToYangxin/result_yx/legacy191_root_cause_20261020_v1')
results=json.loads((OUT/'interpolation_results.json').read_text(encoding='utf-8'))
summary=json.loads((OUT/'diagnostic_summary.json').read_text(encoding='utf-8'))
assert summary['all_converged'] and summary['all_sources_unchanged'] and len(results)==6
d=pd.DataFrame(results)
pivot=d.pivot(index=['lia_unique_id','region_zh'],columns='boundary_exclusion_m',values='interpolation_volume_bias_percent').reset_index().rename(columns={0:'bias_keep_all_interior_pct',100:'bias_100m_exclusion_pct'})
pivot['extra_100m_effect_percentage_points']=pivot.bias_100m_exclusion_pct-pivot.bias_keep_all_interior_pct
assert d.constraint_max_abs_error_m.max()==0
assert d.remap_volume_difference_percent.abs().max()<.02
pivot.to_csv(OUT/'root_cause_evidence_by_region.csv',index=False,encoding='utf-8-sig')
codefiles=['drivers/modern_tau_calibration.py','drivers/lia_3d_zero_boundary_pilot.py','drivers/lia_3d_system_grid_pilot_v1.py','core/solver.py']
code_receipts={}
for rel in codefiles:
 p=ROOT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction'/rel
 b=p.read_bytes();code_receipts[str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(10.5,6.3))
fig.subplots_adjust(left=.10,right=.97,top=.78,bottom=.22)
x=np.arange(len(pivot));width=.25
groups=[(np.full(len(pivot),100.),'原现代参考冰量','#b9cbd3'),
 (100+pivot.bias_keep_all_interior_pct.to_numpy(),'二维扩展：保留全部内部采样点','#27849b'),
 (100+pivot.bias_100m_exclusion_pct.to_numpy(),'二维扩展：排除边界100m内采样点','#ca7846')]
for i,(values,label,color) in enumerate(groups):
 bars=ax.bar(x+(i-1)*width,values,width,color=color,label=label)
 ax.bar_label(bars,labels=[f'{v:.1f}' for v in values],padding=4,fontsize=10)
ax.set_xticks(x,[f'{r.region_zh}\n系统{r.lia_unique_id}' for r in pivot.itertuples()]);ax.set_ylim(0,118)
ax.set_ylabel('相对原现代参考冰量（%）');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.19),ncols=1,fontsize=10)
fig.suptitle('现代状态未变，旧二维扩展仍低估总冰量',x=.06,y=.97,ha='left',fontsize=18,fontweight='bold')
fig.text(.06,.88,'同一现代范围、同一参考冰厚；所有流线采样约束误差均为0。没有使用LIA或气候变化。',fontsize=10.5,color='#526e79')
fig.savefig(OUT/'现代零变化对照_体积偏差.png',dpi=220,bbox_inches='tight');plt.close(fig)

case=3131
with np.load(OUT/f'modern_operator_control_{case}.npz') as arrays:
 mask=arrays['modern_mask'];ref=arrays['reference_H_m'];test=arrays['harmonic_H_exclusion100m']
ys,xs=np.where(mask);sl=(slice(max(0,ys.min()-4),ys.max()+5),slice(max(0,xs.min()-4),xs.max()+5))
ref=np.where(mask,ref,np.nan)[sl];test=np.where(mask,test,np.nan)[sl];diff=test-ref
fig,axs=plt.subplots(1,3,figsize=(12.5,5.7))
fig.subplots_adjust(left=.04,right=.98,top=.78,bottom=.12,wspace=.28)
vmax=max(float(np.nanmax(ref)),float(np.nanmax(test)));dv=max(abs(float(np.nanmin(diff))),abs(float(np.nanmax(diff))))
for ax,arr,title in zip(axs,[ref,test,diff],['现代参考冰厚','旧二维扩展（100m规则）','扩展减参考']):
 im=ax.imshow(arr,cmap='RdBu_r' if ax is axs[2] else 'viridis',vmin=-dv if ax is axs[2] else 0,vmax=dv if ax is axs[2] else vmax)
 ax.set_title(title,fontsize=12);ax.set_xticks([]);ax.set_yticks([])
 fig.colorbar(im,ax=ax,fraction=.047,pad=.04,label='m')
fig.suptitle('系统3131：采样点一致不保证整个冰川的冰量一致',x=.04,y=.96,ha='left',fontsize=17,fontweight='bold')
fig.text(.04,.87,'冻结现代参考数据上的插值算子诊断；两幅冰厚使用同一色标。坐标为相同保存网格。',fontsize=10,color='#526e79')
fig.savefig(OUT/'系统3131_二维冰厚对照.png',dpi=220,bbox_inches='tight');plt.close(fig)

show=pivot[['region_zh','lia_unique_id','bias_keep_all_interior_pct','bias_100m_exclusion_pct','extra_100m_effect_percentage_points']].rename(columns={
 'region_zh':'区域','lia_unique_id':'系统ID','bias_keep_all_interior_pct':'仅二维扩展的体积偏差%','bias_100m_exclusion_pct':'加入100m排除后的偏差%','extra_100m_effect_percentage_points':'100m排除新增影响（百分点）'})
table=show.to_html(index=False,border=0,float_format=lambda v:f'{v:.3f}')
doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>喜马拉雅旧重建问题定位</title><style>body{{background:#f3f6f7;color:#243e4c;font:16px/1.8 'Microsoft YaHei',sans-serif;margin:0}}main{{max-width:1120px;margin:auto;background:white;padding:32px}}h1{{font-size:29px}}h2{{font-size:22px;margin-top:32px}}img{{width:100%}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #dce5e8;text-align:right}}th{{background:#edf3f5}}.note{{background:#fff2e4;padding:15px 20px}}a{{color:#15788d}}</style><main>
<p>2026-10-20 · 旧191套异常的定向诊断</p><h1>已定位：流线厚度标定没有控制二维总体积</h1>
<p><strong>旧流程先用一维反演流线厚度标定应力参数，再通过零边界调和扩展生成二维冰厚；标定目标没有覆盖扩展后的总体积。最后比较时，LIA用该二维重建体积，现代却用另一套反演分配体积。两端的方法偏差没有相消。</strong></p>
<p>代码中的二维扩展函数明确规定边界冰厚为0、流线采样格点固定、其余内部格点满足离散Laplace方程。这个算子可数值收敛并精确满足采样点，却不保证保持现代参考总体积。缺失的是从一维标定到二维体积输出之间的一致性约束，而非单纯把τ参数传错。</p>
<h2>现代零变化对照</h2><p>预先按各喜马拉雅区域单成员系统的现代面积中位位置选出3131、2906、3632，没有按损失正负挑选。将已有现代参考冰厚重投影至原保存网格，保持现代范围不变，沿原现代流线抽取该参考场的准确冰厚，以原二维扩展函数计算。没有进行PP应力求解、τ重新率定、气候变化、LIA重建或冰流模拟；这是一项隔离横向扩展影响的离线诊断。</p>
{table}<img src="现代零变化对照_体积偏差.png" alt="现代参考冰量100与二维扩展后64至72的对照">
<p>六次诊断全部收敛，流线采样格点最大冰厚误差均为0。原现代参考到目标网格的体积映射差仅0.0022%至0.0106%，远小于二维扩展产生的28%至36%偏差。100m边界排除额外改变约0.004至0.100个百分点，在这三个对照中不是主要来源。因此，零边界二维扩展本身已足以让同一个现代参考状态失去大量体积。</p>
<img src="系统3131_二维冰厚对照.png" alt="现代参考和二维扩展的冰厚场与差值">
<h2>另一项已查实的不一致</h2><p>τ标定使用原一维反演厚度及其流线床面；LIA二维计算则从高程分配后的冰厚网格构床。三个样本中，同一现代流线处的一维厚度与网格分配厚度的RMSE约31至55m，说明名称都为F19并不意味着局地几何一致。这一差异的方向并不在三个样本间一致，因此不能单独作为全部旧LIA偏薄的解释。</p>
<h2>已排除与仍待量化的部分</h2><p>既有全191审计已证实体积数组积分与台账一致，τ传递值与标定表一致，共同床的重投影一致，不能将异常归为单位换算、加总或τ漏传。此前V/A的变化还包含面积分母缩小效应：它解释平均厚度为何能在体积减少时增大，但不能解释旧191现代来源体积总量偏大的模型问题。</p>
<p class="note">这三个控制明确证实了旧横向扩展的体积偏差，以及现代二维体积闭合缺失这一方法问题。它们不是旧PP完整现代端点重放，也没有量化全部191套的总偏差；不能把28%至36%直接当作统一修正系数。现阶段已定位一个能产生假变化信号的具体结构性问题，未证明它是所有单条异常的唯一原因。</p>
<h2>修复应落在哪里</h2><p>应使现代与LIA使用一致床面定义和同一二维构建流程，并在现代参照状态检查扩展后的体积与空间冰厚分布。若采用既有654流程的现代二维体积标定，应先明确沿用其方法与输入合同，再转移到LIA并保留独立质量比较。仅调整τ以贴合流线、强制LIA更厚、改符号或给191套统一乘系数，都没有解决本次查实的问题。</p>
<p>旧191套仍保留为来源诊断；当前结果不是已修复后的829套重建，也没有改变生产模型、原数据、当前控制器或任何其他任务。</p>
<p><a href="modern_interpolation_controls.csv">六次对照数据</a> · <a href="root_cause_evidence_by_region.csv">区域诊断摘要</a> · <a href="profile_vs_distributed_thickness.csv">流线与网格厚度差异</a> · <a href="protocol.json">预先固定的诊断设计</a> · <a href="diagnostic_summary.json">运行与源文件核验</a></p></main></html>'''
(OUT/'report.html').write_text(doc,encoding='utf-8')
qc={'status':'pass','completed_utc':datetime.now(timezone.utc).isoformat(),'isolated_failure':'modern-reference volume is not preserved by legacy lateral harmonic extension, despite exact sampled H',
 'controls':6,'regions':3,'constraint_max_abs_error_m':float(d.constraint_max_abs_error_m.max()),
 'maximum_absolute_remap_volume_error_percent':float(d.remap_volume_difference_percent.abs().max()),
 'interpolation_bias_percent_range':[float(d.interpolation_volume_bias_percent.min()),float(d.interpolation_volume_bias_percent.max())],
 'additional100m_effect_pp_range':[float(pivot.extra_100m_effect_percentage_points.min()),float(pivot.extra_100m_effect_percentage_points.max())],
 'source_code_receipts':code_receipts,'sources_unchanged':summary['all_sources_unchanged'],
 'no_dynamic_or_LIA_reconstruction_runs':True,'not_full_legacy_modern_PP_endpoint_replay':True,'not_a_full191_bias_correction':True,'visual_review':'pending'}
(OUT/'root_cause_QC.json').write_text(json.dumps(qc,ensure_ascii=False,indent=2),encoding='utf-8')
shutil.copy2(__file__,OUT/'reproduce_report.py')
print(show.to_string(index=False))
