from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil, html
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path('H:/ToYangxin/result_yx')
OUT=BASE/'mean_thickness829_20261020_v1'
OUT.mkdir(parents=True,exist_ok=True)
SOURCES={
 'population':BASE/'lia_expanded_gt10_20261017_v1/data/system_results829.csv',
 'frozen_ids':Path('E:/3spyderProject/1_LIA_himalaya_oggm/validation_data/results/paleo_surface_validation_v0/lia_oggm_independent_axes829_20261019_v1/systems_manifest.json'),
 'previous654':BASE/'hma_harmonized_analysis_20261012_v1/selected654_analysis/selected654_merged_analysis.csv',
 'legacy191':BASE/'hma_harmonized_analysis_20261012_v1/legacy_volume_comparability/legacy191_source_pairing_diagnostic.csv',
 'legacy_qc':BASE/'hma_harmonized_analysis_20261012_v1/legacy_volume_comparability/QC.json',
 'legacy_interpretation':BASE/'hma_harmonized_analysis_20261012_v1/legacy_volume_comparability/summary.json',
 'legacy_members':BASE/'hma_harmonized_analysis_20261012_v1/legacy_volume_comparability/modern_member_array_audit.csv'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
initial_hash={k:sha(p) for k,p in SOURCES.items()}
s=pd.read_csv(SOURCES['population'])
c=pd.read_csv(SOURCES['previous654']).set_index('record_id')
a=pd.read_csv(SOURCES['legacy191']).set_index('lia_unique_id')
members=pd.read_csv(SOURCES['legacy_members'])
frozen=json.loads(SOURCES['frozen_ids'].read_text(encoding='utf-8'))
assert len(s)==829 and s.record_id.is_unique and set(s.record_id)=={r['record_id'] for r in frozen}
assert s.source_cohort.value_counts().to_dict()=={'Previous654':638,'Himalaya191':191}
assert json.loads(SOURCES['legacy_qc'].read_text(encoding='utf-8'))['status']=='pass'
ZH={'Hindu Kush':'兴都库什','Karakoram':'喀喇昆仑','Pamir':'帕米尔','Western Tien Shan':'本地天山区（原名西天山）','Western Kunlun Shan':'西昆仑','Nyainqentanglha':'念青唐古拉','Tibetan Interior Mountains':'青藏高原内部山地','West Himalaya':'西喜马拉雅','Central Himalaya':'中喜马拉雅','East Himalaya':'东喜马拉雅'}
ORDER=list(ZH)
rows=[]
for r in s.itertuples():
 x={'record_id':r.record_id,'lia_unique_id':r.lia_unique_id,'cohort':r.source_cohort,
    'region':r.mountain,'region_zh':ZH[r.mountain],
    'population_LIA_area_km2':r.LIA_area_km2,'population_modern_area_km2':r.modern_area_km2}
 if r.source_cohort=='Previous654':
  q=c.loc[r.record_id]
  assert abs(q.lia_area_geodesic_km2-r.LIA_area_km2)<1e-10 and abs(q.modern_area_geodesic_km2-r.modern_area_km2)<1e-10
  x.update(LIA_area_km2=q.lia_area_geodesic_km2,modern_area_km2=q.modern_area_geodesic_km2,
     LIA_volume_km3=q.lia_volume_deduplicated_km3,modern_volume_km3=q.modern_volume_km3,
     interpretation='638套原配对重建的平均冰厚统计',source_area_and_volume='frozen selected654 paired output; overlap-corrected LIA volume',
     source_modern_member_count=np.nan,legacy_model_comparison_only=False)
 else:
  q=a.loc[r.lia_unique_id]
  assert q.has_static_lia191 and not q.strictly_harmonized_with_selected654
  x.update(LIA_area_km2=q.lia_outline_area_manifest_km2,modern_area_km2=q.modern_outline_union_area_manifest_km2,
     LIA_volume_km3=q.lia_saved_array_volume_km3,modern_volume_km3=q.modern_reference_native_sum_km3,
     interpretation='旧191冰面与现代反演参考的来源比较；仅诊断，不作实际增厚结论',
     source_area_and_volume='audited legacy191 source area ledger and native inversion member volumes; not 829 truncated member-list area',
     source_modern_member_count=int(q.modern_member_count),legacy_model_comparison_only=True)
 rows.append(x)
d=pd.DataFrame(rows)
assert np.isfinite(d[['LIA_area_km2','modern_area_km2','LIA_volume_km3','modern_volume_km3']]).all().all()
assert (d[['LIA_area_km2','modern_area_km2','LIA_volume_km3','modern_volume_km3']]>0).all().all()
d['LIA_mean_thickness_m']=1000*d.LIA_volume_km3/d.LIA_area_km2
d['modern_mean_thickness_m']=1000*d.modern_volume_km3/d.modern_area_km2
d['mean_thickness_change_modern_minus_LIA_m']=d.modern_mean_thickness_m-d.LIA_mean_thickness_m
d['mean_thickness_change_percent']=100*d.mean_thickness_change_modern_minus_LIA_m/d.LIA_mean_thickness_m
d['equivalent_change_on_LIA_area_m']=1000*(d.modern_volume_km3-d.LIA_volume_km3)/d.LIA_area_km2
d['equivalent_change_on_mean_endpoint_area_m']=1000*(d.modern_volume_km3-d.LIA_volume_km3)/((d.LIA_area_km2+d.modern_area_km2)/2)
d['source_modern_area_minus_population_area_km2']=d.modern_area_km2-d.population_modern_area_km2

def stats(g,region,zh):
 al=g.LIA_area_km2.sum();am=g.modern_area_km2.sum();vl=g.LIA_volume_km3.sum();vm=g.modern_volume_km3.sum()
 hl=1000*vl/al;hm=1000*vm/am
 return {'region':region,'region_zh':zh,'systems':len(g),'cohort':g.cohort.iloc[0],
   'LIA_area_km2':al,'modern_area_km2':am,'LIA_volume_km3':vl,'modern_volume_km3':vm,
   'LIA_area_weighted_mean_thickness_m':hl,'modern_area_weighted_mean_thickness_m':hm,
   'change_modern_minus_LIA_m':hm-hl,'change_percent':100*(hm-hl)/hl,
   'equivalent_change_on_LIA_area_m':1000*(vm-vl)/al,
   'equivalent_change_on_mean_endpoint_area_m':1000*(vm-vl)/((al+am)/2),
   'individual_change_arithmetic_mean_m':g.mean_thickness_change_modern_minus_LIA_m.mean(),
   'individual_change_median_m':g.mean_thickness_change_modern_minus_LIA_m.median(),
   'individual_change_q25_m':g.mean_thickness_change_modern_minus_LIA_m.quantile(.25),
   'individual_change_q75_m':g.mean_thickness_change_modern_minus_LIA_m.quantile(.75),
   'individual_mean_thickness_decrease_count':int((g.mean_thickness_change_modern_minus_LIA_m<0).sum()),
   'individual_mean_thickness_increase_count':int((g.mean_thickness_change_modern_minus_LIA_m>0).sum()),
   'legacy_model_comparison_only':bool(g.legacy_model_comparison_only.any()),'interpretation':g.interpretation.iloc[0]}

regional=pd.DataFrame([stats(d[d.region==m],m,ZH[m]) for m in ORDER])
cohort=pd.DataFrame([stats(g,k,('原配对重建638套' if k=='Previous654' else '旧喜马拉雅191套，仅来源比较')) for k,g in d.groupby('cohort',sort=False)])
primary=cohort[cohort.cohort=='Previous654'].iloc[0]
checks={'all829_unique_and_match_frozen_population':len(d)==829 and d.record_id.is_unique,
 '638_plus191':d.cohort.value_counts().to_dict()=={'Previous654':638,'Himalaya191':191},
 'ten_mountain_regions':len(regional)==10 and regional.systems.sum()==829,
 'no_models_or_rasters_recomputed':True,'no_source_values_clipped':True,'no_pooled_829_physical_estimate':True}
for key,g in d.groupby('cohort'):
 rr=regional[regional.cohort==key]; cr=cohort[cohort.cohort==key].iloc[0]
 for field in ['LIA_area_km2','modern_area_km2','LIA_volume_km3','modern_volume_km3']:
  checks[key+'_'+field+'_regional_sum_closes']=bool(np.isclose(rr[field].sum(),cr[field],atol=1e-9,rtol=0))
 for prefix in ['LIA','modern']:
  direct=np.average(g[prefix+'_mean_thickness_m'],weights=g[prefix+'_area_km2'])
  checks[key+'_'+prefix+'_independent_weighted_average']=bool(np.isclose(direct,cr[prefix+'_area_weighted_mean_thickness_m'],atol=1e-10,rtol=0))
# Confirm the native legacy volume ledger against member arrays without re-integrating any raster.
native=members.groupby('lia_unique_id').native_volume_km3.sum()
checks['legacy_native_member_sums_match_saved_audit']=bool(np.allclose(native.loc[a.index],a.modern_reference_native_sum_km3,atol=1e-10,rtol=0))
assert all(checks.values())

CN={'region_zh':'山脉区域','systems':'冰川系统数','LIA_area_weighted_mean_thickness_m':'小冰期平均冰厚_m',
 'modern_area_weighted_mean_thickness_m':'现代平均冰厚_m','change_modern_minus_LIA_m':'平均冰厚变化_现代减小冰期_m',
 'equivalent_change_on_LIA_area_m':'原小冰期面积上等效厚度变化_m',
 'equivalent_change_on_mean_endpoint_area_m':'两期平均面积上等效厚度变化_m',
 'individual_change_arithmetic_mean_m':'逐冰川厚度变化的算术平均_m','individual_change_median_m':'逐冰川厚度变化的中位数_m','interpretation':'结果含义'}
cols=list(CN)
regional.to_csv(OUT/'regional_mean_thickness829.csv',index=False,encoding='utf-8-sig')
regional[cols].rename(columns=CN).to_csv(OUT/'平均冰厚变化_区域829.csv',index=False,encoding='utf-8-sig')
d.to_csv(OUT/'mean_thickness_per_system829.csv',index=False,encoding='utf-8-sig')
cohort.to_csv(OUT/'cohort_summaries_separate.csv',index=False,encoding='utf-8-sig')
changes=d[d.legacy_model_comparison_only & (d.source_modern_area_minus_population_area_km2.abs()>1e-4)]
changes[['record_id','lia_unique_id','source_modern_member_count','population_modern_area_km2','modern_area_km2','source_modern_area_minus_population_area_km2']].to_csv(OUT/'legacy7_area_membership_difference.csv',index=False,encoding='utf-8-sig')

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(13.2,6.8),gridspec_kw={'width_ratios':[1.35,1]})
fig.subplots_adjust(left=.22,right=.97,top=.77,bottom=.16,wspace=.65)
fig.suptitle('829套冰川系统：按山脉统计两期平均冰厚之差',x=.04,y=.98,ha='left',fontsize=18,fontweight='bold')
fig.text(.04,.91,'平均冰厚＝总体积÷总面积；变化＝现代−小冰期，单位m。两套来源分开呈现。',fontsize=11,color='#506b77')
for ax,legacy in zip(axs,[False,True]):
 g=regional[regional.legacy_model_comparison_only==legacy].copy();y=np.arange(len(g))
 values=g.change_modern_minus_LIA_m.to_numpy()
 ax.barh(y,values,color='#9ba4aa' if legacy else '#287f99',height=.58)
 ax.set_yticks(y,[f'{r.region_zh}（{r.systems}）' for r in g.itertuples()]);ax.invert_yaxis();ax.axvline(0,color='#52626a',lw=1)
 ax.set_title('旧喜马拉雅191套：仅来源比较' if legacy else '原配对重建638套',fontsize=12,pad=16)
 ax.set_xlabel('平均冰厚变化（m）');ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
 ax.set_xlim((-12,1) if not legacy else (-1,29))
 for yi,v in zip(y,values):ax.text(v+(.35 if v>=0 else -.2),yi,f'{v:+.2f}',va='center',ha='left' if v>=0 else 'right',fontsize=10)
fig.text(.04,.045,'喜马拉雅的正值受旧重建与现代参考方法差异影响，不解释为实际增厚；不合成全829套的统一物理均值。',fontsize=10,color='#675747')
fig.savefig(OUT/'区域平均冰厚变化829.png',dpi=220,bbox_inches='tight');plt.close(fig)

def table(frame):return frame.to_html(index=False,border=0,float_format=lambda v:f'{v:,.2f}')
show=regional[['region_zh','systems','LIA_area_weighted_mean_thickness_m','modern_area_weighted_mean_thickness_m','change_modern_minus_LIA_m']].rename(columns={k:v.replace('_m','（m）') for k,v in CN.items()})
doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>829套冰川区域平均冰厚变化</title><style>body{{background:#f3f6f7;color:#23404e;font:16px/1.8 'Microsoft YaHei',sans-serif;margin:0}}main{{max-width:1100px;margin:auto;padding:30px;background:white}}h1{{font-size:29px}}h2{{font-size:21px;margin-top:30px}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:9px 11px;border-bottom:1px solid #dce5e8;text-align:right}}th{{background:#edf3f5}}th:first-child,td:first-child{{text-align:left}}img{{width:100%}}a{{color:#147a90}}.note{{padding:14px 19px;background:#fff3e6}}.scroll{{overflow:auto}}</style><main><p>2026-10-20 · 已冻结829套 · 本地10山脉区</p><h1>两期平均冰厚与区域变化</h1><p>区域平均冰厚按各期的总体积除以该期总面积计算：H̄＝1000ΣV（km³）/ΣA（km²）。变化为现代减小冰期，负值表示平均冰厚减少。各期按各自面积加权，不是逐冰川变化值的简单平均。</p><p><strong>原配对模型638套：{primary.LIA_area_weighted_mean_thickness_m:.2f} → {primary.modern_area_weighted_mean_thickness_m:.2f} m，变化{primary.change_modern_minus_LIA_m:+.2f} m。</strong>若以原小冰期面积固定归一化冰量差，则等效厚度变化为{primary.equivalent_change_on_LIA_area_m:+.2f} m；按两期平均面积归一化为{primary.equivalent_change_on_mean_endpoint_area_m:+.2f} m。这些量与两期各自平均冰厚之差含义不同。</p><div class="scroll">{table(show)}</div><p class="note">西、中、东喜马拉雅合计191套为旧LIA重建与现代反演参考的来源比较。其床面、标定与638套不同，旧档也没有同方法的现代二维重建端点；表中正值不能作为实际增厚或区域增冰结论。本页保留191套算术诊断，但不合成全829套的统一物理平均冰厚变化。</p><img src="区域平均冰厚变化829.png" alt="638套配对模型与191套旧来源比较的区域冰厚变化"><p><a href="平均冰厚变化_区域829.csv">下载中文区域表</a> · <a href="mean_thickness_per_system829.csv">下载829套逐冰川明细</a> · <a href="cohort_summaries_separate.csv">分来源总体统计</a></p><h2>来源与口径</h2><p>人口范围沿用2026-10-17/19冻结的829套：原654中符合主现代面积&gt;10 km²条件的638套，加上Himalaya191；没有按变化正负删样本。原654体积使用已冻结重叠修正。区域按既有本地10区；本地原名西天山的边界也覆盖东天山。</p><p>喜马拉雅7套现代成员字符串存在原DBF截断，829的简化成员清单为904个，而已审计旧模型实际使用950个。计算这191套时，体积与面积均采用该旧模型审计的配套账册，避免用950成员体积除以904成员面积；这不重定义829套系统身份。2692旧模型含9个另行标记的候选成员，其不确定性保留。<a href="legacy7_area_membership_difference.csv">7套面积口径差异</a>另表保存。旧191本身未与638套统一标定，以上只是来源诊断。</p><p>这是两期几何与模型体积的统计，不等于共同有冰区冰面降低，也不是年变化率。现代轮廓原638来源于1993–2008年，喜马拉雅原轮廓为1999–2010年，小冰期年代并非统一时点。没有新运行冰川模型、重建、气候模拟或重算栅格。</p></main></html>'''
(OUT/'report.html').write_text(doc,encoding='utf-8')
for k,p in SOURCES.items():checks[k+'_unchanged']=sha(p)==initial_hash[k]
checks={k:bool(v) for k,v in checks.items()}
assert all(checks.values())
summary={'status':'pass','created_utc':datetime.now(timezone.utc).isoformat(),'population':829,'primary_cohort_systems':638,'legacy_diagnostic_systems':191,
 'regional_rows':10,'primary638':primary.to_dict(),'cohorts_separate':cohort.to_dict(orient='records'),
 'definition':'Hbar_t=1000*sum(V_t_km3)/sum(A_t_km2); change=modern minus LIA; independent endpoint area weighting',
 'equivalent_change_definition':'1000*(sum(V_MOD)-sum(V_LIA))/reference_area; LIA and endpoint-mean area variants saved separately',
 'legacy_area_membership_discrepancy_systems':len(changes),'legacy_member_volume_count':len(members),'no_pooled829_physical_total':True,
 'new_model_runs':0,'source_receipts':{k:{'path':str(p),'sha256':initial_hash[k]} for k,p in SOURCES.items()},'checks':checks,'visual_review':'pending'}
(OUT/'summary_and_QC.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
shutil.copy2(__file__,OUT/'reproduce_analysis.py')
print(show.to_string(index=False))
print(json.dumps({'population':829,'checks_passed':sum(checks.values()),'output':str(OUT)},ensure_ascii=False))
