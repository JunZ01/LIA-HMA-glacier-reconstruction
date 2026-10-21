from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, shutil
import pandas as pd
import numpy as np

BASE=Path('H:/ToYangxin/result_yx')
SRC=BASE/'mean_thickness829_20261020_v1'
OUT=BASE/'mean_thickness829_20261020_v2'
OUT.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources={n:SRC/n for n in ['regional_mean_thickness829.csv','mean_thickness_per_system829.csv','cohort_summaries_separate.csv','summary_and_QC.json']}
hashes={n:sha(p) for n,p in sources.items()}
reg=pd.read_csv(sources['regional_mean_thickness829.csv'])
per=pd.read_csv(sources['mean_thickness_per_system829.csv'])
coh=pd.read_csv(sources['cohort_summaries_separate.csv'])
assert len(per)==829 and per.record_id.is_unique and len(reg)==10
assert int(per.legacy_model_comparison_only.sum())==191

# Exact algebraic decomposition with LIA area as the fixed reference. These
# terms explain the arithmetic only; they do not identify physical causes.
diagnostics=[]
for frame,scope in [(reg,'region'),(coh,'cohort')]:
 for r in frame[frame.cohort=='Himalaya191'].itertuples():
  al,am,vl,vm=r.LIA_area_km2,r.modern_area_km2,r.LIA_volume_km3,r.modern_volume_km3
  volume_term=1000*(vm-vl)/al
  denominator_term=1000*vm*(1/am-1/al)
  diagnostics.append({'scope':scope,'region':r.region,'region_zh':r.region_zh,'systems':r.systems,
     'LIA_area_km2':al,'modern_area_km2':am,'LIA_source_volume_km3':vl,'modern_reference_volume_km3':vm,
     'area_change_percent':100*(am/al-1),'source_volume_change_percent':100*(vm/vl-1),
     'raw_VoverA_difference_m':r.change_modern_minus_LIA_m,
     'volume_difference_over_LIA_area_term_m':volume_term,
     'changing_area_denominator_term_m':denominator_term,
     'identity_residual_m':volume_term+denominator_term-r.change_modern_minus_LIA_m,
     'interpretation':'算术分解；不是实际增厚、动力归因或独立观测结论'})
diag=pd.DataFrame(diagnostics)
assert diag.identity_residual_m.abs().max()<1e-10
diag.to_csv(OUT/'legacy191_arithmetic_explanation.csv',index=False,encoding='utf-8-sig')

display=reg[['region','region_zh','systems','cohort']].copy()
valid=~reg.legacy_model_comparison_only
display['paired_model_change_available']=valid
for new,old in [
 ('LIA_mean_thickness_m','LIA_area_weighted_mean_thickness_m'),
 ('modern_mean_thickness_m','modern_area_weighted_mean_thickness_m'),
 ('primary_equivalent_change_on_LIA_area_m','equivalent_change_on_LIA_area_m'),
 ('auxiliary_mean_thickness_difference_m','change_modern_minus_LIA_m'),
 ('equivalent_change_on_mean_endpoint_area_m','equivalent_change_on_mean_endpoint_area_m')]:
 display[new]=reg[old].where(valid,np.nan)
display['status']=np.where(valid,'已有配对模型估计','旧LIA与现代参考方法未统一，变化值待核对')
display.to_csv(OUT/'regional_results_corrected829.csv',index=False,encoding='utf-8-sig')
cn=display[['region_zh','systems','primary_equivalent_change_on_LIA_area_m','auxiliary_mean_thickness_difference_m','status']].rename(columns={
 'region_zh':'山脉区域','systems':'冰川系统数','primary_equivalent_change_on_LIA_area_m':'小冰期固定参考面积上等效厚度变化_m',
 'auxiliary_mean_thickness_difference_m':'两期各自范围平均冰厚之差_m','status':'结果状态'})
cn.to_csv(OUT/'修正版区域厚度变化829.csv',index=False,encoding='utf-8-sig')

detail=per[['record_id','lia_unique_id','cohort','region','region_zh']].copy()
available=~per.legacy_model_comparison_only
detail['paired_model_change_available']=available
for col in ['LIA_mean_thickness_m','modern_mean_thickness_m','mean_thickness_change_modern_minus_LIA_m',
            'equivalent_change_on_LIA_area_m','equivalent_change_on_mean_endpoint_area_m']:
 detail[col]=per[col].where(available,np.nan)
 detail['legacy_diagnostic_only_'+col]=per[col].where(~available,np.nan)
detail['status']=np.where(available,'已有配对模型估计','旧两期方法未统一，待核对；诊断列不作实际变化结论')
detail.to_csv(OUT/'per_system829_corrected.csv',index=False,encoding='utf-8-sig')

primary=coh[coh.cohort=='Previous654'].iloc[0]
why=diag[diag.scope=='region'][['region_zh','systems','area_change_percent','source_volume_change_percent','raw_VoverA_difference_m']].rename(columns={
 'region_zh':'区域','systems':'数量','area_change_percent':'面积变化%','source_volume_change_percent':'旧来源体积变化%',
 'raw_VoverA_difference_m':'旧V/A差（m）'})
def table(d):return d.to_html(index=False,border=0,float_format=lambda x:f'{x:,.2f}',na_rep='待核对')
html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>厚度变化解释修正</title><style>body{{font:16px/1.8 'Microsoft YaHei',sans-serif;color:#263f4b;background:#f3f6f7;margin:0}}main{{max-width:1080px;background:white;margin:auto;padding:32px}}h1{{font-size:28px}}h2{{font-size:21px;margin-top:30px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #dce5e8;text-align:right}}th{{background:#edf3f5}}th:first-child,td:first-child{{text-align:left}}a{{color:#13758a}}.note{{background:#fff2e4;padding:16px 20px}}.scroll{{overflow:auto}}</style><main>
<p>2026-10-20 · 829套样本保留 · 对v1解释与主表的修正</p><h1>喜马拉雅正差不能作为实际增厚结论</h1>
<p class="note">此前将旧喜马拉雅191套的来源比较值放在区域冰厚变化主表中，容易被理解为普遍增厚，属于呈现错误。修正版将其从变化结论中撤下，191套身份和全部原始数值仍保留；未知值标为待核对，不填0、不改负号、不按符号删冰川。</p>
<h2>正差怎样产生？</h2><p>一是统计范围变化。平均冰厚H̄=V/A，在薄冰舌消失、面积缩小时，即使冰量减少，剩余冰区的平均冰厚也可能增加。中喜马拉雅旧来源体积减少3.71%，面积减少17.91%，所以V/A反而增大17.30%，出现+12.28 m。它不代表同一位置冰体变厚12.28 m。</p>
<div class="scroll">{table(why)}</div><p>以原小冰期面积A₀为固定参考，有精确恒等式：H̄₁−H̄₀ = 1000(V₁−V₀)/A₀ + 1000V₁(1/A₁−1/A₀)。旧191套合计的+17.89 m可代数分成+2.30 m体积差项与+15.60 m面积分母变化项。这一分解解释计算值，不是物理过程归因。</p>
<p>二是两期来源方法不一致。旧191套LIA体积为488.83 km³，现代反演参考为503.99 km³，其中131套现代参考体积更大；旧LIA使用内部主流线厚度拟合，现代值来自反演分配冰厚，缺少同一二维重建算子的现代控制端点。已审计的数组积分可复现，但不能据此证明历史增冰。到底床面、边界或标定转移各贡献多少，现有审计没有辨明，不能指定单一根因。</p>
<h2>修正后的区域厚度表</h2><p>主列改为固定小冰期面积上的等效厚度变化：ΔH_eq=1000(V_modern−V_LIA)/A_LIA，负值表示净冰量损失折算的等效厚度减少。辅助列保留两期各自范围内的V/A之差，明确其统计范围会变化。两列都是模型估计，均不是年变化率或独立实测冰面变化。</p><div class="scroll">{table(cn)}</div>
<p>可用的638套配对模型合计：固定小冰期面积上的等效厚度变化为{primary.equivalent_change_on_LIA_area_m:.2f} m；两期各自范围平均冰厚之差为{primary.change_modern_minus_LIA_m:.2f} m。829套中其余191套暂不进入正式厚度变化汇总，未计算统一全829套物理均值。</p>
<h2>当前应优先关注的指标</h2><ol><li>冰量变化ΔV及其相对LIA比例：同时报告绝对损失与相对损失。</li><li>固定参考面积上的等效厚度变化ΔV/A_LIA：用于区域净亏损强度比较，明确它不是逐点变薄。</li><li>面积变化ΔA及其比例：描述范围退缩，与冰量变化逐冰川对照。</li><li>ELA变化ΔELA：比较山脉差异，并注明所用ELA方法和参数敏感性；不直接等同温度变化。</li><li>长度变化：当前保存参考轴指数作为辅助，真实同主支末端退缩另需其相应几何证据。</li></ol>
<p>两期V/A差适合作为冰体结构变化的辅助指标。喜马拉雅下一步须建立可比的两期冰厚与体积基准，确定变化符号的可信度。不能通过强制负值、删掉正值或拿其他候选模拟替换旧端点来修正。</p>
<p><a href="修正版区域厚度变化829.csv">中文区域表</a> · <a href="per_system829_corrected.csv">829套明细与诊断列</a> · <a href="legacy191_arithmetic_explanation.csv">喜马拉雅算术分解</a> · <a href="../mean_thickness829_20261020_v1/report.html">保留的v1历史输出</a></p>
<p>依据既有方法审计与当前冻结表，未新运行重建、冰流、气候模拟，未更改源数据、全局账册或其他任务。</p></main></html>'''
(OUT/'report.html').write_text(html,encoding='utf-8')
checks={'829_records_preserved':len(detail)==829,'638_paired_records':int(available.sum())==638,
 '191_pending_records':int((~available).sum())==191,
 '191_formal_changes_blank':detail.loc[~available,'equivalent_change_on_LIA_area_m'].isna().all(),
 '191_diagnostics_preserved':np.array_equal(detail.loc[~available,'legacy_diagnostic_only_mean_thickness_change_modern_minus_LIA_m'].to_numpy(),per.loc[~available,'mean_thickness_change_modern_minus_LIA_m'].to_numpy()),
 '638_values_unchanged':np.array_equal(detail.loc[available,'equivalent_change_on_LIA_area_m'].to_numpy(),per.loc[available,'equivalent_change_on_LIA_area_m'].to_numpy()),
 'decomposition_closed':bool(diag.identity_residual_m.abs().max()<1e-10),
 'original_sources_unchanged':all(sha(p)==hashes[n] for n,p in sources.items()),
 'ten_regions_count_closed':int(display.systems.sum())==829}
assert all(checks.values())
qc={'status':'pass','created_utc':datetime.now(timezone.utc).isoformat(),'correction':'withdraw legacy191 source-pairing differences from physical change conclusions; retain all829 and raw diagnostic signs',
 'source_hashes':hashes,'checks':{k:bool(v) for k,v in checks.items()},'new_model_runs':0,
 'scientific_limits':'No single root cause of legacy calibration/bed/boundary bias is established. Algebraic decomposition does not establish dynamic or climate mechanisms.'}
(OUT/'correction_QC.json').write_text(json.dumps(qc,ensure_ascii=False,indent=2),encoding='utf-8')
shutil.copy2(__file__,OUT/'reproduce_correction.py')
print(diag[['region_zh','systems','raw_VoverA_difference_m','volume_difference_over_LIA_area_term_m','changing_area_denominator_term_m']].to_string(index=False))
print('Correction complete; all829 preserved; legacy191 change conclusions withdrawn; no source modified.')
