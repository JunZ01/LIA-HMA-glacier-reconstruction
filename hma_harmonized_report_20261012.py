from pathlib import Path
import json,html,hashlib,shutil
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path('H:/ToYangxin/result_yx/hma_harmonized_analysis_20261012_v1')
S=O/'selected654_analysis';M=O/'modern_identity';N=O/'new_inventory_matching';L=O/'legacy_volume_comparability'
def j(p):return json.loads(p.read_text(encoding='utf-8'))
def rec(d):return d.astype(object).where(pd.notna(d),None).to_dict(orient='records')
def dump(name,obj):(O/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf-8')
summary=j(S/'summary.json');total=summary['total'];legacy=j(L/'summary.json')
core=pd.read_csv(S/'selected654_merged_analysis.csv');old=pd.read_csv(M/'legacy245_endpoint_area_scenarios.csv');oldprimary=old[old.scenario=='confirmed_1196'].copy()
regions=pd.read_csv(S/'mountain_volume_area_length_budget654.csv');ua=pd.read_csv(M/'harmonized899_spatial_union_area_scenarios.csv');up=ua[(ua.legacy_scenario=='confirmed_1196')&(~ua.region.str.startswith('ALL'))].copy()
zh={'Hindu Kush':'兴都库什','West Himalaya':'西喜马拉雅','East Himalaya':'东喜马拉雅','Central Himalaya':'中喜马拉雅','Karakoram':'喀喇昆仑','Western Tien Shan':'本地天山区（原名西天山）','Western Kunlun Shan':'西昆仑','Nyainqentanglha':'念青唐古拉','Tibetan Interior Mountains':'青藏高原内部山地','Pamir':'帕米尔'}
master=[]
for r in core.itertuples():master.append({'system_key':'selected654:'+str(r.lia_unique_id),'cohort':'selected654','source_id':r.lia_unique_id,'record_id':r.record_id,'mountain':r.mountain,'mountain_zh':zh[r.mountain],'lia_area_km2':r.lia_area_geodesic_km2,'modern_area_km2':r.modern_area_geodesic_km2,'area_loss_km2':r.area_difference_LIA_minus_modern_km2,'area_loss_percent':r.area_decrease_relative_LIA_percent,'modern_membership_scenario':'frozen1256','modern_epoch':'1993–2008','paired_model_volume_available':True,'LIA_volume_km3':r.lia_volume_deduplicated_km3,'modern_volume_km3':r.modern_volume_km3,'net_volume_loss_km3':r.volume_loss_deduplicated_km3,'volume_loss_percent':r.volume_loss_deduplicated_percent,'LIA_stored_axis_m':r.LIA_max_stored_axis_m,'modern_stored_axis_m':r.modern_max_stored_axis_m,'stored_axis_loss_percent':r.axis_length_decrease_relative_LIA_percent,'volume_status':'冻结配对模型；现代体积匹配为标定','model_family':r.model_family})
audit=pd.read_csv(L/'per_system_audit.csv').set_index('lia_unique_id')
for r in oldprimary.itertuples():master.append({'system_key':'legacy245:'+str(r.lia_unique_id),'cohort':'legacy245','source_id':r.lia_unique_id,'record_id':'','mountain':r.mountain10,'mountain_zh':zh[r.mountain10],'lia_area_km2':r.lia_area_geodesic_km2,'modern_area_km2':r.modern_union_area_geodesic_km2,'area_loss_km2':r.lia_minus_modern_signed_area_km2,'area_loss_percent':r.area_reduction_relative_lia_pct,'modern_membership_scenario':'confirmed1196_existing_mapping','modern_epoch':'1999–2010','paired_model_volume_available':False,'LIA_volume_km3':None,'modern_volume_km3':None,'net_volume_loss_km3':None,'volume_loss_percent':None,'LIA_stored_axis_m':None,'modern_stored_axis_m':None,'stored_axis_loss_percent':None,'volume_status':'旧191试验方法待统一' if audit.loc[r.lia_unique_id,'has_static_lia191'] else '旧上游反演未完成','model_family':'未达到654共同标定口径'})
master=pd.DataFrame(master);assert len(master)==899 and master.system_key.is_unique
master.to_csv(O/'unified899_area_registry.csv',index=False,encoding='utf-8-sig')
regrows=[]
for r in up.itertuples():
 sub=master[master.mountain==r.region];rv=regions[regions.mountain==r.region]
 x={'mountain':r.region,'mountain_zh':zh[r.region],'area_systems':len(sub),'volume_systems':len(core[core.mountain==r.region]),'LIA_area_sum_km2':sub.lia_area_km2.sum(),'modern_area_sum_km2':sub.modern_area_km2.sum(),'area_loss_sum_percent':100*(sub.lia_area_km2.sum()-sub.modern_area_km2.sum())/sub.lia_area_km2.sum(),'LIA_spatial_union_km2':r.combined_lia_spatial_union_area_km2,'modern_spatial_union_km2':r.combined_modern_spatial_union_area_km2,'area_loss_union_percent':r.combined_spatial_union_area_reduction_relative_lia_pct,'net_modeled_volume_loss_km3':None,'modeled_volume_loss_percent':None,'stored_axis_loss_percent':None,'volume_scope':'654冻结模型' if len(rv) else '旧方法未统一；不加入冰量预算'}
 if len(rv):
  q=rv.iloc[0];x.update(net_modeled_volume_loss_km3=float(q.net_volume_loss_km3),modeled_volume_loss_percent=float(q.volume_loss_percent),stored_axis_loss_percent=float(q.axis_length_index_loss_percent))
 regrows.append(x)
reg=pd.DataFrame(regrows);reg.to_csv(O/'mountain10_first_analysis.csv',index=False,encoding='utf-8-sig')
matches=pd.read_csv(N/'system_matching.csv');work=[]
for r in audit.reset_index().itertuples():work.append({'source':'legacy245','source_id':r.lia_unique_id,'current_LIA_result_exists':r.has_static_lia191,'next_required_step':'统一原生冰厚床与现代二维体积标定；先检查标定闭合' if r.has_static_lia191 else '建立同一静态路线输入；旧气候MB括号失败保持记录','new_model_launched':False,'volume_eligible_now':False})
pd.DataFrame(work).to_csv(O/'legacy245_harmonization_worklist.csv',index=False,encoding='utf-8-sig')
priority=matches[['lia_system_id','region_label','completion_status','candidate_member_count','shared_modern_member_count','lia_area_geodesic_km2','exclusive_diagnostic_modern_union_area_geodesic_km2','diagnostic_LIA_ge_10_km2']].copy();priority['next_required_step']='确认LIA完成状态与主导现代归属；共同分区与年代来源登记';priority['included_in654_volume_budget']=False;priority.sort_values('exclusive_diagnostic_modern_union_area_geodesic_km2',ascending=False).to_csv(O/'new_inventory_followup_priority.csv',index=False,encoding='utf-8-sig')
methods=[
 '全899为654与旧245两套来源系统的面积比较登记，不是统一面积门槛的HMA总体样本。',
 '主面积表的旧245采用既有确认1196成员口径；原1159和含9候选的1205全部另表保留。确认分类来自已有映射，不是本轮重新人工裁定。',
 '面积同时给逐系统求和与跨系统空间union，二者不可混用；源轮廓不修改。',
 '冰量主统计仍为全部654。36条模型负损失保留；6条用户已剔除系统不恢复。',
 '冰量百分比=sum(V_LIA−V_MOD)/sum(V_LIA)，面积与长度指数比例同理；不是个体百分比的平均。',
 '长度是两端各自最长已保存参考轴指数，不是同一主支上的真实末端退缩。',
 '654现代参考期1993–2008，旧245为1999–2010；LIA年代未逐一确定，不计算统一年变化率。',
 '空间分解按已保存厚度>0网格；与矢量退缩区不同。共同区厚度差只有床一致子集可解释为地表降低。',
 '极小负厚度作为独立数值残项保留；旧overlap修正另列，总量与冻结结果闭合。',
 '旧191有结果不等于与654标定方法相同；其配对诊断131条负差不作为区域增冰结论。',
 '唐古拉691/东天山640未有needxd完成标记；1331条全部保留为候选。微小相交不自动并系统。',
 'F19与Millan现代产品对照是原生支持域积分的敏感性提示，不是误差条；两端都需同产品重建才能测试ΔV稳健性。',
 '本轮未启动新的冰川冰面重建或动力学模拟；统计关联检验另附，未给出正式冰量置信区间或气候因果归因。'
]
payload={'title':'HMA统一分析首轮','date':'2026-10-12','mountain_rows':rec(reg),'master899':rec(master),'selected654':rec(core),'decomposition654':rec(pd.read_csv(S/'spatial_volume_decomposition654.csv')),'legacy_area_scenarios':rec(old),'new_inventory_matching':rec(matches),'methods':methods,'total654':total,'legacy_audit':legacy,'sensitivity':rec(pd.read_csv(S/'sensitivity_summaries.csv')),'sources':[{'name':'654主分析','path':str(S/'summary.json')},{'name':'245身份与面积','path':str(M/'summary.json')},{'name':'旧191冰量方法审计','path':str(L/'summary.json')},{'name':'唐古拉/东天山配对','path':str(N/'summary.json')}]}
dump('workbook_payload.json',payload)

order=reg.sort_values('area_loss_union_percent',ascending=False)
fig,ax=plt.subplots(figsize=(12,6),layout='constrained')
cc=['#bd732a' if n in ['West Himalaya','Central Himalaya','East Himalaya'] else '#267e96' for n in order.mountain]
bars=ax.barh(order.mountain,order.area_loss_union_percent,color=cc);ax.invert_yaxis();ax.set_xlabel('LIA-to-modern area reduction, spatial-union footprint (%)');ax.set_title('Ten mountain regions: source-cohort area comparison (899 systems)',loc='left',fontsize=15)
for bar,n in zip(bars,order.area_systems):ax.text(bar.get_width()+.2,bar.get_y()+bar.get_height()/2,f'n={n}',va='center',fontsize=9)
ax.set_xlim(0,order.area_loss_union_percent.max()*1.16);ax.spines[['top','right']].set_visible(False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#267e96',label='Selected 654; modern 1993–2008'),Patch(color='#bd732a',label='Legacy 245; modern 1999–2010')],loc='lower right',fontsize=9)
fig.savefig(O/'mountain10_area_comparison899.png',dpi=300);plt.close(fig)

def table(df):return df.to_html(index=False,border=0,float_format=lambda x:f'{x:,.2f}',classes='data',na_rep='—')
display=reg[['mountain_zh','area_systems','area_loss_union_percent','volume_systems','net_modeled_volume_loss_km3','modeled_volume_loss_percent']].rename(columns={'mountain_zh':'山脉区','area_systems':'面积系统数','area_loss_union_percent':'面积减少%（union）','volume_systems':'冰量系统数','net_modeled_volume_loss_km3':'模型净损失km³','modeled_volume_loss_percent':'模型冰量减少%'})
part=pd.DataFrame([{'空间项':'共同正厚度网格：床一致','冰量项km³':total['common_bed_consistent_thickness_loss_km3']},{'空间项':'共同正厚度网格：床不一致或缺失','冰量项km³':total['common_bed_inconsistent_or_unknown_thickness_loss_km3']},{'空间项':'仅LIA端正厚度网格','冰量项km³':total['LIA_only_positive_volume_km3']},{'空间项':'仅现代端正厚度网格（扣除）','冰量项km³':-total['modern_only_positive_volume_km3']}])
uniformall=ua[(ua.legacy_scenario=='confirmed_1196')&ua.region.str.startswith('ALL')].iloc[0]
cv=j(O/'climate_geometry_exploration/cv_summary.json')
cv_metrics=pd.DataFrame(cv['overall_out_of_fold_metrics']).set_index('model').loc[['training_mean_baseline','geometry_ridge','geometry_climate_ridge']].reset_index()
cv_display=cv_metrics[['model','MAE_percentage_points','RMSE_percentage_points','out_of_fold_R2']].copy()
cv_display['model']=cv_display['model'].map({'training_mean_baseline':'训练样本均值基线','geometry_ridge':'6项几何变量','geometry_climate_ridge':'几何＋3项现代气候背景'})
cv_display.columns=['模型','折外平均绝对误差（百分点）','折外均方根误差（百分点）','折外R²']
cv_section=f'''<h2>6. 几何—气候背景检验：现代气候态未改善跨山脉预测</h2>
<p>已为全部654个系统接入1256个现代成员的高程、坡度和几何结构，并提取1981–2010年气候背景。使用七个山脉逐一留出的验证：每次用其余六个山脉拟合，再预测未参与拟合的山脉。保留全部36个负损失值，响应为模型推导的有符号冰量损失比例，每个系统等权。</p>
<div class="scroll">{table(cv_display)}</div>
<p>6项几何变量使整体平均绝对误差从9.40降至8.50个百分点，七个山脉各自都有改善；加入3项现代气候背景后整体误差为8.84个百分点，仅两个山脉改善。折外R²约0.11与0.06，说明现有变量对留出山脉中单个系统损失比例的预测能力仍有限。这是固定线性模型下的初步关联检验，<strong>不能解释成气候没有作用，也不能把几何预测能力视为独立机制验证</strong>。</p>
<img src="climate_geometry_exploration/cv_mountain_prediction_comparison.png" alt="七山脉留一验证：均值、几何、几何加气候模型的折外误差">
<p class="small">几何项：log(LIA面积)、现代成员面积加权平均高程、成员高差的面积加权平均、面积加权坡度、log(1＋成员数)、最大成员面积份额。气候项：年平均格点气温、log(1＋年降水量)、5–9月降水占比。固定Ridge系数α=1，标准化仅用训练山脉，未进行调参。14次Ridge拟合及7个训练均值基线均经独立复算，最大预测差为8.88×10⁻¹⁴个百分点。</p>
<p class="note">气候输入是164个粗网格点的现代背景，未经冰川高程订正，不能当作冰川表面气温或小冰期以来的气候变化。7次训练与留出组合中没有共享气候格点，但区域内重复使用和空间相关仍存在；654个系统、164个格点、7个验证折均不等于正式有效样本量。几何也参与冰面与冰量重建，且现代几何不等同于小冰期几何，预测关系中可能包含构建数据时引入的关联。</p>
<p>因此，这轮结果支持的下一步是补齐历史气候变化与年代约束，并检验两端冰厚方案的不确定性。初步背景关联已完成，历史气候归因尚未完成。</p>
<p><a href="climate_geometry_exploration/geometry_covariates654.csv">654几何属性</a> · <a href="climate_geometry_exploration/climate_background654.csv">654气候背景</a> · <a href="climate_geometry_exploration/cv_predictions1962.csv">全部折外预测</a> · <a href="climate_geometry_exploration/cv_summary.json">验证汇总</a> · <a href="climate_geometry_exploration/cv_QC.json">独立计算QC</a></p>'''
rules=''.join('<li>'+html.escape(s)+'</li>' for s in methods)
doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HMA统一分析首轮结果</title><style>body{{margin:0;background:#f4f7f7;color:#213b44;font:16px/1.75 'Microsoft YaHei',sans-serif}}main{{max-width:1200px;margin:auto;background:white;padding:38px}}h1{{font-size:31px}}h2{{margin-top:38px;padding-top:22px;border-top:1px solid #d9e4e5;font-size:23px}}h3{{font-size:19px}}.small{{font-size:13px;color:#607680}}.note{{padding:16px 22px;border-left:4px solid #c27c31;background:#fff6eb}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}}.card{{border:1px solid #dce5e5;padding:16px;border-radius:8px}}.big{{font-size:30px;color:#147d92;font-weight:bold}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px 13px;border-bottom:1px solid #dce5e5;text-align:left;vertical-align:top}}th{{background:#edf4f4}}a{{color:#007c91}}li{{margin:8px 0}}.scroll{{overflow:auto}}@media(max-width:700px){{main{{padding:18px}}.cards{{grid-template-columns:1fr}}}}</style><main>
<p class="small">2026-10-12 · 冻结结果再分析与多源口径核验 · 未重跑冰川冰面重建或动力学模拟</p><h1>HMA统一分析：首轮结果与关键方法差异</h1><p>已完成899条来源系统的面积登记、654套模型的区域冰量预算、空间分解与几何—气候背景检验，以及唐古拉、东天山1331条轮廓的现代成员候选配对。本轮已产生可核查的统计结果；共同方法的全HMA冰量预算与气候归因尚未完成。</p>
<div class="cards"><div class="card"><div class="big">{total['net_volume_loss_km3']:.2f} km³</div>654模型净冰量损失<br><span class="small">相对LIA冰量减少{total['volume_loss_percent']:.2f}%</span></div><div class="card"><div class="big">{total['common_positive_share_of_net_loss_percent']:.1f}%</div>净损失中的共同有冰区厚度差<br><span class="small">空间收支，不能直接作为消融过程归因</span></div><div class="card"><div class="big">899</div>已统一面积登记的来源系统<br><span class="small">654＋245；现代参考年份不同</span></div></div>
<p><a href="HMA统一分析首轮.xlsx">下载分析工作簿</a> · <a href="mountain10_first_analysis.csv">山脉总表</a> · <a href="unified899_area_registry.csv">899系统面积登记</a> · <a href="selected654_analysis/selected654_merged_analysis.csv">654完整分析表</a></p>
<h2>1. 区域变化量：绝对损失与相对损失应同时看</h2><p>654套的LIA模型冰量为{total['lia_volume_km3']:.2f} km³，现代为{total['modern_volume_km3']:.2f} km³。喀喇昆仑的净损失最大（100.76 km³），但相对损失为7.08%；兴都库什相对损失为15.09%，念青唐古拉为14.47%。这些都是当前选定样本与模型的累计变化，不代表各山脉全部冰川或统一年代的变化率。</p><div class="scroll">{table(display)}</div><p class="small">面积列为跨系统去重的空间union；654冻结结果仍同时保留逐系统求和口径。喜马拉雅三行没有可并入当前标定方法的冰量，空值不等于0。</p><img src="selected654_analysis/selected654_regional_budget.png" alt="654系统的区域冰量、面积长度比例与规模集中度"><p>按初始冰量排序最大的66条（约10%系统）占初始冰量58.08%，贡献净损失40.52%。因此“储存量占比”和“损失贡献”不同。这个样本关系同时受规模与所在山脉影响，尚不能解释成大冰川对气候更不敏感。</p>
<h2>2. 空间分解：大部分变化发生在轮廓仍保留的有冰区域</h2><p>共同正厚度网格贡献{total['common_positive_thickness_loss_km3']:.2f} km³，占净损失70.35%；其中满足共同床差≤0.002 m的部分为{total['common_bed_consistent_thickness_loss_km3']:.2f} km³，占净损失{100*total['common_bed_consistent_thickness_loss_km3']/total['net_volume_loss_km3']:.2f}%。仅LIA端有正厚度的网格贡献{total['LIA_only_positive_volume_km3']:.2f} km³，现代端独有部分抵消约{total['modern_only_positive_volume_km3']:.3f} km³。</p><div class="scroll">{table(part)}</div><p class="small">另扣已冻结overlap修正0.0000322265 km³，并保留约−9.95×10⁻¹² km³的存储负厚度数值残项。此为两端厚度数组的空间恒等分解；正厚度边界并非矢量冰川边界，不把它叫作实测末端退缩或气候过程贡献。</p><img src="selected654_analysis/selected654_spatial_volume_partition.png" alt="七个山脉模型冰量差的空间分解">
<h2>3. 喜马拉雅面积可以比较，旧冰量需要重新统一标定</h2><p>旧245原Shapefile成员名单受254字符字段限制，7个系统名单发生截断。现存确认映射恢复37个成员，得到1196条；2692的9个候选成员仍单独保留。原1159、确认1196和含候选1205三种方案均已计算。</p><p>确认1196口径下，喜马拉雅245的系统面积减少16.661%；跨系统去重后为16.667%。654与旧245之间没有现代轮廓的正面积重叠。合并899条来源系统后，空间union面积为{uniformall.combined_lia_spatial_union_area_km2:,.2f}→{uniformall.combined_modern_spatial_union_area_km2:,.2f} km²，减少{uniformall.combined_spatial_union_area_reduction_relative_lia_pct:.3f}%。该合计描述当前资料集合，不能外推HMA总体。</p><img src="mountain10_area_comparison899.png" alt="含喜马拉雅的十个山脉面积减少比例"><p class="note"><strong>旧191冰量不能直接并表。</strong>它用主流线厚度误差标定，654用现代二维总体积标定，床面来源也不同。既有旧191的LIA体积488.826 km³、现代反演参考503.988 km³，131条出现负差。共同床核验和原数组积分均通过，因此这不是简单表格加总错误；在完成同方法现代二维标定之前，不将该负差解释为喜马拉雅增冰。另54条缺失来自上游气候MB括号失败，不填0，也未启动替代模型。</p>
<h2>4. 唐古拉、东天山：配对已完成，仍需确定正式系统与完成状态</h2><p>唐古拉691条全部找到现代相交候选；东天山640条中627条找到候选，13条无候选均小于1 km²，不能据此认定已经消失。共核验3586个现代成员池，形成3096条正面积相交候选。</p><p>共享现代成员多数来自很小的边界穿越：唐古拉429个共享成员中427个的次大相交比例低于5%；东天山198个中161个低于5%。因此保留全部相交证据和最大相交排他归属诊断，不把全相交连通图直接当成冰川系统。5%/50%仅用于敏感性分析，没有用来删样本。</p><p>两套新LIA文件均无needxd完成字段，暂未加入正式冰量预算。<a href="new_inventory_matching/system_matching.csv">1331条逐系统配对表</a>和<a href="new_inventory_followup_priority.csv">后续核对顺序</a>已保存。</p>
<p class="small">采用本地10个山脉分区：唐古拉归入较宽的青藏高原内部山地，现有该区83套不能全部称为唐古拉冰川；原名“西天山”的本地边界也覆盖东天山。新轮廓的区域来源标签另外保留。</p>
<h2>5. 冰厚产品敏感性成为下一项必要实验</h2><p>在全部现代成员均有两套正厚度支持、且各自覆盖至少90%的511个系统中，421个系统的F19与Millan现代冰量差的绝对值，大于当前模型LIA—现代冰量变化的绝对值。这个筛查使用两产品原生支持域积分，尚非完全同域对照，也不是误差区间。</p><p><strong>下一步应在相同轮廓、共同数据支持范围和一致标定下，对两种床面方案分别重建LIA与现代端点。</strong>只替换现代冰量会破坏两期重建误差的相关性，不能代表冰量变化的不确定性。这项检验将决定哪些区域总量、变化比例和空间分解可作为稳健结论。</p>
{cv_section}
<h2>已落实的后续工作清单</h2><ol><li><strong>喜马拉雅同方法重建：</strong><a href="legacy245_harmonization_worklist.csv">245条逐系统清单</a>已建。先核验原生冰厚床和现代二维体积闭合，再登记共同路线；旧191和缺失54分别保留原因。</li><li><strong>冰厚方案的两端配对敏感性：</strong>先比较现代共同支持域，再进行同一方案内的两端重建，分别传播到区域冰量、损失比例和共同冰区厚度变化。</li><li><strong>历史气候与年代约束：</strong>现代背景的七山脉留一检验已完成；后续应检验小冰期至各编目端点的气候变化，逐步纳入高程分布、表碛与资料质量，在共同模型与独立观测约束下解释区域差异。</li></ol>
<h2>核验与方法</h2><p>654全部保留，36条模型冰量增加也保留。3270个用于分解的栅格文件hash匹配；逐系统体积闭合最大误差1.35×10⁻⁸ km³。独立核验包含全样本表格重算和8个预选源NetCDF样本，以及一个微小负厚度边界样本。</p><p><a href="selected654_analysis/QC.json">主统计QC</a> · <a href="selected654_independent_qc/independent_analysis_QC.json">独立QC</a> · <a href="modern_identity/final_QC.json">身份与面积QC</a> · <a href="legacy_volume_comparability/QC.json">旧冰量方法QC</a> · <a href="new_inventory_matching/independent_QC.json">新编目配对QC</a></p><details><summary>全部解释规则</summary><ol>{rules}</ol></details></main></html>'''
(O/'report.html').write_text(doc,encoding='utf-8')
dump('first_analysis_summary.json',{'selected654':total,'area_source_systems':899,'area_membership_primary':'confirmed1196 existing mapping + frozen654','area_union_primary':rec(ua[(ua.legacy_scenario=='confirmed_1196')&ua.region.str.startswith('ALL')])[0],'legacy_same_method_volume_eligible':0,'new_inventory_candidate_records':1331,'new_ice_reconstruction_or_dynamic_runs':0,'statistical_ridge_fits':14,'training_mean_baselines':7,'geometry_climate_background_association_completed':True,'geometry_climate_validation_summary':cv,'climate_attribution_completed':False,'next_worklists':['legacy245_harmonization_worklist.csv','new_inventory_followup_priority.csv'],'all_required_source_hashes_checked_in_component_QC':True})
shutil.copy2(__file__,O/'reproduce_report.py');print('written report, tables and workbook payload')
