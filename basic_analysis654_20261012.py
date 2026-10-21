from pathlib import Path
from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib, json, html, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('H:/ToYangxin/result_yx/basic_analysis654_20261012_v1')
SOURCE=Path('H:/ToYangxin/result_yx/hma_harmonized_analysis_20261012_v1/selected654_analysis/selected654_merged_analysis.csv')
OUT.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source_hash=sha(SOURCE)
d=pd.read_csv(SOURCE)
assert len(d)==654 and d.record_id.is_unique
ind=json.loads((OUT/'independent_check.json').read_text(encoding='utf-8'))
assert ind['status']=='pass' and ind['source_sha256']==source_hash
METRICS={
 'area':('面积','lia_area_geodesic_km2','modern_area_geodesic_km2','area_difference_LIA_minus_modern_km2','area_decrease_relative_LIA_percent','km²',1,0),
 'volume':('模型冰量','lia_volume_deduplicated_km3','modern_volume_km3','volume_loss_deduplicated_km3','volume_loss_deduplicated_percent','km³',1,0),
 'stored_axis':('参考轴长度指数','LIA_max_stored_axis_m','modern_max_stored_axis_m','axis_length_decrease_m','axis_length_decrease_relative_LIA_percent','km',.001,1e-6)}
ZH={'Karakoram':'喀喇昆仑','Pamir':'帕米尔','Western Tien Shan':'本地天山区（原名西天山）','Tibetan Interior Mountains':'青藏高原内部山地','Nyainqentanglha':'念青唐古拉','Western Kunlun Shan':'西昆仑','Hindu Kush':'兴都库什'}
ORDER=['Karakoram','Pamir','Western Tien Shan','Tibetan Interior Mountains','Nyainqentanglha','Western Kunlun Shan','Hindu Kush']
BINS=[0,10,25,50,100,np.inf]
LABELS=['<10','10–<25','25–<50','50–<100','≥100']
d['LIA_area_group_km2']=pd.cut(d.lia_area_geodesic_km2,BINS,labels=LABELS,right=False)
assert d.LIA_area_group_km2.notna().all()

def stats(sub):
 row={'systems':len(sub)}
 for k,(label,a,b,delta,pct,unit,scale,tol) in METRICS.items():
  row.update({k+'_LIA_sum':float(sub[a].sum()*scale),k+'_modern_sum':float(sub[b].sum()*scale),
    k+'_signed_loss_sum':float(sub[delta].sum()*scale),k+'_total_loss_pct':float(sub[delta].sum()/sub[a].sum()*100),
    k+'_median_loss_pct':float(sub[pct].median()),k+'_q25_loss_pct':float(sub[pct].quantile(.25)),k+'_q75_loss_pct':float(sub[pct].quantile(.75)),
    k+'_median_signed_loss':float(sub[delta].median()*scale),k+'_decrease_count':int((sub[delta]>tol).sum()),
    k+'_increase_count':int((sub[delta]<-tol).sum()),k+'_equal_count':int((sub[delta].abs()<=tol).sum()),
    k+'_LIA_share_pct':float(sub[a].sum()/d[a].sum()*100),k+'_net_loss_share_pct':float(sub[delta].sum()/d[delta].sum()*100)})
 return row

total=stats(d)
mount=pd.DataFrame([{'mountain':m,'mountain_zh':ZH[m],**stats(d[d.mountain==m])} for m in ORDER])
size=pd.DataFrame([{'LIA_area_class_km2':s,**stats(d[d.LIA_area_group_km2==s])} for s in LABELS])
overall=[]
for k,(label,a,b,delta,pct,unit,scale,tol) in METRICS.items():
 overall.append({'指标':label,'单位':unit,'系统数':654,'小冰期合计':total[k+'_LIA_sum'],'现代合计':total[k+'_modern_sum'],
   '减少量合计':total[k+'_signed_loss_sum'],'总量减少比例_pct':total[k+'_total_loss_pct'],'个体比例中位数_pct':total[k+'_median_loss_pct'],
   '个体比例第25百分位_pct':total[k+'_q25_loss_pct'],'个体比例第75百分位_pct':total[k+'_q75_loss_pct'],
   '个体减少量中位数':total[k+'_median_signed_loss'],'减少系统数':total[k+'_decrease_count'],'增加系统数':total[k+'_increase_count'],'相等系统数':total[k+'_equal_count']})
overall=pd.DataFrame(overall)
overall.to_csv(OUT/'总体统计.csv',index=False,encoding='utf-8-sig')
mount.to_csv(OUT/'山脉统计.csv',index=False,encoding='utf-8-sig')
size.to_csv(OUT/'规模分组统计.csv',index=False,encoding='utf-8-sig')
keep=['record_id','mountain','lia_area_geodesic_km2','modern_area_geodesic_km2','area_difference_LIA_minus_modern_km2','area_decrease_relative_LIA_percent','lia_volume_deduplicated_km3','modern_volume_km3','volume_loss_deduplicated_km3','volume_loss_deduplicated_percent','LIA_max_stored_axis_m','modern_max_stored_axis_m','axis_length_decrease_m','axis_length_decrease_relative_LIA_percent','LIA_area_group_km2']
d[keep].to_csv(OUT/'654逐冰川基础表.csv',index=False,encoding='utf-8-sig')

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titlelocation':'left','figure.facecolor':'white','savefig.facecolor':'white'})
COLORS=['#258497','#ca7542','#638056']
def save(fig,name):
 fig.savefig(OUT/(name+'.png'),dpi=220,bbox_inches='tight')
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
 plt.close(fig)

fig,axs=plt.subplots(2,3,figsize=(13.8,7.8))
fig.subplots_adjust(left=.07,right=.99,top=.84,bottom=.12,wspace=.30,hspace=.62)
fig.suptitle('654套冰川系统：总体变化与个体差异',x=.07,y=.985,ha='left',fontsize=18,fontweight='bold')
fig.text(.07,.92,'上排：两期合计及其变化比例；下排：全部654套的个体变化比例分布。两者回答不同的问题。',fontsize=11,color='#4b5c65')
for i,(k,(label,a,b,delta,pct,unit,scale,tol)) in enumerate(METRICS.items()):
 ax=axs[0,i]; vals=[total[k+'_LIA_sum'],total[k+'_modern_sum']]
 bars=ax.bar(['小冰期','现代'],vals,color=['#adc9d0',COLORS[i]],width=.55)
 ax.set_title(f'{label}合计（{unit}）',fontsize=12,pad=27)
 ax.text(0,1.025,f'合计减少 {total[k+"_total_loss_pct"]:.2f}%',transform=ax.transAxes,color=COLORS[i],fontsize=11,fontweight='bold')
 ax.bar_label(bars,labels=[f'{v:,.1f}' for v in vals],padding=4,fontsize=10)
 ax.set_ylim(0,max(vals)*1.19);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
 ax=axs[1,i]
 ax.hist(d[pct],bins=28,color=COLORS[i],edgecolor='white',linewidth=.7)
 ax.axvline(0,color='#565f63',lw=.8)
 ax.axvline(total[k+'_median_loss_pct'],color='#223c4b',ls='--',lw=1.5)
 ax.set_title(f'个体中位数 {total[k+"_median_loss_pct"]:.2f}%',fontsize=12)
 ax.set_xlabel('减少比例（%；负值表示增加）');ax.set_ylabel('系统数')
 ax.text(.98,.96,f'中间50%：{total[k+"_q25_loss_pct"]:.2f}–{total[k+"_q75_loss_pct"]:.2f}%',transform=ax.transAxes,ha='right',va='top',fontsize=9)
fig.text(.07,.025,'模型冰量来自已有配对重建；参考轴指数不是实测末端退缩。所有负值保留，未按变化方向筛选。',fontsize=10,color='#4b5c65')
save(fig,'图1_总体与个体分布')

fig,axs=plt.subplots(1,3,figsize=(15.4,6.7),sharey=True,gridspec_kw={'width_ratios':[.9,1.1,1.65]})
fig.subplots_adjust(left=.20,right=.99,top=.82,bottom=.19,wspace=.13)
fig.suptitle('按山脉看：有多少、减少多少、减少比例多大',x=.035,y=.98,ha='left',fontsize=18,fontweight='bold')
fig.text(.035,.91,'当前654套分布在本地10个山脉区中的7区；图中比例由区内两期总量计算。',fontsize=11,color='#4b5c65')
y=np.arange(len(mount))
for ax,key,title,color in [(axs[0],'systems','样本数量（套）','#668994'),(axs[1],'volume_signed_loss_sum','模型冰量净减少（km³）',COLORS[1])]:
 bars=ax.barh(y,mount[key],color=color,height=.54)
 ax.bar_label(bars,labels=[f'{v:.0f}' if key=='systems' else f'{v:.2f}' for v in mount[key]],padding=4,fontsize=10)
 ax.set_xlim(0,mount[key].max()*1.25);ax.set_title(title,fontsize=12,pad=17)
 ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
axs[0].set_yticks(y,mount.mountain_zh);axs[0].invert_yaxis()
for j,(k,label) in enumerate([('area','面积'),('volume','模型冰量'),('stored_axis','参考轴指数')]):
 axs[2].barh(y+(j-1)*.23,mount[k+'_total_loss_pct'],height=.21,color=COLORS[j],label=label)
axs[2].set_title('减少比例（%）',fontsize=12,pad=17);axs[2].set_xlim(0,18)
axs[2].legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncols=3,frameon=False,fontsize=9)
axs[2].grid(axis='x',alpha=.16);axs[2].set_axisbelow(True)
for ax in axs[1:]:ax.tick_params(axis='y',left=False)
fig.text(.035,.03,'仅描述当前样本。青藏高原内部山地不全是唐古拉；本地“西天山”边界也覆盖东天山。',fontsize=10,color='#4b5c65')
save(fig,'图2_山脉对比')

fig,axs=plt.subplots(2,2,figsize=(12,8))
fig.subplots_adjust(left=.08,right=.98,top=.85,bottom=.12,hspace=.42,wspace=.28)
fig.suptitle('按小冰期面积分组：数量、储存量和变化',x=.07,y=.985,ha='left',fontsize=18,fontweight='bold')
fig.text(.07,.93,'分组仅用于展示全部654套；没有新增面积门槛，也没有删去小于10 km²的系统。',fontsize=11,color='#4b5c65')
x=np.arange(len(size))
for ax,key,title,fmt,color in [(axs[0,0],'systems','各规模组数量（套）','{:.0f}','#668994'),(axs[0,1],'volume_LIA_share_pct','占样本初始冰量（%）','{:.1f}',COLORS[1])]:
 bars=ax.bar(x,size[key],color=color,width=.58);ax.bar_label(bars,labels=[fmt.format(v) for v in size[key]],padding=3)
 ax.set_ylim(0,size[key].max()*1.23);ax.set_title(title,fontsize=12)
for ax,k,color in [(axs[1,0],'area',COLORS[0]),(axs[1,1],'volume',COLORS[1])]:
 bars=ax.bar(x,size[k+'_total_loss_pct'],width=.58,color=color)
 ax.bar_label(bars,labels=[f'{v:.2f}' for v in size[k+'_total_loss_pct']],padding=4)
 ax.scatter(x,size[k+'_median_loss_pct'],color='#243d4b',s=30,zorder=3,label='个体比例中位数')
 ax.set_ylim(0,max(size[k+'_total_loss_pct'].max(),size[k+'_median_loss_pct'].max())*1.25)
 ax.set_title(('面积' if k=='area' else '模型冰量')+'减少比例（%）',fontsize=12)
 ax.legend(loc='upper right',fontsize=8,frameon=False)
 ax.set_xlabel('小冰期面积组（km²）')
for ax in axs.flat:
 ax.set_xticks(x,LABELS);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.text(.07,.025,'下排柱形为组内总量的减少比例，黑点为个体比例中位数。分组差别可能同时受山脉组成影响。',fontsize=10,color='#4b5c65')
save(fig,'图3_规模分组')

checks={'all654_unique':len(d)==654 and d.record_id.is_unique,'source_matches_independent_check':source_hash==ind['source_sha256'],'all7_mountain_counts_close':int(mount.systems.sum())==654,'all5_size_group_counts_close':int(size.systems.sum())==654}
for k,v in METRICS.items():
 for suffix,indkey in [('total_loss_pct','loss_percent_of_LIA_total'),('median_loss_pct','individual_loss_percent_median'),('q25_loss_pct','individual_loss_percent_q25'),('q75_loss_pct','individual_loss_percent_q75')]:
  checks[k+'_'+suffix+'_independent_match']=bool(np.isclose(total[k+'_'+suffix],ind['overall'][k][indkey],rtol=0,atol=1e-10))
 for grouping,frame in [('mountain',mount),('size',size)]:
  checks[k+'_'+grouping+'_total_closure']=bool(np.isclose(frame[k+'_signed_loss_sum'].sum(),total[k+'_signed_loss_sum'],rtol=0,atol=1e-9))
checks['source_unchanged']=sha(SOURCE)==source_hash
assert all(checks.values())

def table(frame):return frame.to_html(index=False,border=0,float_format=lambda v:f'{v:,.2f}',classes='data')
overview=overall[['指标','单位','小冰期合计','现代合计','减少量合计','总量减少比例_pct','个体比例中位数_pct']].rename(columns={'总量减少比例_pct':'总量减少（%）','个体比例中位数_pct':'个体中位数（%）'})
mt=mount[['mountain_zh','systems','area_total_loss_pct','volume_signed_loss_sum','volume_total_loss_pct','stored_axis_total_loss_pct']].rename(columns={'mountain_zh':'山脉','systems':'数量','area_total_loss_pct':'面积减少（%）','volume_signed_loss_sum':'冰量净减少（km³）','volume_total_loss_pct':'冰量减少（%）','stored_axis_total_loss_pct':'参考轴缩短（%）'})
sz=size[['LIA_area_class_km2','systems','area_LIA_share_pct','volume_LIA_share_pct','area_total_loss_pct','volume_total_loss_pct']].rename(columns={'LIA_area_class_km2':'小冰期面积组（km²）','systems':'数量','area_LIA_share_pct':'初始面积占比（%）','volume_LIA_share_pct':'初始冰量占比（%）','area_total_loss_pct':'面积减少（%）','volume_total_loss_pct':'冰量减少（%）'})
doc=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>654套冰川基础分析</title><style>body{{margin:0;background:#f3f6f7;color:#223c48;font:16px/1.8 'Microsoft YaHei',sans-serif}}main{{max-width:1120px;background:white;margin:auto;padding:32px}}h1{{font-size:30px}}h2{{font-size:23px;border-top:1px solid #dfe7e9;padding-top:25px;margin-top:35px}}img{{width:100%;height:auto}}a{{color:#117589}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:9px 11px;border-bottom:1px solid #dce5e8;text-align:right}}th{{background:#edf3f5}}th:first-child,td:first-child{{text-align:left}}.scroll{{overflow-x:auto}}.muted{{color:#5c737e;font-size:14px}}.note{{background:#eef5f6;padding:16px 20px}}@media(max-width:700px){{main{{padding:16px}}}}</style></head><body><main>
<p class="muted">基础描述统计 · 2026-10-12 · 654套既定样本</p><h1>先看清楚：冰川有多少，变化了多少？</h1>
<p>本页只整理已有数据的总体、个体分布、山脉对比和规模分组。冰量与长度仍使用此前的654套结果；六套已剔除系统不恢复，所有增加值也保留。</p>
<p><a href="总体统计.csv">总体表</a> · <a href="山脉统计.csv">山脉表</a> · <a href="规模分组统计.csv">规模分组表</a> · <a href="654逐冰川基础表.csv">654套明细</a></p>
<h2>1. 总体变化：面积、冰量和参考轴分别看</h2><div class="scroll">{table(overview)}</div>
<p class="note"><strong>两个比例要分开：</strong>“总量减少比例”由全部小冰期总量和现代总量计算，大冰川权重更大；“个体比例中位数”描述样本中居中的一套冰川。面积对应4.61%和5.42%，模型冰量对应9.81%和10.74%。</p>
<img src="图1_总体与个体分布.png" alt="654套总体与个体分布"><p class="muted">652套面积减少、2套增加；618套模型冰量减少、36套增加；624套参考轴缩短、29套增长、1套数值上相等。模型增加值并不等于独立观测证实的增冰。</p>
<h2>2. 山脉对比：绝对变化与相对变化</h2><p>喀喇昆仑包含175套，是本样本数量最多的山脉区，模型冰量绝对减少100.76 km³；兴都库什相对减少15.09%，西昆仑为6.39%。这是当前选定样本的对比。</p>
<img src="图2_山脉对比.png" alt="各山脉的数量、冰量净减少与变化比例"><div class="scroll">{table(mt)}</div>
<h2>3. 规模分组：样本与冰量集中在哪里？</h2><p>654套中373套的小冰期面积为10–&lt;25 km²。面积≥100 km²的47套仅占系统数7.19%，却占初始面积39.77%、初始模型冰量51.61%；这组面积和冰量分别减少1.49%和6.14%。先将其作为样本特征记录，不将分组差别直接解释为因果机制。</p>
<img src="图3_规模分组.png" alt="五个小冰期面积组的数量、初始冰量份额及变化"><div class="scroll">{table(sz)}</div>
<h2>读图与数据口径</h2><p class="muted">所有变化定义为“小冰期减现代”，负值表示增加。面积为逐系统WGS84面积之和；冰量为已有、保留重叠修正的模型体积。参考轴是两期各自最长的已保存路径，不能直接称为真实末端退缩。当前654套分布在本地10区中的7区；青藏高原内部山地不全是唐古拉。现代轮廓年代为1993–2008年，小冰期年代尚未逐一统一，因此本页不计算年变化率。喜马拉雅245套及唐古拉、东天山候选编目不加入本页的654套总数。</p>
<p class="muted">以上为基础统计，未新运行重建模型。可下载三张图的矢量版：<a href="图1_总体与个体分布.pdf">总体图</a> · <a href="图2_山脉对比.pdf">山脉图</a> · <a href="图3_规模分组.pdf">规模图</a>。</p></main></body></html>'''
(OUT/'基础分析.html').write_text(doc,encoding='utf-8')
class Links(HTMLParser):
 def __init__(self):super().__init__();self.items=[]
 def handle_starttag(self,tag,attrs):
  self.items.extend(v for k,v in attrs if k in ('src','href'))
links=Links();links.feed(doc)
checks['all_report_links_exist']=all((OUT/p).is_file() for p in links.items)
assert all(checks.values())
summary={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'descriptive statistics only; frozen selected654; no new glacier model, climate prediction or causal attribution','source':str(SOURCE),'source_sha256':source_hash,'total':total,'size_bins_km2':LABELS,'bin_boundary_rule':'Left inclusive, right exclusive; display only, no selection','individual_quantiles':'linear interpolation','length_equal_tolerance_m':1e-6,'visual_review':'pending','checks':checks}
(OUT/'summary_and_QC.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
shutil.copy2(__file__,OUT/'reproduce_basic_analysis.py')
print(json.dumps({'status':'complete','checks_passed':sum(checks.values()),'systems':len(d),'figures':3,'tables':4,'output':str(OUT)},ensure_ascii=False))
