from pathlib import Path
import json, hashlib, html, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely import wkt
from shapely.ops import transform as transform_geometry, substring
from pyproj import Transformer

OUT=Path('H:/ToYangxin/result_yx/length_change_selected654_20261010_v1')
SELECTED=Path('H:/ToYangxin/result_yx/selected_reconstruction_20261010_v1_654')
MEASURE=Path(os.environ.get('YX_LENGTH_MEASUREMENT_DIR',str(OUT/'geometry_measurement')))
measure=MEASURE/'system_measurements654.csv'
d=pd.read_csv(measure)
assert len(d)==654 and d.record_id.is_unique
paired=pd.read_csv(SELECTED/'paired_volume_changes.csv')
area=pd.read_csv(SELECTED/'selected_inventory_area_and_modern_volume.csv')
assert set(d.record_id)==set(paired.record_id)
assert not set(d.record_id)&{'YX-03055','YX-03621','YX-05103','YX-05176','YX-14077','YX-28236'}
extra=['record_id','mountain','lia_volume_deduplicated_km3','modern_volume_km3','volume_loss_deduplicated_percent']
d=d.drop(columns=['mountain'],errors='ignore').merge(paired[extra],on='record_id',validate='one_to_one')
d=d.merge(area[['record_id','lia_area_geodesic_km2','modern_area_geodesic_km2','area_decrease_relative_LIA_percent']],on='record_id',validate='one_to_one')
primary=d[d.primary_unambiguous_status.astype(str).str.lower().eq('true')].copy()
assert primary.primary_terminal_uncovered_m.notna().all()
primary['reference_axis_terminal_segment_km']=primary.primary_terminal_uncovered_m/1000
d.to_csv(OUT/'length_diagnostics_selected654.csv',index=False,encoding='utf-8-sig')
primary.to_csv(OUT/'primary_unambiguous_reference_axis_subset.csv',index=False,encoding='utf-8-sig')
primary_ids=set(primary.record_id)
status_cn={
    'primary_unambiguous_reference_axis_terminal_gap':'明确单轴端部空段',
    'true_LIA_vector_mismatch_review':'参考轴与矢量边界不完全重合',
    'fragmented_modern_axis_coverage_review':'现代轴上存在多段或断裂',
    'measurement_error_review':'测量待核',
    'stored_tail_censored_review':'现代覆盖到保存轴尾，变化被截断',
    'multiple_component_or_root_diagnostic':'多组件或多出口，仅作诊断',
    'candidate_directed_graph_diagnostic':'候选多出口图来源，仅作诊断',
    'no_modern_axis_intersection_review':'现代轮廓与参考轴不相交',
    'modern_head_not_covered_review':'缺少共同上游参考点',
    'non_simple_axis_review':'参考轴存在重复或交叉'}
regions=[]
order=pd.read_csv(SELECTED/'local_10_mountain_summary.csv').mountain.tolist()
for mountain in order:
    allrows=d[d.mountain==mountain]
    sub=primary[primary.mountain==mountain]
    values=sub.primary_terminal_uncovered_m
    regions.append(dict(mountain=mountain,selected_systems=len(allrows),primary_unambiguous_systems=len(sub),
        diagnostic_or_censored_systems=len(allrows)-len(sub),
        primary_reference_terminal_segment_mean_m=values.mean(),
        primary_reference_terminal_segment_median_m=values.median(),
        primary_reference_terminal_segment_p25_m=values.quantile(.25),
        primary_reference_terminal_segment_p75_m=values.quantile(.75),
        primary_reference_terminal_segment_max_m=values.max(),
        interpretation='fixed stored-axis terminal uncovered length in explicit positive, uncensored subset; not whole-glacier length or population mean'))
regions=pd.DataFrame(regions);regions.to_csv(OUT/'local10_length_diagnostic_summary.csv',index=False,encoding='utf-8-sig')
statuses=d.groupby(['source_family','length_diagnostic_status'],dropna=False).size().reset_index(name='systems')
statuses.to_csv(OUT/'source_family_and_length_QC_counts.csv',index=False,encoding='utf-8-sig')
v=primary.primary_terminal_uncovered_m
summary={'status':'computed-fixed-reference-axis-geometric-diagnostics',
    'selected_systems':654,'primary_unambiguous_systems':len(primary),
    'diagnostic_or_censored_systems':654-len(primary),
    'primary_mean_terminal_uncovered_m':float(v.mean()) if len(v) else None,
    'primary_median_terminal_uncovered_m':float(v.median()) if len(v) else None,
    'primary_p25_terminal_uncovered_m':float(v.quantile(.25)) if len(v) else None,
    'primary_p75_terminal_uncovered_m':float(v.quantile(.75)) if len(v) else None,
    'primary_max_terminal_uncovered_m':float(v.max()) if len(v) else None,
    'primary_sample_is_positive_uncensored_geometrically_unambiguous_subset':True,
    'not_mean_retreat_or_absolute_glacier_length_for_all654':True,
    'measurement_is_horizontal_planform_arc_length_in_projected_meters':True,
    'source_family_counts':{str(k):int(x) for k,x in d.source_family.value_counts(dropna=False).items()},
    'status_counts':{str(k):int(x) for k,x in d.length_diagnostic_status.value_counts(dropna=False).items()},
    'models_recomputed':0,'head_or_tail_extensions':0,'geometry_gate_or_volume_changes':0,
    'annual_length_change_rate_or_formal_CI_computed':False,
    'root_axis_dataset':'source_axes/root_axes.csv','geometry_measurement':str(measure),
    'prior_selected_package_unchanged':True}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

polygons=gpd.read_file(SELECTED/'selected_reconstruction_results.gpkg',layer='lia_selected654')
polygons=polygons[['record_id','geometry']].merge(d,on='record_id',validate='one_to_one')
polygons.to_file(OUT/'length_diagnostics.gpkg',layer='system_length_diagnostics654',driver='GPKG')
axes=gpd.read_file(OUT/'source_axes/axes.gpkg',layer='root_axes_wgs84')
roots=pd.read_csv(MEASURE/'root_measurements.csv')
key='root_key' if 'root_key' in roots else 'reference_root_key'
roots=roots.rename(columns={key:'root_key'})
props=[c for c in ['root_key','terminal_uncovered_m','modern_axis_total_covered_m',
        'length_diagnostic_status','primary_unambiguous_status'] if c in roots]
drop=[c for c in props if c!='root_key' and c in axes]
axes=axes.drop(columns=drop).merge(roots[props],on='root_key',validate='one_to_one')
axes.to_file(OUT/'length_diagnostics.gpkg',layer='root_axis_measurements695',driver='GPKG')

plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False
cn={'Hindu Kush':'兴都库什','Karakoram':'喀喇昆仑','Western Tien Shan':'西天山','Western Kunlun Shan':'西昆仑',
    'Nyainqentanglha':'念青唐古拉','Tibetan Interior Mountains':'西藏内陆山脉','Pamir':'帕米尔',
    'West Himalaya':'西喜马拉雅','Central Himalaya':'中喜马拉雅','East Himalaya':'东喜马拉雅'}
fig,axs=plt.subplots(1,2,figsize=(12,5))
if len(v):
    axs[0].hist(v/1000,bins=25,color='#2f7caa',edgecolor='white')
    axs[0].axvline(v.median()/1000,color='#db7a2d',linestyle='--',label=f'中位数 {v.median()/1000:.3f} km')
    axs[0].legend();axs[0].set_xlabel('参考主轴末端空段（km）');axs[0].set_ylabel('系统数')
    valid=regions[regions.primary_unambiguous_systems>0].sort_values('primary_reference_terminal_segment_median_m')
    y=np.arange(len(valid));axs[1].barh(y,valid.primary_reference_terminal_segment_median_m/1000,color='#43a28a')
    axs[1].set_yticks(y,[f'{cn.get(r.mountain,r.mountain)}（n={int(r.primary_unambiguous_systems)}）' for r in valid.itertuples()])
    axs[1].set_xlabel('明确子样本中位末端空段（km）')
for ax in axs:ax.spines[['top','right']].set_visible(False)
fig.suptitle(f'固定小冰期参考轴：{len(primary)}/654个系统为明确正空段子样本；其余单列诊断')
fig.tight_layout();fig.savefig(OUT/'reference_axis_terminal_segment_distribution.png',dpi=160);plt.close(fig)

source_axes=pd.read_csv(OUT/'source_axes/root_axes.csv')
modern_polys=gpd.read_file(SELECTED/'selected_reconstruction_results.gpkg',layer='modern_selected654').set_index('record_id')
lia_polys=polygons.set_index('record_id')
fig,axs=plt.subplots(2,2,figsize=(13,10))
for ax,rid in zip(axs.flat,['YX-00006','YX-25361','YX-04312','YX-11977']):
    row=source_axes[(source_axes.record_id==rid)&source_axes.is_system_reference.astype(str).str.lower().eq('true')].iloc[0]
    line=wkt.loads(row.geometry_native_wkt)
    transformer=Transformer.from_crs('EPSG:4326',row.native_crs_wkt,always_xy=True)
    lp=transform_geometry(transformer.transform,lia_polys.loc[rid].geometry)
    mp=transform_geometry(transformer.transform,modern_polys.loc[rid].geometry)
    gpd.GeoSeries([lp]).boundary.plot(ax=ax,color='#9baaba',linewidth=.8)
    gpd.GeoSeries([mp]).boundary.plot(ax=ax,color='#227ca5',linewidth=.8)
    gpd.GeoSeries([line]).plot(ax=ax,color='#d08036',linewidth=1.2)
    info=d.set_index('record_id').loc[rid]
    if pd.notna(info.modern_last_s_m) and info.modern_last_s_m<line.length:
        seg=substring(line,float(info.modern_last_s_m),line.length)
        if seg.geom_type=='LineString':gpd.GeoSeries([seg]).plot(ax=ax,color='#bc3d42',linewidth=3)
    ax.scatter([line.coords[0][0]],[line.coords[0][1]],color='#3f9c73',s=20)
    ax.set_title(f'{rid}\n{status_cn.get(info.length_diagnostic_status,info.length_diagnostic_status)}',fontsize=10)
    ax.set_aspect('equal');ax.ticklabel_format(style='plain',useOffset=False)
    ax.set_xlabel('原米制投影X(m)');ax.set_ylabel('Y(m)')
fig.suptitle('灰：LIA边界；蓝：现代边界；橙：固定参考轴；红：末端空段；绿：保存轴头',fontsize=11)
fig.tight_layout();fig.savefig(OUT/'same_axis_boundary_examples.png',dpi=150);plt.close(fig)

table=regions.copy();table['mountain']=table.mountain.map(lambda s:cn.get(s,s))
table=table[['mountain','selected_systems','primary_unambiguous_systems','diagnostic_or_censored_systems',
    'primary_reference_terminal_segment_mean_m','primary_reference_terminal_segment_median_m',
    'primary_reference_terminal_segment_p25_m','primary_reference_terminal_segment_p75_m']]
table.columns=['山脉','保留系统','明确子样本','诊断/截断','平均末段(m)','中位末段(m)','P25(m)','P75(m)']
def number(x):return f'{x:,.2f}' if x is not None else '暂无'
examples=d[d.record_id.isin(['YX-00006','YX-00080','YX-00134','YX-25361','YX-04312','YX-10304','YX-11977','YX-21852'])]
examples=examples[[c for c in ['record_id','reference_axis_length_m','modern_axis_total_covered_m','terminal_uncovered_m','length_diagnostic_status','primary_unambiguous_status'] if c in examples]].copy()
examples['length_diagnostic_status']=examples.length_diagnostic_status.map(lambda s:status_cn.get(s,s))
examples['primary_unambiguous_status']=examples.primary_unambiguous_status.map(lambda v:'是' if str(v).lower()=='true' else '否')
examples=examples.rename(columns={'record_id':'系统','reference_axis_length_m':'保存主轴长度(m)',
    'modern_axis_total_covered_m':'现代覆盖主轴长度(m)','terminal_uncovered_m':'末端空段(m)',
    'length_diagnostic_status':'长度诊断状态','primary_unambiguous_status':'属于明确子样本'})
status_table=d.length_diagnostic_status.value_counts().rename_axis('状态').reset_index(name='系统数')
status_table['状态']=status_table['状态'].map(lambda s:status_cn.get(s,s))
refs='<a href="https://docs.oggm.org/en/stable/generated/oggm.tasks.initialize_flowlines.html">OGGM初始化流线说明</a> · <a href="https://docs.oggm.org/en/stable/generated/oggm.global_tasks.write_centerlines_to_shape.html">OGGM中心线导出与边界匹配参数</a>'
report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>654系统长度变化：固定主轴几何诊断</title>
<style>body{{font-family:Microsoft YaHei,Arial,sans-serif;max-width:1100px;margin:35px auto;padding:0 18px;line-height:1.85;color:#203047;background:#f6f8fa}}main{{background:white;padding:30px;border-radius:10px}}h1{{font-size:26px}}a{{color:#0369a1}}img{{width:100%}}table{{border-collapse:collapse;min-width:850px;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}.scroll{{overflow-x:auto}}.call{{background:#eef5f9;padding:15px;border-radius:6px}}</style>
<main><h1>654系统长度补充：固定参考主轴的端部变化</h1>
<p>当前样本保持654个系统（655条源记录），此前排除的6个系统未重新纳入。本次只使用已保存的路线和真实两期轮廓，未重建冰面或运行冰川模型。</p>
<div class="call">{len(primary)}个系统符合预先定义的明确正空段条件。其参考主轴末端空段平均{number(summary['primary_mean_terminal_uncovered_m'])} m，中位{number(summary['primary_median_terminal_uncovered_m'])} m，P25–P75为{number(summary['primary_p25_terminal_uncovered_m'])}–{number(summary['primary_p75_terminal_uncovered_m'])} m。这些是明确子样本的几何指标，不能称全部654系统的平均真实退缩或完整冰川长度。</div>
<h2>长度口径</h2><p>对小冰期保存网络的每个出口，从源头到根出口拼接完整路径，保留实际连接段；同一出口选择最长路径，系统代表轴按小冰期路径长度确定，现代轮廓不参与选轴。格点坐标转为原米制投影后，将同一轴分别与两期真实轮廓相交，保留所有沿程区间。本指标是平面沿程，不是三维冰面坡长。参考主轴末端空段=保存轴长度−现代轮廓最后交点沿程。轴上现代覆盖长度另列，不把多段覆盖之和称末端变化。</p>
<p>650个系统使用原初始化路线来源，4个使用候选多出口图来源，分开记录。已有流线可能裁剪源头、结束于矢量边界之前，连接路径或源DEM也存在限制。本次不恢复源头、不延长尾部，因此不输出完整实测两期冰川长度；现代覆盖到保存轴尾的系统标为截断，不能把0空段解释为无变化或前进。</p>
<p>明确子样本要求：原路线来源、单组件/出口、轴简单且整体在小冰期矢量轮廓内、现代仅有一个连续交集且覆盖同一轴头、现代末端位于保存轴尾之前。多出口、多段交集、缺共同轴头、离轴以及矢量/格点差异全部列出。另{654-len(primary)}个系统保留诊断记录，而非再次剔除出654样本。长度年度率和正式置信区间未计算。</p>
<div class="scroll">{status_table.to_html(index=False)}</div>
<h2>山脉统计</h2><div class="scroll">{table.to_html(index=False,na_rep='—',float_format=lambda x:f'{x:.2f}')}</div>
<img src="reference_axis_terminal_segment_distribution.png">
<h2>典型系统与异常</h2><div class="scroll">{examples.to_html(index=False,na_rep='—',float_format=lambda x:f'{x:.3f}')}</div>
<p>04312有多个出口和现代轴内断段，其短末端空段不能单独解释为整个系统退缩距离；25361显示完整连接段必须计入参考轴。实际状态以表和逐区间数据为准。</p><img src="same_axis_boundary_examples.png">
<h2>成果与复核</h2><ul><li><a href="length_diagnostics_selected654.csv">654系统完整长度诊断与面积/冰量配对表</a></li>
<li><a href="primary_unambiguous_reference_axis_subset.csv">明确正空段子样本</a></li>
<li><a href="local10_length_diagnostic_summary.csv">10山脉长度诊断汇总</a></li>
<li><a href="source_family_and_length_QC_counts.csv">来源和长度质量状态数量</a></li>
<li><a href="source_axes/root_axes.csv">各出口完整参考轴及来源</a></li>
<li><a href="source_axes/axes.gpkg">参考轴GIS图层</a></li>
<li><a href="length_diagnostics.gpkg">654系统及全部出口长度诊断GIS</a></li>
<li><a href="{html.escape((MEASURE/'root_measurements.csv').relative_to(OUT).as_posix(),quote=True)}">各出口交点与长度测量</a></li>
<li><a href="{html.escape((SELECTED/'report.html').as_uri(),quote=True)}">654系统面积、冰面与冰量原成果</a></li></ul>
<p>方法背景：{refs}。官方说明初始化会改变几何流线并裁剪支流末段；本项目具体裁剪、源头和连接长度以保存源数据的审计为准。</p></main></html>'''
(OUT/'report.html').write_text(report,encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
