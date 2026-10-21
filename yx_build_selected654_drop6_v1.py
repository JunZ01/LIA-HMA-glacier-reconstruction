from pathlib import Path
import json, csv, shutil, hashlib, html
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC=Path('H:/ToYangxin/result_yx/completed_reconstruction_20261009_v4_654')
OUT=Path('H:/ToYangxin/result_yx/selected_reconstruction_20261010_v1_654')
EXCLUDED={'YX-03055','YX-03621','YX-05103','YX-05176','YX-14077','YX-28236'}
assert not OUT.exists()
OUT.mkdir()
used={}
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(name):
    p=SRC/name; used[str(p)]=digest(p)
    return pd.read_csv(p,encoding='utf-8-sig')
def write(df,name):df.to_csv(OUT/name,index=False,encoding='utf-8-sig')
def save(name,j):(OUT/name).write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')

status=read('all_660_system_status.csv')
selected=status[~status.record_id.isin(EXCLUDED)].copy()
assert len(selected)==654 and selected.record_id.is_unique
assert set(status.record_id)-set(selected.record_id)==EXCLUDED
assert selected.status.eq('reconstructed-QC-passed').all()
write(selected,'selected_system_status.csv')
mapping=read('all_662_source_record_mapping.csv')
selected_mapping=mapping[mapping.system_record_id.isin(set(selected.record_id))].copy()
removed_mapping=mapping[~mapping.system_record_id.isin(set(selected.record_id))].copy()
assert len(selected_mapping)==655 and len(removed_mapping)==7
assert set(removed_mapping.source_record_id)==EXCLUDED|{'YX-29305'}
assert selected_mapping.source_record_id.is_unique
write(selected_mapping,'selected_source_record_mapping.csv')
write(removed_mapping,'excluded_source_records.csv')
excluded=read('quality_analysis/terminal_unestimated6_source_and_model_review.csv')
assert set(excluded.record_id)==EXCLUDED
excluded['exclusion_decision']='2026-10-10 user: 丢弃这6条冰川'
write(excluded,'excluded_systems.csv')

paired=read('paired_volume_changes.csv')
assert set(paired.record_id)==set(selected.record_id)
# Preserve every paired value and its serialization exactly.
shutil.copy2(SRC/'paired_volume_changes.csv',OUT/'paired_volume_changes.csv')
assert digest(SRC/'paired_volume_changes.csv')==digest(OUT/'paired_volume_changes.csv')
baseline=read('full_inventory_baseline/all660_inventory_area_and_modern_volume.csv')
area=baseline[baseline.record_id.isin(set(selected.record_id))].copy()
lookup=paired.set_index('record_id')
area['LIA_reconstructed_volume_km3']=area.record_id.map(lookup.lia_volume_deduplicated_km3)
area['reconstructed_LIA_to_modern_volume_change_km3']=area.record_id.map(lookup.volume_loss_deduplicated_km3)
write(area,'selected_inventory_area_and_modern_volume.csv')
members=read('full_inventory_baseline/all1263_modern_member_native_product_baseline.csv')
members=members[members.system_record_id.isin(set(selected.record_id))].copy()
assert len(members)==1256 and members.rgi_id.is_unique
write(members,'selected_modern_member_native_product_baseline.csv')

for name in ['array_and_volume_QC.csv','common_grid_bed_alignment_diagnostics.csv',
             'common_ice_surface_change_statistics.csv','gis_raster_index.csv',
             'modern_product_support.csv','modern_surface_spatial_diagnostics.csv',
             'modern_lia_geometry_consistency.csv','thickness_change_raster_QC.csv',
             'all_model_result_lineage.csv','joint_endpoint_groups.csv']:
    frame=read(name); key='record_id' if 'record_id' in frame else 'system_record_id'
    frame=frame[frame[key].isin(set(selected.record_id))].copy()
    write(frame,name)
    if name=='gis_raster_index.csv':
        assert set(frame.record_id)==set(selected.record_id)
        assert all(Path(p).is_file() for p in frame.delivery_path)
for name in ['overlap_duplicate_volume_correction.json','new_model_supplements_provenance.json']:
    used[str(SRC/name)]=digest(SRC/name); shutil.copy2(SRC/name,OUT/name)

original_summary=json.loads((SRC/'summary.json').read_text(encoding='utf-8'))
used[str(SRC/'summary.json')]=digest(SRC/'summary.json')
mountain_order=list(read('local_10_mountain_summary.csv').mountain)
region_rows=[]
for mountain in mountain_order:
    ar=area[area.mountain==mountain]; pv=paired[paired.mountain==mountain]
    has=len(ar)>0
    la=ar.lia_area_geodesic_km2.sum() if has else np.nan
    ma=ar.modern_area_geodesic_km2.sum() if has else np.nan
    lv=pv.lia_volume_deduplicated_km3.sum() if has else np.nan
    mv=pv.modern_volume_km3.sum() if has else np.nan
    region_rows.append(dict(mountain=mountain,selected_systems=len(ar),
        selected_source_records=int(ar.source_record_count.sum()),reconstructed_systems=len(pv),
        excluded_systems_from_original=int((excluded.mountain==mountain).sum()),
        lia_area_geodesic_km2=la,modern_area_geodesic_km2=ma,
        area_loss_km2=la-ma,area_loss_percent=(la-ma)/la*100 if has else np.nan,
        lia_volume_km3=lv,modern_volume_km3=mv,volume_loss_km3=lv-mv,
        volume_loss_percent=(lv-mv)/lv*100 if has else np.nan,
        modern_F19_native_supported_volume_km3=ar.modern_F19_native_supported_volume_km3.sum() if has else np.nan,
        no_selected_sample=not has,sample_note='selected654 after user exclusion' if has else 'no selected sample'))
regions=pd.DataFrame(region_rows);write(regions,'local_10_mountain_summary.csv')
lia_area=float(area.lia_area_geodesic_km2.sum()); modern_area=float(area.modern_area_geodesic_km2.sum())
native=float(area.modern_F19_native_supported_volume_km3.sum())
summary=dict(status='user-selected654-all-retained-models-independently-verified',
    source_records=655,systems=654,reconstructed_QC_pass=654,review_systems=0,
    excluded_systems=sorted(EXCLUDED),excluded_system_count=6,excluded_source_record_count=7,
    original_assessed_systems=660,original_changed_source_records=662,
    selection='needxd=1;102 unchanged records excluded; user then excludes six held systems incl joint05176+29305',
    source_records_are_not_unique_systems=True,
    lia_area_geodesic_km2=lia_area,modern_area_geodesic_km2=modern_area,
    area_loss_km2=lia_area-modern_area,area_loss_percent=(lia_area-modern_area)/lia_area*100,
    area_definition='sum of WGS84 ellipsoidal endpoint areas; prior small overlap treatment unchanged',
    lia_volume_deduplicated_km3=original_summary['lia_volume_deduplicated_km3'],
    modern_volume_km3=original_summary['modern_volume_km3'],
    volume_loss_km3=original_summary['volume_loss_km3'],volume_loss_percent=original_summary['volume_loss_percent'],
    modern_members_unique=1256,modern_F19_native_supported_volume_km3=native,
    modern_F19_missing_member_count=int(area.modern_F19_missing_or_unrepresented_member_count.sum()),
    modern_F19_missing_member_area_km2=float(area.modern_F19_missing_or_unrepresented_member_area_km2.sum()),
    modern_F19_systems_with_missing_members=int((area.modern_F19_missing_or_unrepresented_member_count>0).sum()),
    modeled_volumes_unchanged=True,paired_table_byte_identical=True,model_recalculations=0,
    original_inventory_or_delivery_modified=False,reference_delivery=str(SRC),outputs_directory=str(OUT))
for field in ['lia_volume_km3','modern_volume_km3','volume_loss_km3']:
    key='lia_volume_deduplicated_km3' if field=='lia_volume_km3' else field
    assert abs(regions[field].sum()-summary[key])<1e-8
assert regions.selected_systems.sum()==654 and regions.selected_source_records.sum()==655
assert abs(members.far_volume_km3.sum()-native)<1e-8
save('summary.json',summary)

gpkg=SRC/'reconstruction_results.gpkg';used[str(gpkg)]=digest(gpkg)
lia=gpd.read_file(gpkg,layer='lia_systems_and_volume_changes')
lia=lia[lia.record_id.isin(set(selected.record_id))].copy()
modern=gpd.read_file(gpkg,layer='modern_system_endpoints')
modern=modern[modern.record_id.isin(set(selected.record_id))].copy()
mountains=gpd.read_file(gpkg,layer='local_mountains_10')
assert len(lia)==len(modern)==654
dest=OUT/'selected_reconstruction_results.gpkg'
lia.to_file(dest,layer='lia_selected654',driver='GPKG')
modern.to_file(dest,layer='modern_selected654',driver='GPKG')
mountains.to_file(dest,layer='local_mountains_10',driver='GPKG')
shape=lia[['lia_unique_id','record_id','source_record_ids','mountain','geometry']].copy()
shape=shape.rename(columns={'lia_unique_id':'UniqueID','source_record_ids':'source_ids'})
shape['needxd']=1
arlook=area.set_index('record_id')
shape['LIA_km2']=shape.record_id.map(arlook.lia_area_geodesic_km2)
shape['MOD_km2']=shape.record_id.map(arlook.modern_area_geodesic_km2)
shape['V_LIA_km3']=shape.record_id.map(lookup.lia_volume_deduplicated_km3)
shape['V_MOD_km3']=shape.record_id.map(lookup.modern_volume_km3)
shape['DV_km3']=shape.record_id.map(lookup.volume_loss_deduplicated_km3)
shape['DV_pct']=shape.record_id.map(lookup.volume_loss_deduplicated_percent)
shape.to_file(OUT/'RGI_box_selected654.shp',driver='ESRI Shapefile',encoding='UTF-8')

plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False
cn={'Hindu Kush':'兴都库什','Karakoram':'喀喇昆仑','Western Tien Shan':'西天山',
    'Western Kunlun Shan':'西昆仑','Nyainqentanglha':'念青唐古拉',
    'Tibetan Interior Mountains':'西藏内陆山脉','Pamir':'帕米尔'}
valid=regions[~regions.no_selected_sample].sort_values('volume_loss_percent')
fig,axs=plt.subplots(1,2,figsize=(13,5.5))
ys=np.arange(len(valid))
for ax,key,label in [(axs[0],'area_loss_percent','面积减少（%）'),(axs[1],'volume_loss_percent','模型冰量减少（%）')]:
    ax.barh(ys,valid[key],color='#327fa7' if key.startswith('area') else '#43a28a')
    ax.set_yticks(ys,[cn.get(m,m) for m in valid.mountain]);ax.set_xlabel(label)
    ax.spines[['top','right']].set_visible(False)
    for y,v in zip(ys,valid[key]):ax.text(v+.15,y,f'{v:.2f}',va='center',fontsize=9)
    ax.set_xlim(0,valid[key].max()*1.16)
fig.suptitle('用户剔除6个系统后的654系统样本 · 本地山脉面积与冰量变化')
fig.tight_layout();fig.savefig(OUT/'selected_mountain_changes.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(12,6.5))
mountains.boundary.plot(ax=ax,color='#9da9b5',linewidth=.6)
display=lia.copy();display.geometry=display.geometry.simplify(.001,preserve_topology=True)
display.plot(ax=ax,column='volume_loss_deduplicated_percent',cmap='RdYlBu_r',vmin=-10,vmax=40,
    legend=True,legend_kwds={'label':'模型冰量减少（%）','shrink':.6},linewidth=0)
ax.set_xlim(65,107);ax.set_ylim(26,46);ax.set_xlabel('经度');ax.set_ylabel('纬度')
ax.set_title('当前入选654系统 · 6个被排除系统不在图层中')
fig.tight_layout();fig.savefig(OUT/'selected_reconstruction_map.png',dpi=160);plt.close(fig)

quality=(SRC/'quality_analysis/report.html').as_uri()
upstream=(SRC/'completion_overview.html').as_uri()
table=regions[['mountain','selected_systems','selected_source_records','lia_area_geodesic_km2',
    'modern_area_geodesic_km2','area_loss_percent','lia_volume_km3','modern_volume_km3','volume_loss_percent']].copy()
table['mountain']=table.mountain.map(lambda x:cn.get(x,x))
table.columns=['山脉','系统数','源记录数','LIA面积(km²)','现代面积(km²)','面积减少(%)','LIA冰量(km³)','现代冰量(km³)','冰量减少(%)']
body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>排除6系统后的654系统成果</title>
<style>body{{font-family:Microsoft YaHei,Arial,sans-serif;max-width:1100px;margin:36px auto;padding:0 18px;line-height:1.8;color:#203047;background:#f5f7fa}}main{{background:white;padding:30px;border-radius:10px}}h1{{font-size:26px}}a{{color:#0369a1}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #ddd;padding:8px;text-align:left}}.scroll{{overflow-x:auto}}.note{{background:#eef4f8;padding:14px;border-radius:6px}}</style>
<main><h1>当前分析样本：654个冰川系统</h1>
<p>按2026年10月10日用户要求，排除YX-03055、03621、05103、05176、14077、28236。05176与29305为同一联合系统，一并排除；原662条变化源记录因此保留655条，对应654个系统。原编目、旧报告和失败证据均保留。</p>
<table><tr><th>当前654系统</th><th>小冰期</th><th>现代</th><th>减少</th></tr>
<tr><td>面积</td><td>{lia_area:,.2f} km²</td><td>{modern_area:,.2f} km²</td><td>{lia_area-modern_area:,.2f} km²（{summary['area_loss_percent']:.2f}%）</td></tr>
<tr><td>模型冰量</td><td>{summary['lia_volume_deduplicated_km3']:,.2f} km³</td><td>{summary['modern_volume_km3']:,.2f} km³</td><td>{summary['volume_loss_km3']:,.2f} km³（{summary['volume_loss_percent']:.2f}%）</td></tr></table>
<p class="note">当前面积与模型冰量均使用同一654系统。所有保留系统已有核验结果，此次只改变分析范围，模型和栅格未重算。排除依据为用户对6个未通过质量条件系统的处理决定；结果对应筛选后的变化大冰川样本。现代冰量闭合属于标定，冰量不是直接观测，未计算统一年变化率或正式置信区间。</p>
<h2>可用成果</h2><ul><li><a href="RGI_box_selected654.shp">654系统小冰期边界Shapefile</a>（使用时保留同目录.shx/.dbf/.prj/.cpg配套文件）</li>
<li><a href="selected_reconstruction_results.gpkg">654系统小冰期、现代边界及10山脉GIS图层</a></li>
<li><a href="selected_inventory_area_and_modern_volume.csv">逐系统面积与原生现代资料基线</a></li>
<li><a href="paired_volume_changes.csv">逐系统重建冰量变化</a></li>
<li><a href="local_10_mountain_summary.csv">10山脉统一样本汇总</a></li>
<li><a href="selected_source_record_mapping.csv">655条源记录至654系统映射</a></li>
<li><a href="gis_raster_index.csv">已核验冰面、冰厚及变化栅格位置</a></li>
<li><a href="common_ice_surface_change_statistics.csv">共同有冰区域冰面变化</a></li>
<li><a href="excluded_systems.csv">排除的6系统及既有原因</a></li>
<li><a href="excluded_source_records.csv">排除的7条源记录</a></li>
<li><a href="{html.escape(quality,quote=True)}">已有654重建质量证据报告（含原660系统筛选背景）</a></li>
<li><a href="{html.escape(upstream,quote=True)}">原660评估范围的历史成果入口</a></li></ul>
<h2>10山脉统计</h2><p>7个山脉含保留样本。西、中、东喜马拉雅3个分区无入选样本，统计保持空值。</p><div class="scroll">{table.to_html(index=False,na_rep='—',float_format=lambda x:f'{x:.2f}')}</div>
<img src="selected_mountain_changes.png"><img src="selected_reconstruction_map.png">
<p>原生F19已支持现代体积合计{native:,.6f} km³；与现代模型体积的差别来自标定残差。1256个现代成员中18个缺少原生冰厚支持，涉及14系统，保持原资料缺口披露，不把缺口解释为零冰量。当前选取不改变既有模型资料、床面或重建方法。</p></main></html>'''
(OUT/'report.html').write_text(body,encoding='utf-8')
(OUT/'README.txt').write_text('当前用户选择：654系统/655源记录。排除6系统、7源记录。\n原始RGI_box.shp和旧封存包未修改。RGI_box_selected654.shp是已核验联合系统几何的导出，不是删除原编目记录。\n全部模型表值不变，冰面/冰厚栅格沿用并引用原verified654包；gis_raster_index.csv提供绝对路径。\n面积和冰量均使用新654系统样本；源记录655包含仍保留的联合系统11977+29307。\n原native冰厚缺口和模型估计限制保留，10山脉中3喜马拉雅分区无样本。\n',encoding='utf-8')
save('selection_provenance.json',dict(user_instruction='丢弃这6条冰川',effective_date='2026-10-10',
    excluded_systems=sorted(EXCLUDED),excluded_source_records=sorted(set(removed_mapping.source_record_id)),
    prior_deliveries_and_raw_inventory_modified=False,model_runs=0,
    retained_rasters_reference_directory=str(SRC/'rasters'),used_input_sha256=used))
print(json.dumps(summary,ensure_ascii=False,indent=2))
