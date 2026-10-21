"""Read-only source audit. Derived overlays are not reconstruction inputs."""
import os,json,hashlib,itertools,html,shutil
from pathlib import Path
import numpy as np,pandas as pd,geopandas as gpd
from shapely import make_valid
from shapely.geometry import Polygon,MultiPolygon,GeometryCollection
from shapely.geometry.polygon import orient
from pyproj import Geod
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path('E:/3spyderProject/1_LIA_himalaya_oggm')
OUT=Path('H:/ToYangxin/result_yx/hma_mountain_coverage_20261012_v1')
OUT.mkdir(parents=True,exist_ok=True)
BASE=ROOT/'Himalaya_greatGlacier/With_Rec_climate_organized/outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06'
NEW=Path('H:/ToYangxin/result_yx/selected_reconstruction_20261010_v1_654')
DATA=Path('D:/01 文件整理/04 数据')
GEOD=Geod(ellps='WGS84')
def area(g):
    if g is None or g.is_empty:return 0.
    if isinstance(g,Polygon):return abs(GEOD.geometry_area_perimeter(orient(g,sign=1))[0])/1e6
    if hasattr(g,'geoms'):return sum(area(x) for x in g.geoms)
    return 0.
def writejson(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cleanrecords(df):return json.loads(df.to_json(orient='records',force_ascii=False))

partition=DATA/'HMA_newregion/shp_wgs84/HMA_newregion_wgs84.shp'
regions=gpd.read_file(partition).to_crs(4326)
ZH={'Hindu Kush':'兴都库什','West Himalaya':'西喜马拉雅','East Himalaya':'东喜马拉雅','Central Himalaya':'中喜马拉雅','Karakoram':'喀喇昆仑','Western Tien Shan':'西天山','Western Kunlun Shan':'西昆仑','Nyainqentanglha':'念青唐古拉','Tibetan Interior Mountains':'青藏高原内部山地','Pamir':'帕米尔','Outside / unresolved':'10区范围外／归属待核'}
sources=[
 ('selected654','本次完成编目654',NEW/'RGI_box_selected654.shp','UniqueID'),
 ('legacy245','旧245系统编目',ROOT/'Himalaya_greatGlacier/Him_10_lia_with_large.shp','UniqueID'),
 ('tanggula691','唐古拉独立编目',DATA/'other_lia_shp/tanggula_complete/tanggula_LIA_with_UniqueId.shp','UniqueId'),
 ('east_tianshan640','东天山独立编目',DATA/'other_lia_shp/east_tianshan/east_tianshan_LIA_with_UniqueId.shp','UniqueId'),
]
oldmanifest=pd.read_csv(BASE/'system_manifest_work/system_manifest_245_v0.csv').set_index('lia_unique_id')
oldstatic=pd.read_csv(ROOT/'validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/lia_system_grid_results_v1.csv').set_index('lia_unique_id')
newtable=pd.read_csv(NEW/'selected_inventory_area_and_modern_volume.csv').set_index('lia_unique_id')
catalog=[];frames=[];repairs=[];checks=[]
for code,label,path,idcol in sources:
    raw=gpd.read_file(path);invalid=~raw.geometry.is_valid
    original=raw.to_crs(4326)
    derived=raw.copy();derived.geometry=derived.geometry.map(make_valid);derived=derived.to_crs(4326)
    after_transform_invalid=~derived.geometry.is_valid
    derived.geometry=derived.geometry.map(make_valid)
    ids=raw[idcol].astype(int)
    assert ids.is_unique,(code,'nonunique source ids')
    rows=[]
    for i,r in derived.iterrows():
        g=r.geometry;uid=int(r[idcol]);a=area(g)
        pieces=[(str(q.Primary_ID),area(g.intersection(q.geometry))) for _,q in regions.iloc[regions.sindex.query(g,predicate='intersects')].iterrows()]
        pieces=sorted(pieces,key=lambda z:z[1],reverse=True)
        best,f=(pieces[0][0],pieces[0][1]/a) if pieces and a>0 else ('Outside / unresolved',0.)
        assigned=best if f>=.5 else 'Outside / unresolved'
        frozen=str(r.mountain) if code=='selected654' else ''
        if code=='selected654':checks.append({'record_id':r.record_id,'frozen':frozen,'overlay':assigned,'match':frozen==assigned})
        region=frozen if frozen else assigned
        if invalid.iloc[i] or after_transform_invalid.iloc[i]:repairs.append({'cohort':code,'source_id':uid,'source_invalid':bool(invalid.iloc[i]),'post_transform_invalid':bool(after_transform_invalid.iloc[i]),'raw_geodesic_area_km2':area(original.geometry.iloc[i]),'derived_area_km2':a,'difference_km2':a-area(original.geometry.iloc[i])})
        sm=oldmanifest.loc[uid] if code=='legacy245' else None
        nt=newtable.loc[uid] if code=='selected654' else None
        static=code=='selected654' or (code=='legacy245' and uid in oldstatic.index)
        row={'cohort':code,'cohort_label':label,'source_id':uid,'record_key':f'{code}:{uid}','record_id':str(r.record_id) if code=='selected654' else '',
             'mountain10':region,'mountain_zh':ZH[region],'source_locality': '唐古拉' if code=='tanggula691' else ('东天山' if code=='east_tianshan640' else ''),
             'overlay_largest_region':best,'overlay_largest_fraction':f,'source_area_km2':a,
             'lia_area_ge10_km2':bool(a>=10),'source_invalid':bool(invalid.iloc[i]),'derived_repair_applied':bool(invalid.iloc[i] or after_transform_invalid.iloc[i]),
             'has_static_result':static,'static_status':('654冻结结果；模型冰量' if code=='selected654' else ('旧191几何闭合试验；方法待统一' if static else '本审计未核实到对应冰面结果')),
             'legacy_region_label':str(sm.region_label) if sm is not None else '',
             'modern_area_km2':float(nt.modern_area_geodesic_km2) if nt is not None else (float(sm.modern_union_area_km2) if sm is not None else None),
             'modern_id_version':'RGI7' if code=='selected654' else ('RGI6' if code=='legacy245' else 'RGI7文件存在；系统映射未核实'),
             'climate_mapping':'旧245登记存在；质量分层' if code=='legacy245' else '区域气候源存在；本审计未核实逐系统匹配',
             'hugonnet_valid_members':int(sm.hugonnet_valid_member_count) if sm is not None else None,
             'hugonnet_missing_members':int(sm.hugonnet_missing_member_count) if sm is not None else None,
             'timing_status':str(sm.timing_status) if sm is not None else '未核实逐冰川年代',
             'source_path':str(path),'geometry':g}
        rows.append(row)
    frame=gpd.GeoDataFrame(rows,crs=4326);frames.append(frame)
    sidecars=[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(path.parent.glob(path.stem+'.*')) if p.suffix in ['.shp','.shx','.dbf','.prj','.cpg']]
    catalog.append({'cohort':code,'label':label,'source_path':str(path),'n_source_records':len(raw),'source_id_unique':bool(ids.is_unique),'source_crs':str(raw.crs),'invalid_geometry_count':int(invalid.sum()),'derived_lia_area_km2':float(frame.source_area_km2.sum()),'n_lia_area_ge10_km2':int(frame.lia_area_ge10_km2.sum()),'static_result_count':int(frame.has_static_result.sum()),'sidecars':sidecars,'status':'冻结完成集合' if code=='selected654' else ('旧系统集合；191子集有试验重建' if code=='legacy245' else '已找到独立轮廓；完成状态、年代、现代配对尚待验证')})
allg=gpd.GeoDataFrame(pd.concat(frames,ignore_index=True),crs=4326)
allg.drop(columns='geometry').to_csv(OUT/'source_system_registry.csv',index=False,encoding='utf-8-sig')
allg.to_file(OUT/'coverage_geometry.gpkg',layer='source_systems',driver='GPKG')
regions.to_file(OUT/'coverage_geometry.gpkg',layer='mountain10',driver='GPKG')

# Cross-cohort intersections are review flags, not identity decisions.
pairrows=[]
for left,right in itertools.combinations(frames,2):
    for _,l in left.iterrows():
        for _,r in right.iloc[right.sindex.query(l.geometry,predicate='intersects')].iterrows():
            a=area(l.geometry.intersection(r.geometry))
            if a<=1e-8:continue
            fl=a/l.source_area_km2;fr=a/r.source_area_km2
            pairrows.append({'left_cohort':l.cohort,'left_key':l.record_key,'left_region':l.mountain10,'right_cohort':r.cohort,'right_key':r.record_key,'right_region':r.mountain10,'intersection_km2':a,'fraction_left':fl,'fraction_right':fr,'substantial_review_flag':max(fl,fr)>=.5,'interpretation':'空间相交候选；不是已确认同一冰川'})
pairs=pd.DataFrame(pairrows,columns=['left_cohort','left_key','left_region','right_cohort','right_key','right_region','intersection_km2','fraction_left','fraction_right','substantial_review_flag','interpretation'])
pairs['substantial_review_flag']=pairs['substantial_review_flag'].astype(bool)
pairs.to_csv(OUT/'cross_source_overlap_candidates.csv',index=False,encoding='utf-8-sig')
writejson('source_catalog.json',catalog)
writejson('geometry_repairs.json',repairs)
writejson('selected654_partition_crosscheck.json',checks)

summary=[]
for region in list(regions.Primary_ID)+['Outside / unresolved']:
    sub=allg[allg.mountain10==region]
    row={'mountain10':region,'mountain_zh':ZH[region]}
    for code,_,_,_ in sources:
        s=sub[sub.cohort==code]
        row[code+'_n']=len(s);row[code+'_lia_area_km2']=float(s.source_area_km2.sum());row[code+'_lia_ge10_n']=int(s.lia_area_ge10_km2.sum())
    row['legacy191_static_n']=int(sub[(sub.cohort=='legacy245')].has_static_result.sum())
    row['all_source_records_n']=len(sub)
    substantial=pairs[pairs.substantial_review_flag]
    keys=set(substantial.left_key)|set(substantial.right_key)
    row['records_with_substantial_cross_source_overlap']=int(sub.record_key.isin(keys).sum())
    row['interpretation']='来源记录数；跨源未去重；不作为独立冰川总数或总体覆盖率'
    summary.append(row)
summarydf=pd.DataFrame(summary)
summarydf.to_csv(OUT/'mountain10_coverage.csv',index=False,encoding='utf-8-sig')
pairsummary=[]
for (a,b),s in pairs.groupby(['left_cohort','right_cohort']):
    st=s[s.substantial_review_flag]
    pairsummary.append({'left_cohort':a,'right_cohort':b,'intersection_pairs':len(s),'substantial_pairs':len(st),'left_records_substantial':int(st.left_key.nunique()),'right_records_substantial':int(st.right_key.nunique())})
writejson('cross_source_overlap_summary.json',pairsummary)

# Inspect additional discoveries without declaring them accepted LIA observations.
supp=[]
for p in [DATA/'other_lia_shp/hma_model_result/HMA_remain_min0.10_LIA.shp',DATA/'other_lia_shp/tanggula_complete/Karakoram_Mountains_LIA_draft.shp',DATA/'other_lia_shp/Zhang_Tanggula_56box/Zhang_Tanggula_56box.shp']:
    d=gpd.read_file(p);rec={'path':str(p),'features':len(d),'columns':list(d.columns),'included_in_main_registry':False,'sha256_shp':sha(p)}
    for col in ['year','source','lia_status','lia_drawn']:
        if col in d:rec[col+'_counts']={str(k):int(v) for k,v in d[col].value_counts(dropna=False).items()}
    rec['reason']='框图不是冰川轮廓' if '56box' in p.name else ('草稿完成标记未满足已完成条件' if 'draft' in p.name else '模型产物；年代与独立编目属性未核实')
    supp.append(rec)
writejson('supplementary_discoveries.json',supp)

rules=[
 '主分区采用用户自定义10区；本地15区是RGI二级分区，不替代山脉分区。',
 '四个编目来源分别列示。2230是来源记录总数，不是去重后独立冰川数。',
 '654为冻结完成集合：needxd=1、已剔除与现代轮廓相同者及用户丢弃的6系统；本审计不重新筛选。',
 '旧191是旧245的子集，不能作为新增191条重复累加。其试验重建与654尚未完成方法统一，不能直接合并冰量。',
 '唐古拉691、东天山640是文件要素数，包含小冰川；文件存在不等于全部可进入大冰川重建。',
 'LIA面积≥10平方公里仅作大小分布诊断，不替代研究对象的正式筛选定义。',
 '新来源按轮廓与10区相交的最大椭球面积归属，最大占比至少50%；否则记为范围外／待核。654保留冻结标签并独立复核。',
 '无效源几何仅在派生覆盖审计副本中make_valid修复；原始数据未改，修复记录另存。',
 '跨源相交面积大于1e-8平方公里记为候选；相交占任一源至少50%仅作为重点核查标记，不自动合并。',
 '青藏高原内部山地不等于唐古拉；唐古拉按独立编目来源单列标签，不凭名称修改10区边界。',
 '本地10区名为Western Tien Shan的多边形实际也覆盖东天山来源；表中保留原标签，解释为该本地边界范围，不能按名称推断纯西天山。',
 '旧245的历史四组标签与实际10区空间归属不同：实际几何均落喜马拉雅三分区，旧组名只作来源字段保留。',
 'RGI6气候和物质平衡匹配不能自动迁移至RGI7；LIA起始年代也未统一，当前不计算统一年变化率。',
 '仅以本地资料覆盖审计讨论可行性，不把统计关联解释成气候因果贡献。'
]
payload={'title':'HMA山脉资料覆盖审计','date':'2026-10-12','summary':cleanrecords(summarydf),'systems':cleanrecords(allg.drop(columns='geometry')),'sources':catalog,'overlap_summary':pairsummary,'methods':rules,'supplementary':supp}
writejson('workbook_payload.json',payload)

colors=['#147d92','#ca6f23','#784bb1','#3b924e']
labels=['Selected 654','Legacy 245 (includes static 191)','Tanggula inventory (691 polygons)','East Tien Shan inventory (640 polygons)']
fig,ax=plt.subplots(figsize=(14,9),layout='constrained')
regions.plot(ax=ax,color='#eef1ef',edgecolor='#838e94',linewidth=.8,zorder=1)
for f,c in zip(frames,colors):
    f.plot(ax=ax,color=c,edgecolor=c,linewidth=.25,alpha=.85,zorder=3)
    pts=f.geometry.representative_point();ax.scatter(pts.x,pts.y,s=2,color=c,alpha=.55,zorder=2)
for _,r in regions.iterrows():
    p=r.geometry.representative_point();ax.text(p.x,p.y,r.shortname,ha='center',va='center',fontsize=10,fontweight='bold',color='#35444d',bbox={'facecolor':'white','edgecolor':'none','alpha':.75,'pad':1.5},zorder=4)
ax.text(93,44.2,'East Tien Shan inventory\n(inside the local WTS polygon)',fontsize=9,ha='center',color=colors[3])
ax.annotate('Tanggula source inventory\nwithin broader Interior region',xy=(92.2,33.1),xytext=(95,36),fontsize=9,color=colors[2],arrowprops={'arrowstyle':'->','color':colors[2]})
ax.set(xlim=(68,104),ylim=(25,46),xlabel='Longitude (°E)',ylabel='Latitude (°N)')
ax.grid(alpha=.18);ax.set_title('HMA LIA inventory coverage by mountain range',fontsize=17,loc='left',pad=16)
ax.legend(handles=[Line2D([0],[0],marker='o',linestyle='',color=c,label=l,markersize=6) for c,l in zip(colors,labels)],loc='lower left',framealpha=.95,fontsize=10)
fig.text(.01,-.015,'No cross-source outline intersections were found. Counts are not a harmonized large-glacier census. Boundary: HMA_newregion_wgs84.',fontsize=9)
fig.savefig(OUT/'HMA_mountain_inventory_coverage.png',dpi=300,bbox_inches='tight')
fig.savefig(OUT/'HMA_mountain_inventory_coverage.pdf',bbox_inches='tight')
plt.close(fig)
qc={'source_counts':{c:len(f) for (c,_,_,_),f in zip(sources,frames)},'source_records_total':len(allg),'source_record_keys_unique':bool(allg.record_key.is_unique),'mountain_summary_counts_reconcile':int(summarydf.all_source_records_n.sum())==len(allg),'selected654_partition_mismatches':[x for x in checks if not x['match']],'legacy_static_n':int(frames[1].has_static_result.sum()),'derived_geometry_invalid_count':int((~allg.is_valid).sum()),'derived_repair_count':len(repairs),'cross_source_overlap_summary':pairsummary,'warnings':['覆盖记录跨源尚未去重','唐古拉和东天山完成状态及现代配对尚待核实','重建产品方法未统一'],'simulations_run':False,'source_files_modified':False}
assert qc['source_counts']=={'selected654':654,'legacy245':245,'tanggula691':691,'east_tianshan640':640}
assert qc['source_record_keys_unique'] and qc['mountain_summary_counts_reconcile'] and qc['legacy_static_n']==191
assert not qc['selected654_partition_mismatches'] and qc['derived_geometry_invalid_count']==0
writejson('coverage_QC.json',qc)
shutil.copy2(__file__,OUT/'audit_reproduce.py')
print(json.dumps({'output':str(OUT),'regional':cleanrecords(summarydf[['mountain10','selected654_n','legacy245_n','legacy191_static_n','tanggula691_n','east_tianshan640_n']]),'sources':[{k:x[k] for k in ['cohort','n_source_records','n_lia_area_ge10_km2','invalid_geometry_count']} for x in catalog],'qc':qc},ensure_ascii=False,default=str))
