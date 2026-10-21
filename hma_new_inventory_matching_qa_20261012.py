"""Independent persisted-table and geometry checks; exact RGI7 identity supplement."""
from pathlib import Path
import json,hashlib,datetime
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.ops import unary_union
from shapely.ops import transform
from shapely.geometry.polygon import orient
from pyproj import Geod,Transformer
import networkx as nx

OUT=Path(r'H:/ToYangxin/result_yx/hma_harmonized_analysis_20261012_v1/new_inventory_matching')
ROOT=Path(r'E:/3spyderProject/1_LIA_himalaya_oggm')
reg=json.loads((OUT/'matching_operator_preregistration.json').read_text('utf-8'))
s=pd.read_csv(OUT/'system_matching.csv');m=pd.read_csv(OUT/'modern_members_all.csv');p=pd.read_csv(OUT/'positive_area_member_candidates.csv');c=pd.read_csv(OUT/'bipartite_identity_components.csv')
lg=gpd.read_file(OUT/'matching_geometry_derived.gpkg',layer='lia_systems').set_index('lia_system_id')
mg=gpd.read_file(OUT/'matching_geometry_derived.gpkg',layer='modern_members').set_index('rgi_id')
assert len(s)==1331 and len(m)==3586 and len(p)==3096
assert s.lia_system_id.is_unique and m.rgi_id.is_unique
assert not p[['lia_system_id','rgi_id']].duplicated().any()
assert p.intersection_area_equal_area_km2.gt(0).all()
assert p.intersection_area_geodesic_km2.ge(0).all()
geodesic_zero=p[p.intersection_area_geodesic_km2.eq(0)]
geodesic_zero.to_csv(OUT/'geodesic_area_numerical_zero_diagnostics.csv',index=False,encoding='utf-8-sig')
assert lg.geometry.is_valid.all() and mg.geometry.is_valid.all()
assert set(s.lia_system_id)==set(lg.index) and set(m.rgi_id)==set(mg.index)
counts=p.groupby('lia_system_id').size(); mc=p.groupby('rgi_id').size()
assert np.array_equal(s.set_index('lia_system_id').candidate_member_count,counts.reindex(s.lia_system_id,fill_value=0))
assert np.array_equal(m.set_index('rgi_id').candidate_lia_system_count,mc.reindex(m.rgi_id,fill_value=0))
ap=p[p.diagnostic_exclusive_assignment]
assert ap.rgi_id.is_unique and len(ap)==2060
assert set(ap.rgi_id)==set(p.rgi_id)
assert set(p.lia_system_id).issubset(set(s.lia_system_id))
for rid,g in p.groupby('rgi_id'):
    chosen=g[g.diagnostic_exclusive_assignment].iloc[0]
    assert chosen.intersection_area_geodesic_km2==g.intersection_area_geodesic_km2.max()
assert sum(c.lia_system_count)==1331 and sum(c.modern_member_count)==2060
max_diff=0
for r in p.itertuples():
    a=lg.loc[r.lia_system_id].geometry.intersection(mg.loc[r.rgi_id].geometry).area/1e6
    max_diff=max(max_diff,abs(a-r.intersection_area_equal_area_km2))
assert max_diff<1e-10
closures=[]
for r in s.itertuples():
    geom=lg.loc[r.lia_system_id].geometry
    ids=[] if pd.isna(r.candidate_member_ids) else r.candidate_member_ids.split(';')
    full=unary_union([mg.loc[x].geometry for x in ids])
    inside=geom.intersection(full);remain=geom.difference(full);outside=full.difference(geom)
    closures.extend([abs(geom.area-inside.area-remain.area)/1e6,abs(full.area-inside.area-outside.area)/1e6])
assert max(closures)<1e-6
source_unchanged=all(hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256'] for x in reg['source_files'])
assert source_unchanged

geo_transform=Transformer.from_crs('ESRI:102025',4326,always_xy=True).transform
geod=Geod(ellps='WGS84')
def ga(g):
    if g.is_empty:return 0.
    if g.geom_type=='Polygon':return abs(geod.geometry_area_perimeter(orient(transform(geo_transform,g),1))[0])/1e6
    if hasattr(g,'geoms'):return sum(ga(x) for x in g.geoms)
    return 0.
exclusive=[]
for r in s.itertuples():
    geom=lg.loc[r.lia_system_id].geometry
    ids=[] if pd.isna(r.exclusive_diagnostic_member_ids) else r.exclusive_diagnostic_member_ids.split(';')
    full=unary_union([mg.loc[x].geometry for x in ids]); ins=geom.intersection(full); outside=full.difference(geom)
    fa,ia,oa=ga(full),ga(ins),ga(outside)
    exclusive.append({'cohort':r.cohort,'lia_system_id':r.lia_system_id,'exclusive_diagnostic_member_count':len(ids),
       'exclusive_diagnostic_modern_union_area_geodesic_km2':fa,'exclusive_diagnostic_modern_inside_lia_area_geodesic_km2':ia,
       'exclusive_diagnostic_modern_outside_lia_area_geodesic_km2':oa,'exclusive_diagnostic_modern_containment_fraction':ia/fa if fa else None,
       'exclusive_diagnostic_lia_covered_by_modern_fraction':ia/r.lia_area_geodesic_km2,
       'assignment_status':'diagnostic_only_not_accepted_pairing'})
pd.DataFrame(exclusive).to_csv(OUT/'exclusive_assignment_containment_diagnostics.csv',index=False,encoding='utf-8-sig')
sensitivity=[];shared=[]
for cohort,sp in p.groupby('cohort'):
    systems=s[s.cohort.eq(cohort)].lia_system_id.tolist()
    for threshold in [0.,.05,.50]:
        take=sp if threshold==0 else sp[sp.intersection_fraction_of_modern_member>=threshold]
        G=nx.Graph();G.add_nodes_from(systems,bipartite=0)
        G.add_nodes_from(take.rgi_id.unique(),bipartite=1)
        G.add_edges_from(take[['lia_system_id','rgi_id']].itertuples(index=False,name=None))
        comps=list(nx.connected_components(G));types={};n_lia=set(systems)
        for cc in comps:
            nl=len(cc & n_lia);nm=len(cc)-nl
            typ='no-modern-candidate' if nm==0 else ('one-to-one' if nl==1 and nm==1 else ('one-LIA-to-many-modern' if nl==1 else ('many-LIA-to-one-modern' if nm==1 else 'many-to-many')))
            types[typ]=types.get(typ,0)+1
        sensitivity.append({'cohort':cohort,'minimum_intersection_fraction_of_modern_member':threshold,
          'threshold_rule':'strictly-positive-area' if threshold==0 else 'greater-than-or-equal',
          'candidate_pairs':len(take),'linked_lia_records':int(take.lia_system_id.nunique()),'linked_modern_members':int(take.rgi_id.nunique()),
          'shared_modern_members':int(take.groupby('rgi_id').size().gt(1).sum()),'graph_components':len(comps),
          **{f'components_{k}':v for k,v in types.items()},'use':'sensitivity_diagnostic_only'})
    for rid,g in sp.groupby('rgi_id'):
        if len(g)<=1:continue
        v=g.sort_values('intersection_fraction_of_modern_member',ascending=False)
        shared.append({'cohort':cohort,'rgi_id':rid,'candidate_lia_count':len(g),
          'best_lia_system_id':v.lia_system_id.iloc[0],'best_overlap_fraction_of_modern_member':v.intersection_fraction_of_modern_member.iloc[0],
          'runnerup_lia_system_id':v.lia_system_id.iloc[1],'runnerup_overlap_fraction_of_modern_member':v.intersection_fraction_of_modern_member.iloc[1],
          'runnerup_overlap_area_geodesic_km2':v.intersection_area_geodesic_km2.iloc[1],
          'candidate_edges_ge5percent':int(g.intersection_fraction_of_modern_member.ge(.05).sum()),
          'candidate_edges_ge50percent':int(g.intersection_fraction_of_modern_member.ge(.5).sum())})
pd.DataFrame(sensitivity).fillna(0).to_csv(OUT/'strong_intersection_graph_sensitivity.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(shared).to_csv(OUT/'shared_modern_member_diagnostics.csv',index=False,encoding='utf-8-sig')

# Exact IDs provide cross-source duplication evidence, not geometry identity or merging.
selected_path=Path(r'H:/ToYangxin/result_yx/selected_reconstruction_20261010_v1_654/selected_modern_member_native_product_baseline.csv')
selected=pd.read_csv(selected_path)
over=p.merge(selected[['rgi_id','system_record_id','lia_unique_id','mountain']],on='rgi_id',how='inner',validate='many_to_one')
over=over.rename(columns={'system_record_id':'selected654_system_record_id','lia_unique_id':'selected654_lia_unique_id','mountain':'selected654_mountain'})
over.to_csv(OUT/'exact_RGI7_overlap_with_selected654.csv',index=False,encoding='utf-8-sig')
extra=[]
for cohort,g in s.groupby('cohort'):
    mm=m[m.cohort.eq(cohort)]; ov=over[over.cohort.eq(cohort)]
    sh=pd.DataFrame(shared);sh=sh[sh.cohort.eq(cohort)]
    extra.append({'cohort':cohort,
      'exclusive_diagnostic_modern_union_ge10_km2_count':int(g.exclusive_diagnostic_modern_union_area_geodesic_km2.ge(10).sum()),
      'exclusive_diagnostic_max_member_ge10_km2_count':int(g.exclusive_diagnostic_maximum_member_area_geodesic_km2.ge(10).sum()),
      'LIA_ge10_and_exclusive_modern_union_ge10_km2_count':int((g.lia_area_geodesic_km2.ge(10)&g.exclusive_diagnostic_modern_union_area_geodesic_km2.ge(10)).sum()),
      'exclusive_diagnostic_modern_area_sum_km2':float(g.exclusive_diagnostic_modern_union_area_geodesic_km2.sum()),
      'shared_modern_members_best_intersection_fraction_below_0p5':int((mm.shared_across_lia_systems&mm.best_intersection_fraction_of_modern_member.lt(.5)).sum()),
      'shared_modern_members_runnerup_fraction_below_0p05':int(sh.runnerup_overlap_fraction_of_modern_member.lt(.05).sum()),
      'shared_modern_members_runnerup_fraction_below_0p50':int(sh.runnerup_overlap_fraction_of_modern_member.lt(.5).sum()),
      'new_lia_records_sharing_RGI7_with_selected654':int(ov.lia_system_id.nunique()),
      'selected654_records_sharing_RGI7_with_new_lia':int(ov.selected654_system_record_id.nunique()),
      'shared_unique_RGI7_members_with_selected654':int(ov.rgi_id.nunique())})
qa={'schema':'new-inventory-matching-independent-qa-v1','status':'pass','verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'source_files_unchanged':source_unchanged,'all_3096_pair_projected_intersections_recomputed':True,
    'maximum_persisted_pair_area_difference_km2':max_diff,'maximum_system_projected_closure_residual_km2':max(closures),
    'system_count':1331,'modern_pool_count':3586,'unique_matched_modern_members':2060,'pair_count':3096,
    'geodesic_area_numerical_zero_pair_count':len(geodesic_zero),
    'geodesic_zero_pair_max_projected_area_km2':float(geodesic_zero.intersection_area_equal_area_km2.max()) if len(geodesic_zero) else 0,
    'qa_assertion_correction':'Initial QA assumed all positive projected slivers resolve to positive geodesic area. One 5.92e-21 km2 projected sliver rounds to zero geodesic area. QA now checks preregistered positive projected candidate predicate; pair retained and explicit numerical-zero table exported. No matching recomputation or selection change.',
    'all_systems_preserved':True,'all_modern_pool_members_preserved':True,'unique_exclusive_diagnostic_assignment_verified':True,
    'exact_id_overlap_supplement_source':{'path':str(selected_path),'sha256':hashlib.sha256(selected_path.read_bytes()).hexdigest()},
    'diagnostic_supplement':extra,'strong_intersection_graph_sensitivity':sensitivity,
    'caveat':'Exclusive assignments, threshold counts and exact-RGI7-ID overlap are diagnostic; no acceptance, final large-glacier selection or source merging is applied.'}
(OUT/'independent_QC.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),'utf-8')
for src in [ROOT/'hma_new_inventory_matching_20261012.py',ROOT/'hma_new_inventory_matching_qa_20261012.py']:
    (OUT/src.name).write_bytes(src.read_bytes())
readme='''唐古拉和东天山LIA—RGI7空间配对审计（2026-10-12）\n\n此目录只完成可复核的几何候选配对，没有运行冰面或动力模型，也没有核验编目完成状态。\n唐古拉691条、东天山640条全部保留，二者源文件均无needxd字段。没有按现代最小面积删选。\n\n匹配算子事前记录于matching_operator_preregistration.json。所有真实正相交成员均保存在positive_area_member_candidates.csv；仅接触边界的对单独列出。\nsystem_matching.csv给出每个LIA记录的全部候选、全成员联合/内部/外部面积，以及现代最大相交排他归属诊断。\nmodern_members_all.csv包含本地现代池3586个唯一RGI7 ID，含未与LIA相交的现代记录。\nbipartite_identity_components.csv记录一对一、一对多、多对多关系；同一现代成员跨LIA轮廓时，全部候选全成员面积不可简单相加。\nexact_RGI7_overlap_with_selected654.csv记录新资料与已选654系统共享现代RGI7 ID的证据，不能直接视为同一LIA系统或合并。\n\n面积基于WGS84椭球；真实叠置使用ESRI:102025等积坐标，保存等积面积以检查闭合。源无效几何只在派生副本make_valid，修复前后面积差见derived_geometry_repair_audit.csv，源文件SHA全部不变。\nmatching_geometry_derived.gpkg是修复后的审计副本，不替换源编目。\n\n>=1、>=5、>=10 km²仅为规模诊断，尚未定义统一大冰川样本门槛。LIA>=10 km²计数：唐古拉40、东天山4；最大相交排他归属后现代联合面积>=10 km²：41和3。二者口径不同，不能互换。\n东天山13条LIA在本地现代池无正面积相交，不能仅据此宣布冰川消失。\n现代RGI7图形源年唐古拉1994—2007，东天山2001—2006，不代表2026年。\n\nindependent_QC.json重读所有表和GIS，重新测量3096对相交面积，核验1331条LIA/3586条现代/2060个被匹配现代成员及所有系统的面积闭合。summary.json为主结果，independent_QC.json含追加的精确ID跨源重叠与排他归属规模诊断。\n'''
(OUT/'README.txt').write_text(readme,'utf-8')
manifest=[{'file':x.name,'bytes':x.stat().st_size,'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in sorted(OUT.iterdir()) if x.is_file() and x.name!='file_manifest.json']
(OUT/'file_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),'utf-8')
print(json.dumps(qa,ensure_ascii=False,indent=2))
