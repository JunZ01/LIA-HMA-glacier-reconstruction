"""Readonly polygon matching for two unreviewed LIA inventories.

The preregistered operator is saved before this driver executes. Outputs are
identity diagnostics, not a completed inventory or accepted reconstruction.
"""
from pathlib import Path
import json, hashlib, math, datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.geometry.polygon import orient
from shapely.ops import unary_union, transform
from pyproj import Geod, Transformer

ROOT=Path(r'E:/3spyderProject/1_LIA_himalaya_oggm')
BASE=Path(r'D:/01 文件整理/04 数据/other_lia_shp')
OUT=Path(r'H:/ToYangxin/result_yx/hma_harmonized_analysis_20261012_v1/new_inventory_matching')
REG=json.loads((OUT/'matching_operator_preregistration.json').read_text('utf-8'))
assert not (OUT/'summary.json').exists(), 'Do not overwrite completed result'
GEOD=Geod(ellps='WGS84')
TOGEO=Transformer.from_crs('ESRI:102025',4326,always_xy=True).transform
REPAIRS=[]

def polygon_parts(g):
    if g is None or g.is_empty:return []
    if g.geom_type=='Polygon': return [g]
    if hasattr(g,'geoms'):return [p for sub in g.geoms for p in polygon_parts(sub)]
    return []

def poly(g):
    ps=polygon_parts(g)
    if len(ps)==1:return ps[0]
    if not ps:return Polygon()
    return unary_union(ps)

def area(g):
    if g is None or g.is_empty:return 0.
    geo=transform(TOGEO,g)
    return sum(abs(GEOD.geometry_area_perimeter(orient(p,1))[0]) for p in polygon_parts(geo))/1e6

def components(g):return len(polygon_parts(g))
def union(geoms):return unary_union(geoms) if len(geoms) else Polygon()
def div(a,b):return a/b if b>0 else None

def prepare(path,cohort,role,idcol):
    g=gpd.read_file(path)
    assert g[idcol].notna().all() and g[idcol].is_unique
    for stage in ['source','overlay']:
        if stage=='overlay':g=g.to_crs('ESRI:102025')
        for i,geom in g.geometry.items():
            assert geom is not None and not geom.is_empty
            if geom.is_valid:continue
            fixed=poly(shapely.make_valid(geom))
            assert fixed.is_valid and not fixed.is_empty
            REPAIRS.append({'cohort':cohort,'role':role,'source_id':str(g.loc[i,idcol]),'stage':stage,
              'reason':shapely.is_valid_reason(geom),'source_crs':str(g.crs),'area_before_crs_units2':geom.area,
              'area_after_crs_units2':fixed.area,'area_difference_crs_units2':fixed.area-geom.area,
              'relative_area_difference':div(fixed.area-geom.area,geom.area),
              'parts_before':components(geom),'parts_after':components(fixed),
              'nonpolygon_parts_discarded':shapely.make_valid(geom).geom_type=='GeometryCollection'})
            g.at[i,'geometry']=fixed
        assert g.geometry.is_valid.all()
    return g.reset_index(drop=True)

def safe(v):
    if v is None:return None
    if isinstance(v,(np.integer,)):return int(v)
    if isinstance(v,(np.floating,)):return None if not np.isfinite(v) else float(v)
    return v

class UF:
    def __init__(self,n):self.p=list(range(n))
    def root(self,a):
        while self.p[a]!=a:
            self.p[a]=self.p[self.p[a]];a=self.p[a]
        return a
    def join(self,a,b):
        a,b=self.root(a),self.root(b)
        if a!=b:self.p[b]=a

SUMMARIES=[];ALLSYS=[];ALLMEM=[];ALLPAIR=[];COMP=[];CONTACT=[];LIA_GEOMS=[];MOD_GEOMS=[];QA=[]
configs=[('TANGGULA_LOCAL691','唐古拉', 'tanggula_complete/tanggula_LIA_with_UniqueId.shp','tanggula_complete/rgi7_tanggula_complete_glacier.shp',691),
         ('EAST_TIANSHAN_LOCAL640','东天山','east_tianshan/east_tianshan_LIA_with_UniqueId.shp','east_tianshan/rgi7_east_tianshan_glacier.shp',640)]
for cohort,label,lf,mf,expected in configs:
    lia=prepare(BASE/lf,cohort,'LIA','UniqueId'); mod=prepare(BASE/mf,cohort,'modern','rgi_id')
    assert len(lia)==expected
    lids=[f'{cohort}:{int(x):05d}' for x in lia.UniqueId]
    la=[area(g) for g in lia.geometry]; ma=[area(g) for g in mod.geometry]
    pairs=[];by_lia={i:[] for i in range(len(lia))};by_mod={j:[] for j in range(len(mod))}
    u=UF(len(lia)+len(mod)); idx=mod.sindex
    for i,geom in enumerate(lia.geometry):
        for j in idx.query(geom,predicate='intersects'):
            j=int(j);over=poly(geom.intersection(mod.geometry.iloc[j]))
            if over.area<=0:
                CONTACT.append({'cohort':cohort,'lia_system_id':lids[i],'rgi_id':mod.rgi_id.iloc[j]});continue
            inter=area(over)
            row={'cohort':cohort,'lia_system_id':lids[i],'lia_source_unique_id':int(lia.UniqueId.iloc[i]),'rgi_id':mod.rgi_id.iloc[j],
                 'lia_area_geodesic_km2':la[i],'modern_member_area_geodesic_km2':ma[j],
                 'intersection_area_geodesic_km2':inter,'intersection_area_equal_area_km2':over.area/1e6,
                 'intersection_fraction_of_modern_member':div(inter,ma[j]),'intersection_fraction_of_lia':div(inter,la[i]),'_i':i,'_j':j}
            pairs.append(row);by_lia[i].append(row);by_mod[j].append(row);u.join(i,len(lia)+j)
    # Exclusive assignment is diagnostic only: retain all positive-area pairs.
    for j,rows in by_mod.items():
        rows.sort(key=lambda r:(-r['intersection_area_geodesic_km2'],r['lia_system_id']))
        for rank,row in enumerate(rows):
            row['candidate_rank_for_modern_member']=rank+1
            row['diagnostic_exclusive_assignment']=rank==0
            row['modern_member_candidate_system_count']=len(rows)
            row['exact_maximum_intersection_tie']=len(rows)>1 and rows[0]['intersection_area_geodesic_km2']==rows[1]['intersection_area_geodesic_km2']
    roots={}
    for i in range(len(lia)):
        roots.setdefault(u.root(i),{'li':[],'mi':[]})['li'].append(i)
    for j in range(len(mod)):
        if by_mod[j]:roots[u.root(len(lia)+j)]['mi'].append(j)
    component_by_lia={}; component_by_mod={}
    for k,(rt,group) in enumerate(sorted(roots.items(),key=lambda r:min(lids[i] for i in r[1]['li']))):
        cid=f'{cohort}:G{k+1:05d}'; nl,nm=len(group['li']),len(group['mi'])
        typ='no-modern-candidate' if nm==0 else ('one-to-one' if nl==1 and nm==1 else ('one-LIA-to-many-modern' if nl==1 else ('many-LIA-to-one-modern' if nm==1 else 'many-to-many')))
        row={'cohort':cohort,'component_id':cid,'lia_system_count':nl,'modern_member_count':nm,'relationship':typ,
             'lia_system_ids':';'.join(lids[i] for i in group['li']),
             'rgi_ids':';'.join(mod.rgi_id.iloc[j] for j in group['mi'])}
        COMP.append(row)
        for i in group['li']:component_by_lia[i]=row
        for j in group['mi']:component_by_mod[j]=row
    sysrows=[];memberrows=[];max_closure=0;geod_closure=0
    for i,geom in enumerate(lia.geometry):
        rs=by_lia[i]; js=[r['_j'] for r in rs]; exclusive=[r['_j'] for r in rs if r['diagnostic_exclusive_assignment']]
        full=union([mod.geometry.iloc[j] for j in js]); assigned=union([mod.geometry.iloc[j] for j in exclusive])
        inside=poly(full.intersection(geom)); outside=poly(full.difference(geom)); deglaciated=poly(geom.difference(full))
        aa,ia,oa,da=area(full),area(inside),area(outside),area(deglaciated)
        closure=max(abs(full.area-inside.area-outside.area),abs(geom.area-inside.area-deglaciated.area))/1e6
        max_closure=max(max_closure,closure);geod_closure=max(geod_closure,abs(aa-ia-oa),abs(la[i]-ia-da))
        group=component_by_lia[i]
        r={'cohort':cohort,'region_label':label,'lia_system_id':lids[i],'lia_source_unique_id':int(lia.UniqueId.iloc[i]),
           'completion_status':'unverified_no_needxd_field','reconstruction_status':'not_reconstructed_in_this_audit',
           'candidate_member_count':len(js),'candidate_member_ids':';'.join(mod.rgi_id.iloc[j] for j in js),
           'exclusive_diagnostic_member_count':len(exclusive),'exclusive_diagnostic_member_ids':';'.join(mod.rgi_id.iloc[j] for j in exclusive),
           'shared_modern_member_count':sum(len(by_mod[j])>1 for j in js),'component_id':group['component_id'],'component_relationship':group['relationship'],
           'lia_area_geodesic_km2':la[i],'lia_area_equal_area_km2':geom.area/1e6,'lia_polygon_parts':components(geom),
           'candidate_modern_union_area_geodesic_km2':aa,'candidate_modern_sum_member_area_geodesic_km2':sum(ma[j] for j in js),
           'maximum_candidate_modern_member_area_geodesic_km2':max((ma[j] for j in js),default=0),
           'candidate_modern_inside_lia_area_geodesic_km2':ia,'candidate_modern_outside_lia_area_geodesic_km2':oa,
           'candidate_modern_containment_fraction':div(ia,aa),'lia_covered_by_modern_fraction':div(ia,la[i]),
           'candidate_lia_minus_modern_union_area_geodesic_km2':la[i]-aa,
           'candidate_area_reduction_relative_lia_percent':100*(la[i]-aa)/la[i],
           'deglaciated_lia_area_geodesic_km2':da,
           'candidate_full_modern_connected_components':components(full),'candidate_inside_modern_connected_components':components(inside),
           'multiple_modern_member_diagnostic':len(js)>1,
           'exclusive_diagnostic_modern_union_area_geodesic_km2':area(assigned),
           'exclusive_diagnostic_maximum_member_area_geodesic_km2':max((ma[j] for j in exclusive),default=0),
           'exact_planform_equal_to_candidate_union':geom.equals(full),'equal_area_closure_residual_km2':closure,
           'centroid_lon':transform(TOGEO,geom.centroid).x,'centroid_lat':transform(TOGEO,geom.centroid).y}
        for threshold in [1,5,10]:
            r[f'diagnostic_LIA_ge_{threshold}_km2']=la[i]>=threshold
            r[f'diagnostic_candidate_modern_union_ge_{threshold}_km2']=aa>=threshold
            r[f'diagnostic_maximum_modern_member_ge_{threshold}_km2']=r['maximum_candidate_modern_member_area_geodesic_km2']>=threshold
        sysrows.append(r);LIA_GEOMS.append({'cohort':cohort,'lia_system_id':lids[i],'original_unique_id':int(lia.UniqueId.iloc[i]),'geometry':geom})
    for j,geom in enumerate(mod.geometry):
        rs=by_mod[j]; lu=union([lia.geometry.iloc[r['_i']] for r in rs]); inside=poly(geom.intersection(lu))
        raw=mod.iloc[j].drop('geometry').to_dict()
        r={'cohort':cohort,**{k:safe(v) for k,v in raw.items()},'modern_member_area_geodesic_km2':ma[j],
           'modern_member_area_equal_area_km2':geom.area/1e6,'candidate_lia_system_count':len(rs),'candidate_lia_system_ids':';'.join(r['lia_system_id'] for r in rs),
           'exclusive_diagnostic_lia_system_id':rs[0]['lia_system_id'] if rs else None,
           'modern_inside_any_lia_area_geodesic_km2':area(inside),'modern_outside_all_lia_area_geodesic_km2':area(poly(geom.difference(lu))),
           'modern_inside_any_lia_fraction':div(area(inside),ma[j]),
           'best_intersection_fraction_of_modern_member':rs[0]['intersection_fraction_of_modern_member'] if rs else None,
           'shared_across_lia_systems':len(rs)>1,'component_id':component_by_mod[j]['component_id'] if rs else None,
           'source_rgi_area_minus_derived_area_km2':float(raw['area_km2'])-ma[j]}
        memberrows.append(r);MOD_GEOMS.append({'cohort':cohort,'rgi_id':raw['rgi_id'],'geometry':geom})
    df=pd.DataFrame(sysrows); md=pd.DataFrame(memberrows); pairdf=pd.DataFrame(pairs).drop(columns=['_i','_j'])
    all_lia_union=union(list(lia.geometry)); all_mod_union=union(list(mod.geometry)); matched=union([mod.geometry.iloc[j] for j in by_mod if by_mod[j]])
    subset=[c for c in COMP if c['cohort']==cohort]
    s={'cohort':cohort,'label':label,'lia_systems':len(lia),'modern_member_pool':len(mod),'modern_ids_unique':bool(mod.rgi_id.is_unique),
       'positive_area_candidate_pairs':len(pairs),'systems_with_candidates':int((df.candidate_member_count>0).sum()),
       'systems_without_candidates':int((df.candidate_member_count==0).sum()),'matched_unique_modern_members':int((md.candidate_lia_system_count>0).sum()),
       'unmatched_modern_pool_members':int((md.candidate_lia_system_count==0).sum()),'shared_modern_members':int((md.candidate_lia_system_count>1).sum()),
       'systems_with_shared_modern_members':int((df.shared_modern_member_count>0).sum()),
       'systems_with_multiple_modern_members':int((df.candidate_member_count>1).sum()),
       'systems_without_exclusive_diagnostic_members':int((df.exclusive_diagnostic_member_count==0).sum()),
       'graph_component_types':pd.Series([x['relationship'] for x in subset]).value_counts().to_dict(),
       'exact_equal_planforms':int(df.exact_planform_equal_to_candidate_union.sum()),
       'lia_sum_system_area_geodesic_km2':sum(la),'lia_union_area_geodesic_km2':area(all_lia_union),
       'all_modern_pool_union_area_geodesic_km2':area(all_mod_union),'matched_unique_modern_union_area_geodesic_km2':area(matched),
       'all_candidate_sum_system_modern_area_geodesic_km2':float(df.candidate_modern_union_area_geodesic_km2.sum()),
       'max_equal_area_closure_residual_km2':max_closure,'max_geodesic_overlay_closure_residual_km2':geod_closure,
       'diagnostic_threshold_counts':{k:int(df[k].sum()) for k in df if k.startswith('diagnostic_')},
       'modern_source_dates':sorted(str(x) for x in mod.src_date.unique()),
       'completion_verified':False,'needxd_field_available':False,'scientific_large_glacier_selection_applied':False}
    SUMMARIES.append(s);ALLSYS+=sysrows;ALLMEM+=memberrows;ALLPAIR+=pairdf.to_dict('records')
    QA.append({'cohort':cohort,'area_closure_equal_area_km2':max_closure,'area_closure_geodesic_km2':geod_closure})
    assert max_closure<1e-6
    print(json.dumps(s,ensure_ascii=False),flush=True)

tables={'system_matching':ALLSYS,'modern_members_all':ALLMEM,'positive_area_member_candidates':ALLPAIR,'bipartite_identity_components':COMP,
        'boundary_only_contacts':CONTACT,'derived_geometry_repair_audit':REPAIRS}
for name,rows in tables.items():
    if not rows:
        pd.DataFrame(columns=['cohort','lia_system_id','rgi_id']).to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
    else:pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
gpd.GeoDataFrame(LIA_GEOMS,crs='ESRI:102025').to_file(OUT/'matching_geometry_derived.gpkg',layer='lia_systems',driver='GPKG')
gpd.GeoDataFrame(MOD_GEOMS,crs='ESRI:102025').to_file(OUT/'matching_geometry_derived.gpkg',layer='modern_members',driver='GPKG')
source_unchanged=all(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in REG['source_files'])
assert source_unchanged
assert len(ALLSYS)==1331 and len({r['lia_system_id'] for r in ALLSYS})==1331
assert len({r['rgi_id'] for r in ALLMEM})==len(ALLMEM), 'Cross-cohort duplicate RGI7 ID needs review'
summary={'schema':'new-lia-rgi7-spatial-matching-result-v1','finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'cohorts':SUMMARIES,'source_files_unchanged':source_unchanged,'total_lia_source_records':len(ALLSYS),
         'total_unique_modern_member_pool':len(ALLMEM),'total_positive_area_candidate_pairs':len(ALLPAIR),
         'geometry_repairs':len(REPAIRS),'source_completion_flags_verified':False,'simulations_launched':0,
         'selection_applied':False,'preregistration_sha256':hashlib.sha256((OUT/'matching_operator_preregistration.json').read_bytes()).hexdigest(),
         'notes':['Matching is candidate identity evidence only, not needxd completion verification.',
                  'Many-to-many candidate system totals double-count shared modern members; use member ledger for unique-area budgets.',
                  'The exclusive largest-intersection partition is diagnostic only; inspect shared and low-containment matches before model preparation.',
                  'Threshold counts are illustrative diagnostics and do not define the final large-glacier sample.',
                  'Source date belongs to RGI2000-v7 geometry, not current 2026.',
                  'Geodesic intersection plus difference has small chord/projection closure residual; equal-area overlay is independently exact within stated numerical residual.']}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=safe,allow_nan=False),'utf-8')
print('COMPLETE',OUT)
