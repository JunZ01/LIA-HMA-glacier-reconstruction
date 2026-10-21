"""Read-only harmonization of existing LIA inventory endpoints; no model imports."""
from pathlib import Path
from collections import Counter, defaultdict
import json, hashlib, sys, subprocess
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely import make_valid
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from pyproj import Geod

ROOT=Path(__file__).resolve().parent
OUT=Path('H:/ToYangxin/result_yx/hma_harmonized_analysis_20261012_v1/modern_identity')
SEL=Path('H:/ToYangxin/result_yx/selected_reconstruction_20261010_v1_654')
RAW=Path('H:/ToYangxin/result_yx/surface_reconstruction_20261009_v4')
OLD=ROOT/'Himalaya_greatGlacier'
BASE=OLD/'With_Rec_climate_organized/outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06'
G=Geod(ellps='WGS84')
SOURCES={}
REPAIRS=[]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):
    p=Path(p); SOURCES[str(p)]=sha(p)
    if p.suffix=='.shp':
        for ext in ['.dbf','.shx','.prj','.cpg']:
            f=p.with_suffix(ext)
            if f.exists(): SOURCES[str(f)]=sha(f)
    return p
def poly(g):
    if g.is_empty:return Polygon()
    if isinstance(g,(Polygon,MultiPolygon)):return g
    parts=[]
    for x in g.geoms:
        q=poly(x) if hasattr(x,'geoms') or isinstance(x,Polygon) else Polygon()
        if isinstance(q,Polygon) and not q.is_empty:parts.append(q)
        elif isinstance(q,MultiPolygon):parts.extend(q.geoms)
    return unary_union(parts) if parts else Polygon()
def area(g):
    if g.is_empty:return 0.
    if isinstance(g,Polygon):return abs(G.geometry_area_perimeter(orient(g,1))[0])/1e6
    if hasattr(g,'geoms'):return sum(area(p) for p in g.geoms)
    return 0.
def load_geo(path,key,layer=None):
    d=gpd.read_file(bind(path),layer=layer)
    for i,row in d.iterrows():
        q=row.geometry
        if not q.is_valid:
            r=poly(make_valid(q));REPAIRS.append(dict(source=str(path),layer=layer,key=str(row[key]),phase='native_before_reprojection',native_crs=str(d.crs),old_native_area=float(q.area),new_native_area=float(r.area),absolute_native_area_delta=float(abs(r.area-q.area)),old_type=q.geom_type,new_type=r.geom_type,valid_after=bool(r.is_valid)));d.at[i,'geometry']=r
    d=d.to_crs(4326)
    for i,row in d.iterrows():
        q=row.geometry
        if not q.is_valid:
            r=poly(make_valid(q));REPAIRS.append(dict(source=str(path),layer=layer,key=str(row[key]),phase='geographic_after_reprojection',old_geodesic_area_km2=area(q),new_geodesic_area_km2=area(r),old_type=q.geom_type,new_type=r.geom_type,valid_after=bool(r.is_valid)));d.at[i,'geometry']=r
    assert d.geometry.is_valid.all()
    return d
def ids(x):return set() if pd.isna(x) or x=='' else set(str(x).split(';'))
def jdump(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def csv(name,rows,columns=None):
    d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows,columns=columns)
    d.to_csv(OUT/name,index=False,encoding='utf-8-sig');return d
def overlaps(left,right,lkey,rkey):
    rows=[];si=right.sindex
    for _,a in left.iterrows():
        for j in si.query(a.geometry,predicate='intersects'):
            b=right.iloc[j];inter=area(a.geometry.intersection(b.geometry))
            if inter>0:rows.append({lkey:str(a[lkey]),rkey:str(b[rkey]),'intersection_km2':inter,'fraction_left':inter/area(a.geometry),'fraction_right':inter/area(b.geometry)})
    return rows
def internal_overlaps(d,key):
    rows=[];si=d.sindex
    for i,a in d.iterrows():
        for j in si.query(a.geometry,predicate='intersects'):
            if j<=i:continue
            b=d.iloc[j];inter=area(a.geometry.intersection(b.geometry))
            if inter>0:rows.append({'left_id':str(a[key]),'right_id':str(b[key]),'intersection_km2':inter})
    return rows

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'summary.json').exists():raise RuntimeError('Immutable final output exists; create a new version before rerunning.')
    canonical={}
    for fn in ['LIA_PROJECT_FACTS.md','LIA_EXECUTION_CONTRACT.md','LIA_CURRENT_STATE.json']:
        p=ROOT/fn;t=p.read_text(encoding='utf-8-sig');canonical[fn]={'sha256':sha(p),'bytes':p.stat().st_size,'read_complete_characters':len(t)}
    cp=subprocess.run([sys.executable,str(OLD/'With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py')],capture_output=True,text=True,encoding='utf-8')
    state=json.loads(cp.stdout);assert cp.returncode==0 and state['status']=='pass'
    canonical['validator']={k:state[k] for k in ['status','launch_authorized','counts','failures']};jdump('canonical_read_snapshot.json',canonical)
    lia=load_geo(OLD/'Him_10_lia_with_large.shp','UniqueID').sort_values('UniqueID').reset_index(drop=True)
    oldorig=load_geo(OLD/'Him_10_rgi_related_lia.shp','RGIId').reset_index(drop=True)
    oldfull=load_geo(BASE/'velocity_work/target_rgi60_1205_esri102027.gpkg','RGIId').reset_index(drop=True)
    selg=load_geo(SEL/'selected_reconstruction_results.gpkg','record_id','modern_selected654').reset_index(drop=True)
    newraw=load_geo(RAW/'modern_rgi7_members.gpkg','rgi_id')
    baseline=pd.read_csv(bind(SEL/'selected_modern_member_native_product_baseline.csv'))
    new=newraw[newraw.rgi_id.isin(baseline.rgi_id)].copy().reset_index(drop=True)
    new=new.merge(baseline[['rgi_id','system_record_id','member_area_geodesic_km2']],on='rgi_id',validate='one_to_one')
    manifest=pd.read_csv(bind(BASE/'system_manifest_work/system_manifest_245_v0.csv')).set_index('lia_unique_id')
    mapping=pd.read_json(bind(BASE/'thickness_work/mapping_records.json'))
    registry=pd.read_csv(bind(Path('H:/ToYangxin/result_yx/hma_mountain_coverage_20261012_v1/source_system_registry.csv')))
    oldreg=registry[registry.cohort=='legacy245'].copy()
    oldreg['sid']=oldreg.source_id.astype(int);oldreg=oldreg.set_index('sid')
    print('Loaded',len(lia),len(oldorig),len(oldfull),len(new),flush=True)
    assert len(new)==1256 and new.rgi_id.is_unique and baseline.rgi_id.is_unique
    assert oldorig.RGIId.is_unique and oldfull.RGIId.is_unique
    assert len(mapping)==1205 and mapping.member_rgi_id.is_unique
    assert set(new.rgi_id)==set(baseline.rgi_id)
    om=oldorig.set_index('RGIId');fm=oldfull.set_index('RGIId')
    rows=[];member_rows=[];differences=[];scenario_geom={k:[] for k in ['original_1159','confirmed_1196','download_1205_including_candidates']}
    for _,r in lia.iterrows():
        sid=int(r.UniqueID);m=manifest.loc[sid];original=ids(r.AllRGIs);full=ids(m.download_member_rgi_ids);candidates=ids(m.candidate_member_rgi_ids);confirmed=full-candidates
        assert original.issubset(full)
        if original!=full:
            differences.append({'lia_unique_id':sid,'original_allrgis_character_length':len(r.AllRGIs),'dbf_allrgis_width':254,'original_count':len(original),'source_declared_RGICount':int(r.RGICount),'confirmed_count':len(confirmed),'download_count':len(full),'added_confirmed':';'.join(sorted(confirmed-original)),'added_candidate':';'.join(sorted(candidates-original)),'removed_members':''})
        for mid in sorted(full):
            mm=mapping[mapping.member_rgi_id==mid].iloc[0]
            member_rows.append({'cohort':'legacy245','lia_unique_id':sid,'system_record_id':f'LEGACY-{sid}','rgi_version':6,'rgi_id':mid,'in_original_allrgis':mid in original,'confirmed':mid in confirmed,'candidate':mid in candidates,'membership_certainty':mm.membership_certainty,'relationship_basis':mm.relationship_basis,'mapping_status':mm.mapping_status,'area_geodesic_km2':area(fm.loc[mid].geometry),'modern_date':str(fm.loc[mid].BgnDate)})
        for scenario,sids,geomsource in [('original_1159',original,om),('confirmed_1196',confirmed,fm),('download_1205_including_candidates',full,fm)]:
            gs=[geomsource.loc[x].geometry for x in sorted(sids)];u=poly(unary_union(gs));a=area(r.geometry);ma=area(u);inside=area(u.intersection(r.geometry));outside=area(u.difference(r.geometry))
            rows.append({'cohort':'legacy245','lia_unique_id':sid,'scenario':scenario,'mountain10':oldreg.loc[sid].mountain10,'mountain_zh':oldreg.loc[sid].mountain_zh,'member_count':len(sids),'candidate_count':len(sids&candidates),'lia_area_geodesic_km2':a,'modern_union_area_geodesic_km2':ma,'modern_members_summed_area_geodesic_km2':sum(area(x) for x in gs),'modern_inside_lia_geodesic_km2':inside,'modern_outside_lia_geodesic_km2':outside,'lia_minus_modern_signed_area_km2':a-ma,'area_reduction_relative_lia_pct':100*(a-ma)/a,'modern_containment_fraction':inside/ma,'intersection_closure_error_km2':ma-inside-outside,'original_declared_count':int(r.RGICount),'modern_member_ids':';'.join(sorted(sids)),'source_epoch_min':min(str(geomsource.loc[x].BgnDate) for x in sids),'source_epoch_max':max(str(geomsource.loc[x].BgnDate) for x in sids),'individual_LIA_date_known':False})
            scenario_geom[scenario].append({'lia_unique_id':sid,'scenario':scenario,'geometry':u})
    for _,r in new.iterrows():
        member_rows.append({'cohort':'selected654','lia_unique_id':int(r.lia_unique_id),'system_record_id':r.system_record_id,'rgi_version':7,'rgi_id':r.rgi_id,'in_original_allrgis':None,'confirmed':True,'candidate':False,'membership_certainty':'frozen_selected654_modern_member','relationship_basis':'saved_selected654_baseline','mapping_status':'unchanged','area_geodesic_km2':area(r.geometry),'modern_date':r.src_date})
    results=csv('legacy245_endpoint_area_scenarios.csv',rows)
    csv('all2461_modern_member_registry.csv',member_rows)
    csv('legacy_membership_definition_differences.csv',differences)
    mountain=results.groupby(['scenario','mountain10','mountain_zh'],as_index=False).agg(systems=('lia_unique_id','size'),member_count=('member_count','sum'),lia_area_geodesic_km2=('lia_area_geodesic_km2','sum'),modern_union_area_geodesic_km2=('modern_union_area_geodesic_km2','sum'),area_reduction_km2=('lia_minus_modern_signed_area_km2','sum'),modern_outside_lia_geodesic_km2=('modern_outside_lia_geodesic_km2','sum'))
    mountain['area_reduction_relative_lia_pct']=100*mountain.area_reduction_km2/mountain.lia_area_geodesic_km2
    csv('legacy245_mountain_area_scenarios.csv',mountain)
    selectedchecks=[]
    for _,r in selg.iterrows():
        m=new[new.system_record_id==r.record_id];u=unary_union(m.geometry);selectedchecks.append({'record_id':r.record_id,'member_count':len(m),'saved_endpoint_area_km2':area(r.geometry),'members_union_area_km2':area(u),'absolute_area_difference_km2':abs(area(u)-area(r.geometry)),'symmetric_difference_km2':area(u.symmetric_difference(r.geometry))})
    csv('selected654_member_union_crosscheck.csv',selectedchecks)
    new2=new.rename(columns={'rgi_id':'rgi7_id'});old2=oldfull.rename(columns={'RGIId':'rgi6_id'})
    xp=overlaps(new2,old2,'rgi7_id','rgi6_id')
    csv('cross_version_member_geometry_intersections.csv',xp,['rgi7_id','rgi6_id','intersection_km2','fraction_left','fraction_right'])
    sp=[]
    for scenario,geoms in scenario_geom.items():
        d=gpd.GeoDataFrame(geoms,crs=4326).rename(columns={'lia_unique_id':'legacy_lia_unique_id'})
        x=overlaps(selg,d,'record_id','legacy_lia_unique_id')
        for r in x:r['legacy_scenario']=scenario
        sp.extend(x)
    csv('cross_cohort_modern_system_intersections.csv',sp,['record_id','legacy_lia_unique_id','intersection_km2','fraction_left','fraction_right','legacy_scenario'])
    same6=[]
    for mid in oldorig.RGIId:
        a=om.loc[mid].geometry;b=fm.loc[mid].geometry;same6.append({'rgi6_id':mid,'original_area_km2':area(a),'target_area_km2':area(b),'absolute_area_difference_km2':abs(area(a)-area(b)),'symmetric_difference_km2':area(a.symmetric_difference(b))})
    csv('rgi6_original1159_vs_target1205_geometry_check.csv',same6)
    internal={}
    for name,d,key in [('selected654_modern_members',new,'rgi_id'),('legacy1205_modern_members',oldfull,'RGIId')]:
        x=internal_overlaps(d,key);csv(name+'_internal_intersections.csv',x,['left_id','right_id','intersection_km2']);internal[name]={'positive_area_pairs':len(x),'summed_pair_intersection_km2':sum(a['intersection_km2'] for a in x),'pairs_above_one_square_metre':sum(a['intersection_km2']>1e-6 for a in x)}
    totals={}
    for scenario,d in results.groupby('scenario'):
        a=d.lia_area_geodesic_km2.sum();ma=d.modern_union_area_geodesic_km2.sum();totals[scenario]={'systems':len(d),'modern_members':int(d.member_count.sum()),'lia_area_km2':float(a),'modern_area_km2':float(ma),'signed_area_reduction_km2':float(a-ma),'area_reduction_relative_lia_pct':float(100*(a-ma)/a),'modern_outside_lia_km2':float(d.modern_outside_lia_geodesic_km2.sum()),'negative_signed_area_reduction_systems':int((d.lia_minus_modern_signed_area_km2<0).sum())}
    assert len(results)==735 and results.lia_unique_id.nunique()==245
    assert set(results.groupby('scenario').member_count.sum())=={1159,1196,1205}
    assert results.intersection_closure_error_km2.abs().max()<1e-3
    assert max(x['absolute_area_difference_km2'] for x in selectedchecks)<1e-4
    source_unchanged={p:sha(p)==s for p,s in SOURCES.items()};assert all(source_unchanged.values())
    jdump('geometry_repairs.json',REPAIRS);jdump('input_sha256.json',SOURCES)
    summary={'status':'complete-readonly-geometry-and-membership-audit','model_runs':0,'new_system_selection':False,'source_files_unchanged':all(source_unchanged.values()),'area_definition':'WGS84 ellipsoidal area of oriented polygons; summed per-system union areas; signed loss=(LIA-modern)/LIA; no annualization','repair_policy':'make_valid only on derived copies at native CRS and after reprojection; polygonal parts preserved, original source unchanged','cohorts':{'selected654_systems':654,'selected654_unique_modern_RGI7_members':1256,'legacy245_systems':245,'legacy_original_members':1159,'legacy_confirmed_members':1196,'legacy_download_members_including_candidates':1205,'legacy_candidate_members':9,'legacy_candidate_system':2692},'cross_version_positive_area_member_intersections':len(xp),'cross_cohort_positive_area_system_intersections':len(sp),'cross_version_identity_conclusion':('No positive-area overlap between the 1256 selected RGI7 and any of the 1205 legacy RGI6 footprints. These two bounded frozen cohorts have no shared modern ice footprint. No ID-number cross-version matching or global RGI6/7 split/merge crosswalk inferred.' if not xp else 'Positive-area cross-version overlap exists; inspect table before combining cohorts.'),'allrgis_dbf_field_width':254,'legacy_membership_differences':differences,'legacy_area_scenario_totals':totals,'internal_member_overlap_diagnostics':internal,'selected654_member_area_max_difference_vs_frozen_km2':float(np.max(np.abs(new.geometry.map(area)-new.member_area_geodesic_km2))),'selected654_union_max_symmetric_difference_km2':max(x['symmetric_difference_km2'] for x in selectedchecks),'old_same_RGI6_source_max_area_difference_km2':max(x['absolute_area_difference_km2'] for x in same6),'old_same_RGI6_source_max_symmetric_difference_km2':max(x['symmetric_difference_km2'] for x in same6),'area_inside_outside_max_closure_error_km2':float(results.intersection_closure_error_km2.abs().max()),'geometry_repair_events':len(REPAIRS),'limitations':['Confirmed1196 is the existing mapping classification, not a new independent modern membership adjudication.','Original1159 is the original AllRGIs field scope; six confirmed spatial recoveries add37 members;2692 adds9 ambiguous candidate members.','Original RGICount totals1204; it is neither original listed1159 nor confirmed1196 nor download1205.','Modern endpoints use source inventory acquisition dates; unequal dates and unknown individual LIA dates prevent interpreting loss fractions as rates.','Read-only area/membership harmonization does not qualify legacy191 ice-volume models to the selected654 reconstruction standard.','No claim of complete HMA coverage, identical large-glacier selection, or climate attribution.']}
    jdump('summary.json',summary)
    (OUT/'definitions.txt').write_text('Three legacy modern-area definitions are retained explicitly.\noriginal_1159: original AllRGIs strings and original modern shapefile.\nconfirmed_1196: existing manifest download members minus its nine2692 candidates, using the archived1205 GPKG.\ndownload_1205_including_candidates: all archived download members, with candidate status preserved.\nDBF field AllRGIs has width254. Seven systems reach254 characters/17 listed IDs while their RGICount declares more. Six systems have37 confirmed recovered members;2692 has9 ambiguous candidate members.\nNo source IDs are matched by numeric suffix across RGI versions. Cross-version check uses exact positive-area polygon intersections. Empty output means no overlap in these two sampled cohorts; it is not a general cross-version crosswalk.\nAll areas WGS84 ellipsoidal km2, endpoint unions per system. Signed loss retained, no rate. Source shapefiles and selected654 population unchanged.\n',encoding='utf-8')
    jdump('output_sha256.json',{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='output_sha256.json'})
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
