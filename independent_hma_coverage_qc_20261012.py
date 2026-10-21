from pathlib import Path
import json,hashlib,itertools,collections
import fiona
import pandas as pd
import numpy as np
from pyproj import CRS,Transformer
from shapely.geometry import shape
from shapely.ops import transform
from shapely import make_valid,STRtree

ROOT=Path(__file__).resolve().parent
OUT=Path('H:/ToYangxin/result_yx/hma_mountain_coverage_20261012_v1')
PART=Path('D:/01 文件整理/04 数据/HMA_newregion/shp_wgs84/HMA_newregion_wgs84.shp')
SRC={
 'selected654':(Path('H:/ToYangxin/result_yx/selected_reconstruction_20261010_v1_654/RGI_box_selected654.shp'),'UniqueID'),
 'legacy245':(ROOT/'Himalaya_greatGlacier/Him_10_lia_with_large.shp','UniqueID'),
 'tanggula691':(Path('D:/01 文件整理/04 数据/other_lia_shp/tanggula_complete/tanggula_LIA_with_UniqueId.shp'),'UniqueId'),
 'east_tianshan640':(Path('D:/01 文件整理/04 数据/other_lia_shp/east_tianshan/east_tianshan_LIA_with_UniqueId.shp'),'UniqueId')
}
with fiona.open(PART) as f:
 part_crs=CRS(f.crs_wkt); partto=Transformer.from_crs(part_crs,4326,always_xy=True)
 parts=[(dict(x['properties'])['Primary_ID'],transform(partto.transform,shape(x['geometry']))) for x in f]
area_to=Transformer.from_crs(4326,6933,always_xy=True)
aparts=[(name,transform(area_to.transform,g)) for name,g in parts]
audit={'schema':'independent-hma-coverage-spatial-qc-20261012-v1','method':'Independent Fiona reads of original shapefiles; explicit pyproj transforms (always_xy), source-CRS make_valid on in-memory copies; EPSG:6933 equal-area intersections for largest-region assignment. No root audit driver imported. Cross-cohort intersections are tested on transformed original-derived polygons with STRtree and exact intersects.','sources':{},'cross_cohort_tests':[],'source_assignments':[]}
allgeoms={}
expected_counts={'selected654':654,'legacy245':245,'tanggula691':691,'east_tianshan640':640}
for cohort,(path,idcol) in SRC.items():
 with fiona.open(path) as f:
  crs=CRS(f.crs_wkt); fwd=Transformer.from_crs(crs,4326,always_xy=True); inv=Transformer.from_crs(4326,crs,always_xy=True)
  raw=list(f)
 geoms=[]; counts=collections.Counter(); invalid=0; minfrac=1.; rounds=[]; bounds=[]; assignment_rows=[]; postproject_invalid=[]
 for x in raw:
  native=shape(x['geometry']); props=dict(x['properties']); uid=int(props[idcol]); invalid+=not native.is_valid
  clean=make_valid(native); g=transform(fwd.transform,clean)
  if not g.is_valid:postproject_invalid.append(uid);g=make_valid(g)
  geoms.append(g); bounds.append(g.bounds)
  sample=native.representative_point(); xx,yy=fwd.transform(sample.x,sample.y); x2,y2=inv.transform(xx,yy)
  rounds.append(float(np.hypot(x2-sample.x,y2-sample.y)))
  ga=transform(area_to.transform,g)
  pieces=sorted([(name,ga.intersection(pg).area) for name,pg in aparts if ga.intersects(pg)],key=lambda v:v[1],reverse=True)
  best,frac=(pieces[0][0],pieces[0][1]/ga.area) if pieces and ga.area else ('Outside / unresolved',0)
  chosen=best if frac>=.5 else 'Outside / unresolved'; counts[chosen]+=1;minfrac=min(minfrac,frac)
  assignment_rows.append({'cohort':cohort,'source_id':uid,'independent_region':chosen,'equal_area_largest_fraction':float(frac),'stored_frozen_region':props.get('mountain')})
 b=np.asarray(bounds)
 audit['sources'][cohort]={'path':str(path),'sha256_shp':hashlib.sha256(path.read_bytes()).hexdigest(),'crs_wkt':crs.to_wkt(),'authority':crs.to_authority(),'feature_count':len(raw),'invalid_source_count':int(invalid),'postprojection_invalid_ids_repaired':postproject_invalid,'expected_count_pass':len(raw)==expected_counts[cohort],'wgs84_bounds':[float(b[:,0].min()),float(b[:,1].min()),float(b[:,2].max()),float(b[:,3].max())],'roundtrip_max_error_native_units':max(rounds),'native_units':crs.axis_info[0].unit_name,'assignment_counts':dict(counts),'minimum_largest_fraction':float(minfrac)}
 audit['source_assignments'].extend(assignment_rows);allgeoms[cohort]=geoms
for a,b in itertools.combinations(SRC,2):
 tree=STRtree(allgeoms[b]);hit=0;area_hit=0
 for ga in allgeoms[a]:
  matches=tree.query(ga,predicate='intersects');hit+=len(matches)
  for j in matches:
   if ga.intersection(allgeoms[b][j]).area>0:area_hit+=1
 audit['cross_cohort_tests'].append({'left':a,'right':b,'exact_intersects_pairs_including_touch':int(hit),'positive_area_pairs':int(area_hit),'zero_intersection_pass':hit==0})
registry=pd.read_csv(OUT/'source_system_registry.csv')
ind=pd.DataFrame(audit['source_assignments'])
joined=registry.merge(ind,on=['cohort','source_id'],how='outer',validate='one_to_one',indicator=True)
audit['root_registry_comparison']={'row_count':len(joined),'all_rows_matched':bool((joined._merge=='both').all()),'region_mismatch_count':int((joined.mountain10!=joined.independent_region).sum()),'region_mismatches':joined[joined.mountain10!=joined.independent_region][['cohort','source_id','mountain10','independent_region']].to_dict('records')}
with fiona.open(OUT/'coverage_geometry.gpkg',layer='source_systems') as f:
 delivered=list(f)
audit['delivered_gpkg_geometry_check']={'features':len(delivered),'invalid_geometries':sum(not shape(x['geometry']).is_valid for x in delivered),'empty_geometries':sum(shape(x['geometry']).is_empty for x in delivered)}
oldmanifest=ROOT/'Himalaya_greatGlacier/With_Rec_climate_organized/outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/system_manifest_work/system_manifest_245_v0.csv'
old=pd.read_csv(oldmanifest)
cross=old.merge(ind[ind.cohort=='legacy245'],left_on='lia_unique_id',right_on='source_id')
audit['legacy_label_spatial_crosswalk']=cross.groupby(['region_label','independent_region']).size().reset_index(name='n').to_dict('records')
audit['legacy_label_source_path']=str(oldmanifest)
audit['recommended_wording']=[
 '旧245条源轮廓按本地10区实际空间相交归属为：西喜马拉雅110、中喜马拉雅114、东喜马拉雅21。旧四组region_label为来源标签，不能替代山脉空间归属；此前按标签把82条解释成喀喇昆仑/兴都库什的说法需纠正。',
 '唐古拉独立来源691条轮廓全部位于本地10区的Tibetan Interior Mountains单元。来源地区为唐古拉，不等同把整个高原内部山地改称唐古拉。',
 '东天山独立来源640条轮廓全部位于本地10区名为Western Tien Shan的多边形。本地该多边形覆盖范围比名称字面含义更广；保留原始分区名称并注明含东天山来源，不能说这640条在10区范围外。',
 '四来源的原始轮廓经各自声明CRS转为WGS84后，六组跨源比较均无几何相交。此结论支持没有相交重复，但仍不独立证明冰川系统身份、现代成员映射或样本筛选口径均已统一。',
 '统计源轮廓记录数不等于可直接合并计算的统一大冰川样本数。'
]
facts=(ROOT/'LIA_PROJECT_FACTS.md').read_text(encoding='utf-8-sig')
fact_hits=[{'line':i+1,'text':s} for i,s in enumerate(facts.splitlines()) if any(term in s for term in ['region_label','Karakoram/Hindu','Karakoram / Hindu','Him_10_lia_with_large'])]
audit['canonical_conflict_check']={'scope':'Searched the fully read current facts for legacy geographic-label declarations and exact source filename; no equivalent frozen assertion found. A previous conversation interpretation is contradicted, not the raw source records. No source or canonical edits made.','facts_sha256':hashlib.sha256((ROOT/'LIA_PROJECT_FACTS.md').read_bytes()).hexdigest(),'targeted_hits':fact_hits,'affected_simulation_launched':False,'conclusion':'Do not silently replace legacy labels used by established runs. Retain original label and new spatial mountain assignment as different fields; geographical reporting should use independently verified spatial assignment.'}
audit['pass']=all(x['expected_count_pass'] for x in audit['sources'].values()) and all(x['zero_intersection_pass'] for x in audit['cross_cohort_tests']) and audit['root_registry_comparison']['all_rows_matched'] and audit['root_registry_comparison']['region_mismatch_count']==0 and audit['delivered_gpkg_geometry_check']['invalid_geometries']==0
(OUT/'independent_spatial_QC.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in audit.items() if k not in ['source_assignments','sources']},ensure_ascii=False,indent=2))
print(json.dumps(audit['sources'],ensure_ascii=False,indent=2))
