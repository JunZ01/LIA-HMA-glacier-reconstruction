from pathlib import Path
import hashlib,json,subprocess
import geopandas as gpd
import fiona

ROOT=Path(__file__).resolve().parent
paths={
'custom10':Path('D:/01 文件整理/04 数据/HMA_newregion/shp_wgs84/HMA_newregion_wgs84.shp'),
'comparison19_20':Path('D:/01 文件整理/04 数据/Boundary_Mountain/Boundary_Mountain_YaoXJ.shp'),
'gtng15':ROOT/'validation_data/results/paleo_surface_validation_v0/climate_reconstruction_hma_gtng_subregion_inventory_v1/raw/hma_gtng_202307_subregions.gpkg'
}
out={'schema':'hma-coverage-partition-audit-20261012-v1','scope':'Read-only source discovery and attribute verification; no source modification or simulation.','candidates':{}}
out['canonical_inputs']={}
for name in ['LIA_PROJECT_FACTS.md','LIA_EXECUTION_CONTRACT.md','LIA_CURRENT_STATE.json']:
 p=ROOT/name
 b=p.read_bytes(); text=b.decode('utf-8-sig')
 out['canonical_inputs'][name]={'bytes':len(b),'lines_read':len(text.splitlines()),'sha256':hashlib.sha256(b).hexdigest()}
validation=subprocess.run([r'D:/Anaconda/envs/oggm_env2/python.exe',str(ROOT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py')],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',check=True)
val=json.loads(validation.stdout)
out['controller_validation']={k:val[k] for k in ['status','launch_authorized','failures','counts']}
for key,p in paths.items():
 record={'path':str(p),'exists':p.exists()}
 if p.exists():
  record['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
  record['layers']=list(fiona.listlayers(p))
  g=gpd.read_file(p)
  record.update(feature_count=len(g),crs=str(g.crs),bounds=g.total_bounds.tolist(),columns=list(g.columns),geometry_types=g.geom_type.value_counts().to_dict(),invalid_geometry_count=int((~g.is_valid).sum()))
  record['attribute_rows']=json.loads(g.drop(columns=g.geometry.name).to_json(orient='records',force_ascii=False))
  record['unique_attributes']={c: {'count':int(g[c].nunique(dropna=False)),'values':g[c].drop_duplicates().fillna('').astype(str).tolist()} for c in g.columns if c!=g.geometry.name}
  record['companion_hashes']={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in p.parent.glob(p.stem+'.*') if x.suffix.lower() in ['.shp','.shx','.dbf','.prj','.cpg','.gpkg']}
 out['candidates'][key]=record
out['candidates']['custom10'].update(can_separate_tanggula=False,usable_for_current_reporting=True,label_fields=['Primary_ID','NAME','shortname'],status='User custom ten-region vector, verified as the operational scheme in the climate project. It has a broad Tibetan Interior Mountains class; do not rename that class Tanggula.')
out['candidates']['comparison19_20'].update(can_separate_tanggula='candidate-only; unresolved comparison definitions',usable_for_current_reporting=False,unresolved_proposal_rows=9,label_fields=['Moun_Liu','Primary_ID'],status='39 comparison/union polygons encode two alternative mountain definitions. Nine proposal rows remain Need discussion. Must not silently select one classification or present this as a finalized 15-region scheme.')
out['candidates']['gtng15'].update(can_separate_tanggula=False,usable_for_current_reporting=False,label_fields=['full_name','o2region'],status='15 GTN-G/RGI second-order regions, not the user custom mountain scheme; conflicts with the explicit request to avoid RGI regionalization. Retain only as a provenance discovery.')
out['recommended_partition']='custom10'
out['recommendation_reason']='The user permits ten regions. The ten-region custom vector is verified and consistent with the established climate mountain reporting framework. Report Tanggula inventory coverage separately as source coverage, not as an invented eleventh mutually exclusive reporting region.'
out['search_scope']=[
 {'directory':'D:/01 文件整理/04 数据','search':'rg --files limited to shp/gpkg/geojson/gdbtable file names; verified candidate vector attributes'},
 {'directory':'H:/ToYangxin','search':'rg --files limited to shp/gpkg/geojson with model-run/gdirs exclusions; no additional mountain partition discovered'},
 {'directory':'E:/3spyderProject/HMA_climate_reconstruction_calibration','search':'Targeted text search of Python/JSON/Markdown/YAML for HMA_newregion, Tanggula and vector paths; read the formal custom partition result and decision'},
 {'directory':'F:/work4/HMA_Ch4_figs_v2/fig_package_v2','search':'Targeted text search for region and vector references; no additional independently named finer partition found'},
 {'thread_id':'01a0d351-dbb4-7f33-b8ef-5bdf0d6ad2a0','search':'Read recent thread context only; no messaging or edits'}
]
out['supporting_local_records']=[
 'E:/3spyderProject/HMA_climate_reconstruction_calibration/experiments/custom_hma_mountain_partition_05deg_v1/FINAL_RESULT.md',
 'E:/3spyderProject/HMA_climate_reconstruction_calibration/history/CUSTOM_HMA_MOUNTAIN_PARTITION_05DEG_DECISION_20260903.md',
 'E:/3spyderProject/HMA_climate_reconstruction_calibration/experiments/custom_hma_mountain_partition_05deg_v1/plot_custom_hma_partition_v1.py'
]
out['additional_inventory_paths_for_root_audit']=[
 'D:/01 文件整理/04 数据/other_lia_shp/tanggula_complete/tanggula_LIA_with_UniqueId.shp',
 'D:/01 文件整理/04 数据/other_lia_shp/tanggula_complete/tanggula_complete_box.shp',
 'D:/01 文件整理/04 数据/other_lia_shp/tanggula_complete/rgi7_tanggula_complete_glacier.shp',
 'D:/01 文件整理/04 数据/other_lia_shp/Zhang_Tanggula_56box/Zhang_Tanggula_56box.shp',
 'D:/01 文件整理/04 数据/other_lia_shp/east_tianshan/east_tianshan_LIA_with_UniqueId.shp',
 'D:/01 文件整理/04 数据/other_lia_shp/east_tianshan/rgi7_east_tianshan_glacier.shp'
]
print(json.dumps({'controller':out['controller_validation'],'recommended_partition':out['recommended_partition'],'feature_counts':{k:v['feature_count'] for k,v in out['candidates'].items()}},ensure_ascii=False,indent=2))
(ROOT/'hma_coverage_partition_audit_20261012.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
