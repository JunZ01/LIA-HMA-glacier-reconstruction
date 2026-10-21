from pathlib import Path
import json, os, shutil, time, hashlib

namespace={'__file__':str(Path(__file__).resolve())}
source=Path('yx_followup_transaction_v1.py').read_text(encoding='utf-8')
source=source.replace('_20261009_v1','_20261010_v1')
source=source.replace("user_authorization='完全完成; 并行完成'", "user_authorization='2026-10-10 explicit user request: 丢弃这6条冰川'")
exec(compile(source,'yx_selection_transaction_v1','exec'),namespace)
register=namespace['register'];sha=namespace['sha']
base=Path('H:/ToYangxin/result_yx')
out=base/'selected_reconstruction_20261010_v1_654'
audit=Path(os.environ['YX_SELECTION_AUDIT'])
expected=os.environ['YX_SELECTION_AUDIT_SHA'].lower()
assert sha(audit).lower()==expected
j=json.loads(audit.read_text(encoding='utf-8'));assert j['status']=='pass'
s=json.loads((out/'summary.json').read_text(encoding='utf-8'))
assert s['systems']==654 and s['source_records']==655 and s['excluded_system_count']==6
m=json.loads((out/'selection_sha256_manifest.json').read_text(encoding='utf-8'))
assert len(m['files'])==33
for r in m['files']:
    f=out/r['relative_path']; assert f.stat().st_size==r['bytes']
    assert sha(f).lower()==r['sha256'].lower(),r['relative_path']
print(register(
    'yx_selected_sample654_drop6_v1',
    {'status':'verified-user-selected654-six-systems-excluded',
     'output_directory':str(out),'summary':s,
     'selection_manifest':str(out/'selection_sha256_manifest.json'),
     'selection_manifest_sha256':sha(out/'selection_sha256_manifest.json'),
     'independent_selection_audit':str(audit),'independent_selection_audit_sha256':expected,
     'all_retained_models_already_verified':True,'model_recalculations':0,
     'raw_inventory_or_prior_delivery_modified':False},
    'YX user-selected654 delivery: six systems excluded from current analysis and GIS',
    'The explicit user removal decision is implemented and independently verified in a new immutable selected654 derivative. Retained655 source rows map uniquely to654 systems; excluded7 rows are03055/03621/05103/05176/29305/14077/28236. Source joint11977+29307 remains one retained system. All654 paired reconstruction values and their CSV bytes are unchanged; no endpoint or raster recomputation. Selected LIA endpoint area27209.3228041975km2 and modern25954.3288930462km2 give1254.9939111513km2 or4.6123673132percent reduction using WGS84 per-system summed areas. LIA modeled volume3351.2161671466/modern3022.6269297718/loss328.5892373748km3 or9.8050743666percent remains unchanged, because excluded six were never in those model totals. Independent scope,10-mountain aggregates,GIS and654-feature Shapefile geometry,source preservation and report links pass. The new analysis has654 retained/QC-qualified systems and0 retained model-review systems; that is a sample exclusion, not qualification of excluded systems or removal of historical failures. Original660/662 baseline and rawinventory remain unchanged.1256 native modern members include18 missing-thickness members across14 systems; supported native modernvolume3022.6156327199km3 is not a complete observed-volume claim. All subsequent selected-sample analysis should use this654/655 version; prior660 population is historical upstream context. No formal191 accepted counts, climate processes or model launch authority change.',
    False, [],
    'Use selected654/655-source derivative for subsequent inventory, mountain, ice-surface and volume analysis; preserve six-system exclusion and original evidence. No simulation or old-output deletion.',
    phase='verified-selected-delivery'))
pointer=base/'latest_completed_reconstruction.json'
previous=base/f'latest_completed_reconstruction_prior_pointer_{time.time_ns()}.json'
shutil.copy2(pointer,previous)
value={'path':str(out),'report':str(out/'report.html'),'summary':s,'reconstructed':654,
       'unestimated':0,'excluded_systems':s['excluded_systems'],'excluded_count':6,
       'source_records':655,'original_assessed_population':660,
       'full_population_reconstructed':False,'all_retained_selected_systems_reconstructed':True,
       'full_manifest':str(out/'selection_sha256_manifest.json'),
       'raster_reference_directory':str(Path(s['reference_delivery'])/'rasters'),
       'previous_pointer_snapshot':str(previous)}
tmp=pointer.with_suffix('.selected654.tmp')
tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,pointer)
selected_pointer=base/'latest_selected_reconstruction.json'
tmp=selected_pointer.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,selected_pointer)
print(json.dumps({'published':str(out),'selected_systems':654,'source_records':655,'excluded_systems':6},ensure_ascii=False))
