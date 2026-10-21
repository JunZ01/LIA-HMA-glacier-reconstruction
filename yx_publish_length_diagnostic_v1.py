from pathlib import Path
import json,os,time,hashlib,shutil
namespace={'__file__':str(Path(__file__).resolve())}
source=Path('yx_followup_transaction_v1.py').read_text(encoding='utf-8').replace('_20261009_v1','_20261010_v1')
source=source.replace("user_authorization='完全完成; 并行完成'", "user_authorization='2026-10-10 user request: 长度变化呢？'")
exec(compile(source,'yx_length_transaction_v1','exec'),namespace)
register=namespace['register'];sha=namespace['sha']
out=Path('H:/ToYangxin/result_yx/length_change_selected654_20261010_v1')
audit=Path(os.environ['YX_LENGTH_AUDIT']); expected=os.environ['YX_LENGTH_AUDIT_SHA'].lower()
assert sha(audit).lower()==expected
j=json.loads(audit.read_text(encoding='utf-8'));assert j['status']=='pass'
s=json.loads((out/'summary.json').read_text(encoding='utf-8'));assert s['selected_systems']==654
files=[]
for p in sorted(out.rglob('*')):
    if not p.is_file():continue
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    files.append({'relative_path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
m=out/'length_analysis_sha256_manifest.json';assert not m.exists()
m.write_text(json.dumps({'status':'sealed-independently-verified-fixed-axis-length-diagnostics',
    'independent_QC':str(audit),'independent_QC_sha256':expected,'summary':s,'files':files},ensure_ascii=False,indent=2),encoding='utf-8')
n=s['primary_unambiguous_systems']
print(register(
 'yx_selected654_length_diagnostic_v1',
 {'status':'independently-verified-selected654-fixed-axis-length-diagnostic-terminal',
  'output_directory':str(out),'summary':s,'manifest':str(m),'manifest_sha256':sha(m),
  'independent_QC':str(audit),'independent_QC_sha256':expected,
  'selected_population':654,'source_records':655,'root_axes':695,
  'numerical_intersection_tolerance_m':1e-6,
  'model_recalculations':0,'head_or_tail_extensions':0,
  'original_selected_package_modified':False,'whole_glacier_length_or_all654_observed_retreat_claim':False},
 'YX654 fixed-axis length supplement:695 source axes and654 diagnostics independently verified',
 f'For the user-requested length supplement, all654 retained systems have frozen LIA-only references and695 complete head-to-root source axes with actual connector geometry.650 initialized original source systems and4 candidate-directed systems stay separately labeled. Full paired polygon intersection intervals and all654 statuses are saved; no user-excluded system returns. A disclosed one-time projection-topology technical completion reuses636 already-successful root measurements unchanged and completes only59 previously unmeasured roots in34systems using quantified make_valid on invalid projected copies; source vertices and originals unchanged, no new spatial tolerance, buffering, snapping or extension. Final695 roots/654 systems pass independent interval,source,units,choice,repair,QC and aggregation checks. The preregistered original-family single-component/root/simple/LIA-complete/shared-head/single-modern-interval/uncensored subset contains{n} systems. Its mean terminal uncovered source-axis segment is{s["primary_mean_terminal_uncovered_m"]:.9f}m, median{s["primary_median_terminal_uncovered_m"]:.9f}m, P25{s["primary_p25_terminal_uncovered_m"]:.9f}/P75{s["primary_p75_terminal_uncovered_m"]:.9f}m. These are explicitly positive, geometrically unambiguous reference-axis subset statistics, not whole654 mean retreat, complete glacier length, three-dimensional surface length or observed historical terminus change. Other{654-n} systems retain diagnostic/censored/review rows and are not excluded from the selected654 inventory. Modern reaches saved tail is censored rather than zero measured change, multi-interval coverage is not a terminus distance, candidate axes not pooled as equivalent ordinary routes. Local10-mountain same-subset summaries,paired area/volume metrics,review examples,GIS andfile manifests are delivered. No annual rate, formal CI, model rerun, scientific acceptance or climate/mainline191 change.',
 False,[],
 'Use verified fixed-axis length diagnostic with explicit eligible subset and censoring; retain selected654 area/volume package unchanged. Whole-glacier lengths require separate complete boundary-matched geometry evidence, not extrapolation of these stored axes.',
 phase='verified-length-delivery'))
base=out.parent;pointer=base/'latest_completed_reconstruction.json'
previous=base/f'latest_completed_reconstruction_prior_pointer_{time.time_ns()}.json';shutil.copy2(pointer,previous)
v=json.loads(pointer.read_text(encoding='utf-8'))
assert v['reconstructed']==654 and v['source_records']==655 and v['excluded_count']==6
v['length_report']=str(out/'report.html');v['length_analysis_directory']=str(out)
v['length_summary']=s;v['length_manifest']=str(m);v['previous_pointer_snapshot']=str(previous)
tmp=pointer.with_suffix('.length.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,pointer)
lp=base/'latest_length_analysis.json';tmp=lp.with_suffix('.tmp')
tmp.write_text(json.dumps({'path':str(out),'report':str(out/'report.html'),'summary':s,'manifest':str(m)},ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,lp)
print(json.dumps({'published_length':str(out/'report.html'),'files':len(files),'primary_subset':n},ensure_ascii=False))
