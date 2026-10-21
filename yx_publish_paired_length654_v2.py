from pathlib import Path
import json, os, time, hashlib, shutil

REPO = Path(__file__).resolve().parent
OUT = Path('H:/ToYangxin/result_yx/paired_axis_length_selected654_20261010_v2')

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

namespace = {'__file__': str(Path(__file__).resolve())}
source = (REPO/'yx_followup_transaction_v1.py').read_text(encoding='utf-8').replace('_20261009_v1','_20261010_v2')
source = source.replace("user_authorization='完全完成; 并行完成'", "user_authorization='2026-10-10 explicit user correction: 654个都可以计算 我认为; all654 paired stored-axis length supplement'")
exec(compile(source, 'yx_paired_length654_transaction_v2', 'exec'), namespace)

audit = Path(os.environ['YX_PAIRED_LENGTH_AUDIT'])
audit_sha = os.environ['YX_PAIRED_LENGTH_AUDIT_SHA'].lower()
assert sha(audit) == audit_sha
qc = json.loads(audit.read_text(encoding='utf-8-sig'))
assert qc['status'] == 'pass'
s = json.loads((OUT/'summary.json').read_text(encoding='utf-8-sig'))
assert s['selected_systems'] == s['statistical_systems'] == 654
assert s['systems_missing_either_endpoint'] == 0 and s['former_169_all_included'] == 169
assert sum(s['sign_counts'].values()) == 654
manifest = OUT/'paired_length_analysis_sha256_manifest.json'
assert not manifest.exists(), 'Existing sealed supplement must not be overwritten'
files = [{'relative_path': p.relative_to(OUT).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}
         for p in sorted(OUT.rglob('*')) if p.is_file()]
manifest.write_text(json.dumps({'status':'sealed-independently-verified-all654-paired-stored-axis-length-index',
                              'independent_QC':str(audit),'independent_QC_sha256':audit_sha,'summary':s,'files':files},
                             ensure_ascii=False,indent=2),encoding='utf-8')

evidence = namespace['register'](
    'yx_selected654_paired_axis_length_v2',
    {'status':'independently-verified-all654-paired-stored-axis-length-terminal',
     'output_directory':str(OUT),'selected_population':654,'source_records':655,
     'summary':s,'manifest':str(manifest),'manifest_sha256':sha(manifest),
     'independent_QC':str(audit),'independent_QC_sha256':audit_sha,
     'new_metric_disclosed':True,'LIA_axes_recomputed':False,
     'model_recalculations':0,'new_route_generation':0,'head_or_tail_extrapolation':0,
     'original_selected_and_fixed_axis_packages_modified':False,
     'same_branch_observed_terminus_retreat_claim':False,'complete_physical_glacier_length_claim':False},
    'YX654 paired stored-axis length: all retained systems computed and independently verified',
    f'The user explicitly requests all654 retained systems rather than the prior485-only length subset. A new disclosed paired stored-axis metric is complete and independently verified for every654 system, including all169 prior diagnostic/censored cases and the two cases where modern ice does not intersect the old LIA reference axis. Reuse immutable695 LIA axes and exactly replay the selected frozen modern route/individual-forest/candidate/checkpoint sources with actual connectors in native projected meters, yielding{s["modern_roots"]} modern root axes. Each endpoint independently selects its maximum complete stored head-to-root path across all components/roots. Signed LIA-minus-modern reference-axis changes are retained: {s["sign_counts"]["shorter"]} shorter modern axes, {s["sign_counts"]["equal_within_numerical_1e_6m"]} equal within1e-6m and {s["sign_counts"]["longer"]} longer. All654 length-weighted reduction is{s["length_weighted_decrease_percent"]:.10f}percent; individual median{s["individual_decrease_percent"]["median"]:.10f}percent and mean{s["individual_decrease_percent"]["mean"]:.10f}percent. Median signed length difference{s["individual_decrease_m"]["median"]:.9f}m, mean{s["individual_decrease_m"]["mean"]:.9f}m. These are maximum stored-reference-axis length index changes, explicitly distinct from prior485 same-axis positive terminal gaps, complete physical boundary-to-boundary glacier lengths, and same-branch observed terminus retreat. Initialized cropping, incomplete exterior matching, endpoint head displacement, multiple components/roots and source family differences remain flags, not sample exclusions. No false zero modern length, positive-only filtering, source regeneration, model recomputation, area/volume change, annual rate, formal CI, HMA extrapolation or formal191 count/climate control change. Old654 selected delivery and old fixed-axis length study remain immutable; a new654 CSV, signed distribution/maps, all-root GIS, local10 mountain summaries and audit/manifests are delivered.',
    False, [],
    'Use all654 paired maximum stored-reference-axis length metric with explicit source/branch/crop limitations; keep prior485 fixed-axis terminal-gap results as a distinct historical supplement.',
    phase='verified-all654-paired-length-delivery')

base = OUT.parent
for pointer_name in ['latest_completed_reconstruction.json','latest_length_analysis.json']:
    pointer = base/pointer_name
    prior = base/f'{pointer.stem}_prior_paired_length_pointer_{time.time_ns()}.json'
    shutil.copy2(pointer,prior)
    value = json.loads(pointer.read_text(encoding='utf-8-sig'))
    if pointer_name == 'latest_completed_reconstruction.json':
        assert value['reconstructed'] == 654 and value['source_records'] == 655 and value['excluded_count'] == 6
        value['fixed_axis_length_report'] = str(base/'length_change_selected654_20261010_v1/report.html')
        value['paired_axis_length_report'] = str(OUT/'report.html')
        value['length_report'] = str(OUT/'report.html')
        value['length_analysis_directory'] = str(OUT)
        value['length_summary'] = s
        value['length_manifest'] = str(manifest)
        value['length_metric'] = s['method']
    else:
        value = {'path':str(OUT),'report':str(OUT/'report.html'),'summary':s,'manifest':str(manifest),
                 'fixed_axis_485_study_unchanged':str(base/'length_change_selected654_20261010_v1/report.html')}
    value['previous_pointer_snapshot'] = str(prior)
    tmp = pointer.with_name(pointer.name+'.paired.tmp.'+str(time.time_ns()))
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(tmp,pointer)
print(json.dumps({'report':str(OUT/'report.html'),'manifest_files':len(files),'summary':s,'evidence':str(evidence)},ensure_ascii=False))
