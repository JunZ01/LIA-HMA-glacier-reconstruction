from pathlib import Path
from html.parser import HTMLParser
from datetime import datetime, timezone
import hashlib
import json
import shutil
import pandas as pd
import numpy as np

OUT = Path('H:/ToYangxin/result_yx/hma_harmonized_analysis_20261012_v1')

def read_json(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('href', 'src'):
                self.links.append(value)

checks = {}

def check(name, passed):
    checks[name] = bool(passed)
    if not passed:
        raise AssertionError(name)

parser = Links()
report = (OUT / 'report.html').read_text(encoding='utf-8')
parser.feed(report)
local_links = [p for p in parser.links if not p.startswith(('http:', 'https:', '#'))]
check('all_report_file_links_resolve', all((OUT / p).is_file() for p in local_links))
check('report_includes_climate_cv_and_limits', all(s in report for s in [
    '现代气候态未改善跨山脉预测', '不能解释成气候没有作用', '几何也参与冰面与冰量重建',
    '喜马拉雅三行没有可并入当前标定方法的冰量', '现有该区83套不能全部称为唐古拉冰川']))

master = pd.read_csv(OUT / 'unified899_area_registry.csv')
vol = pd.read_csv(OUT / 'selected654_analysis/selected654_merged_analysis.csv')
new = pd.read_csv(OUT / 'new_inventory_followup_priority.csv')
regions = pd.read_csv(OUT / 'mountain10_first_analysis.csv')
summary = read_json('first_analysis_summary.json')
check('899_unique_source_system_keys', len(master) == 899 and master.system_key.is_unique)
check('654_unique_volume_records', len(vol) == 654 and vol.record_id.is_unique)
check('legacy245_volume_blank_not_zero', len(master[master.cohort == 'legacy245']) == 245 and
      master.loc[master.cohort == 'legacy245', ['LIA_volume_km3', 'modern_volume_km3', 'net_volume_loss_km3', 'volume_loss_percent']].isna().all().all())
check('new_inventory1331_not_in_frozen_budget', len(new) == 1331 and not new.included_in654_volume_budget.any())
check('10_local_mountain_regions', len(regions) == 10 and regions.mountain.is_unique)
check('36_negative_volume_losses_retained', int((vol.volume_loss_deduplicated_km3 < 0).sum()) == 36)
check('volume_total_matches_delivered_summary', np.isclose(vol.volume_loss_deduplicated_km3.sum(),
      summary['selected654']['net_volume_loss_km3'], atol=1e-9, rtol=0))

cv = read_json('climate_geometry_exploration/cv_summary.json')
cvqc = read_json('climate_geometry_exploration/cv_QC.json')
pred = pd.read_csv(OUT / 'climate_geometry_exploration/cv_predictions1962.csv')
check('3_models_each654_complete_predictions', len(pred) == 1962 and
      pred.groupby('model').record_id.nunique().eq(654).all() and
      not pred.duplicated(['model', 'record_id']).any())
check('cv_independent_validation_pass', cvqc['status'] == 'pass' and cvqc['independent_closed_form_pass'])
check('cv_no_new_physical_model_runs', cv['source_model_runs'] == 0 and cv['new_ice_reconstruction_runs'] == 0)
check('delivery_no_new_physical_model_runs', summary['new_ice_reconstruction_or_dynamic_runs'] == 0)
check('historical_climate_attribution_not_claimed_complete', summary['climate_attribution_completed'] is False)

wb = read_json('workbook_QC.json')
check('workbook_payload_matches_validated_input', wb['input_payload_sha256'] == sha(OUT / 'workbook_payload.json'))
check('workbook6_sheets_and_counts', (wb['sheets'], wb['master_rows'], wb['volume_rows'], wb['legacy_scenario_rows'], wb['new_candidate_rows']) == (6, 899, 654, 735, 1331))
check('workbook_formula_and_blank_checks_pass', all(wb[k] == 'pass' for k in ['summary_reconciliation', 'legacy_volume_blank_not_zero', 'formula_input_change_and_restore']))
check('workbook_visual_review_complete', 'pass' in str(wb['visual_review']).lower())
component_reports = [
    'selected654_independent_qc/independent_analysis_QC.json',
    'selected654_independent_qc/cv_report_review.json',
    'legacy_volume_comparability/QC.json',
    'new_inventory_matching/independent_QC.json',
    'climate_geometry_exploration/cv_QC.json']
check('component_independent_QC_pass', all(read_json(p)['status'] == 'pass' for p in component_reports))
identity = read_json('modern_identity/final_QC.json')
check('identity_source_and_output_hash_checks_pass', identity['all_original_output_hashes_match'] and
      identity['supplement_output_hashes_match'] and all(identity['source_member_matching'].values()))

result = {
    'status': 'pass', 'completed_utc': datetime.now(timezone.utc).isoformat(),
    'checks': checks, 'checks_passed': sum(checks.values()),
    'unique_report_local_links': len(set(local_links)),
    'scientific_figures_visually_reviewed': [
        'mountain10_area_comparison899.png',
        'selected654_analysis/selected654_regional_budget.png',
        'selected654_analysis/selected654_spatial_volume_partition.png',
        'climate_geometry_exploration/cv_mountain_prediction_comparison.png'],
    'scope': 'Completed first harmonized descriptive analysis, spatial decomposition, and prespecified geometry-climate background prediction check. No new glacier reconstruction or dynamic simulation.',
    'pending_scientific_work': [
        'Legacy245 same-method modern2D calibration and paired reconstruction',
        'New1331 inventory completion and glacier-system identity adjudication',
        'Paired endpoint reconstruction under alternative thickness products',
        'Historical climate and inventory-age constraints; formal uncertainty and causal attribution'],
    'verification_limits': [
        'HTML local targets and structure checked; scientific figure PNGs visually reviewed. HTML browser rendering not tested.',
        'Workbook validated and rendered using artifact-tool; native Excel engine not tested.',
        'This delivery check consumes completed independent component QC; it does not repeat glacier model runs or source raster integrations.'],
    'source_canonical_receipt': 'canonical_receipt_and_controller_validation.json',
    'component_reports': component_reports + ['modern_identity/final_QC.json', 'workbook_QC.json']
}
(OUT / 'delivery_QC.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
shutil.copy2(__file__, OUT / 'reproduce_delivery_qc.py')
manifest = []
for p in sorted(OUT.rglob('*')):
    if p.is_file() and p.name != 'delivery_manifest.json':
        manifest.append({'path': p.relative_to(OUT).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)})
(OUT / 'delivery_manifest.json').write_text(json.dumps({'generated_utc': datetime.now(timezone.utc).isoformat(),
    'file_count': len(manifest), 'files': manifest}, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'status': result['status'], 'checks_passed': result['checks_passed'],
                  'manifest_files': len(manifest), 'linked_files': len(set(local_links))}))
