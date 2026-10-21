from pathlib import Path
import json
namespace={'__file__':str(Path(__file__).resolve())}
source=Path('yx_followup_transaction_v1.py').read_text(encoding='utf-8').replace('_20261009_v1','_20261010_v1')
source=source.replace("user_authorization='完全完成; 并行完成'", "user_authorization='2026-10-10 explicit length-analysis request'")
exec(compile(source,'yx_length_transaction_v1','exec'),namespace)
p=Path('H:/ToYangxin/result_yx/length_change_selected654_20261010_v1/geometry_measurement/projection_topology_conflict_audit.json')
assert namespace['sha'](p).lower()=='8fe22cdf702548af320caee624d5417a876dc63f658d1ea057223ad52df88346'
j=json.loads(p.read_text(encoding='utf-8'))
assert j['all_source_geographic_polygons_valid'] and j['candidate_make_valid_all_valid']
assert j['nonpolygon_residual_count']==0 and j['maximum_original_vertex_to_restored_vertex_distance_m']==0
assert j['direct_invalid_vs_candidate_interval_count_conflicts']==0
assert j['maximum_comparable_interval_endpoint_difference_m']==0
print(namespace['register'](
 'yx_selected654_length_diagnostic_v1',
 {'status':'identified-pure-projection-topology-technical-r1-authorized',
  'geometry_only_analysis_authorized':True,'model_recalculations':False,
  'source_root_axes':695,'population':654,
  'technical_conflict_audit':str(p),'technical_conflict_audit_sha256':namespace['sha'](p),
  'technical_retry_count':1,'failed_roots_to_complete':59,'failed_systems':34,
  'completed_636_successful_roots_reused_without_remeasurement':True,
  'new_version_output':'H:/ToYangxin/result_yx/length_change_selected654_20261010_v1/geometry_measurement/technical_r1',
  'disclosed_new_preprocessing':'make_valid only on invalid projected modern polygon copies, retain Polygon/MultiPolygon; no original edits',
  'audit_measured_max_area_difference_m2':j['maximum_absolute_candidate_area_difference_m2'],
  'audit_measured_max_boundary_difference_m':j['maximum_boundary_Hausdorff_m'],
  'metric_primary_QC_and_1e6_numerical_merge_tolerance_unchanged':True,
  'new_buffer_snap_densification_axis_or_endpoint_extensions':False},
 'YX654 fixed-axis measurement: quantified projection selfcontact resolved by one disclosed technical completion',
 'The first geometry-only pass retains636 successful roots and59 unmeasured roots in34systems due only to projected modern polygon validity assertions. All source geographic polygons remain valid and SHA-identical. Read-only candidate make_valid restores valid projected polygon copies without losing any original vertex or leaving nonpolygon remnants: maximum area difference0.0004139840603m2/relative2.9184925579e-11, boundaryHausdorff9.8225427791e-11m, originalvertex displacement0. Across117 computable direct-original-invalid vs candidate endpoint-root intersections, every interval count and endpoint is exactly unchanged; one original GEOS intersection fails and remains evidenced. This resolves a numerical projection selfcontact conflict, not physical geometry or model failure. Transparently authorize one new versioned technicalr1 using make_valid only for these invalid projected copies, and complete only59 previously unmeasured roots; reuse636 success rows and interval cache exactly without remeasurement. Original first pass, sources, axes, metric, preregistered primary QC,1e-6m numerical interval merging, no-extension rule and model values remain unchanged. No buffering, snapping, densification, new spatial tolerance or replacement source geometry is allowed. Preserve reported repair effects and independently audit new final695/654 results before reporting. No scientific model or formal191 effects.',
 False,[],
 'Complete only59 failed geometry measurements in technical_r1, retain636 successful cached measurements unchanged, independently verify all695 root/654 system diagnostic aggregation.',
 phase='technical-r1-authorized'))
