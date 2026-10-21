from pathlib import Path
namespace={'__file__':str(Path(__file__).resolve())}
source=Path('yx_followup_transaction_v1.py').read_text(encoding='utf-8')
source=source.replace('_20261009_v1','_20261010_v1')
source=source.replace("user_authorization='完全完成; 并行完成'", "user_authorization='2026-10-10 explicit user request: 长度变化呢？'")
exec(compile(source,'yx_length_transaction_v1','exec'),namespace)
out=Path('H:/ToYangxin/result_yx/length_change_selected654_20261010_v1')
print(namespace['register'](
 'yx_selected654_length_diagnostic_v1',
 {'status':'registered-readonly-fixed-LIA-axis-length-diagnostic',
  'population':654,'source_records':655,'source_selection':'H:/ToYangxin/result_yx/selected_reconstruction_20261010_v1_654',
  'output_directory':str(out),'model_recalculations':False,
  'geometry_only_analysis_authorized':True,'stored_original_source_axes':650,
  'stored_candidate_directed_graph_source_axes_separate':4,
  'source_axis_definition':'longest original complete head-to-root path within each root, including actual mask-constrained connector paths; system reference chosen by LIA-only maximum before modern intersections',
  'paired_metric':'terminal segment of fixed LIA reference axis beyond last modern polygon intersection, meters; separately total/continuous modern covered axis lengths',
  'no_full_glacier_length_or_actual_observed_retreat_claim':True,
  'head_and_tail_extension_or_source_reconstruction_authorized':False,
  'native_grid_coordinates_convert_to_projected_meters':True,
  'all_root_axes_retained':True,'main_original650_and_candidate4_statistics_separate':True,
  'numerical_intersection_tolerance_m':1e-6,
  'primary_unambiguous_class':'original source initialized-axis family, one component/root, simple full axis inside LIA, one continuous modern intersection covering same upstream reference head; modern endpoint ends before stored axis end',
  'censored_or_review_cases':'modern reaches stored tail, multiple roots/components, LIA axis outside vector, modern multi-interval/internal gap, no shared head, no intersection, nonsimple path',
  'no_model_gate_or_parameter_changes':True},
 'YX selected654 length supplement: preregister fixed-source-axis geometry diagnostic and separate censoring',
 'The user requests length change for the current654 retained systems. Existing OGGM route pickles are initialized and may be cropped, often built on raw native-product bed; four newer candidate graph sources use a different routing family. Independent longest-line lengths or root-only lengths cannot be silently treated as measured historical length or retreat. Authorize only read-only geometry analysis from frozen existing sources: reconstruct complete head-to-root axes with original mask-constrained connectors and convert grid coordinates to original meter projection; no PP/2D, centerline regeneration, head restoration, endpoint extrapolation or source mutation. Choose longest LIA source path per root and the LIA-only maximum for system reference before consulting modern outlines. Export reference-axis length, all modern intersection intervals/covered lengths and terminal uncovered axis segment separately. Primary unambiguous original-axis subset requires single component/root, simple entire source axis within true LIA vector, single continuous modern intersection including same upstream head and ending before stored axis tail; modern reaches tail is censored, not zero observed change. Multi-root/candidate family, fragmentation, loss of common head, vector/raster mismatch and absence are explicitly retained as diagnostics/review rows. All654 remain in population ledger, additional length QC never discards them or changes prior qualified volume counts. Per-root axes and endpoint/map samples support review. Statistics must state their evaluated subset and must not extrapolate to whole654 or HMA; no annual rate, true whole-glacier absolute length, formal uncertainty or observed advance inferred from truncated source axes. New supplement independently verified; original selected package unchanged.',
 False,[],
 'Compute stored-source geometric axis diagnostics once in new versioned directory, retain all654 row statuses and root axes, independently verify intervals/sources/aggregation before reporting.',
 phase='geometry-study-authorized'))
