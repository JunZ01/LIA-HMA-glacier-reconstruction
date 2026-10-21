from pathlib import Path

namespace = {'__file__': str(Path(__file__).resolve())}
source = Path('yx_followup_transaction_v1.py').read_text(encoding='utf-8')
source = source.replace('_20261009_v1', '_20261010_v2')
source = source.replace("user_authorization='完全完成; 并行完成'", "user_authorization='2026-10-10 explicit user correction: 654个都可以计算 我认为; complete all retained systems under disclosed paired length definition'")
exec(compile(source, 'yx_paired_length654_transaction_v2', 'exec'), namespace)

out = Path('H:/ToYangxin/result_yx/paired_axis_length_selected654_20261010_v2')
print(namespace['register'](
    'yx_selected654_paired_axis_length_v2',
    {
        'status': 'registered-all654-paired-stored-axis-length-analysis',
        'population': 654,
        'source_records': 655,
        'output_directory': str(out),
        'selected_package': 'H:/ToYangxin/result_yx/selected_reconstruction_20261010_v1_654',
        'immutable_LIA_axes_reuse': 'H:/ToYangxin/result_yx/length_change_selected654_20261010_v1/source_axes/root_axes.csv',
        'user_corrected_all654_length_population': True,
        'new_metric_disclosed': True,
        'paired_length_definition': 'For each endpoint separately, maximum complete stored head-to-root planform axis arc length across all saved components and roots, including original mask-constrained connectors; native projected meter coordinates. Independent endpoint maxima are a stored-route geometric length index, not same-branch observed terminus retreat.',
        'difference_m': 'LIA_max_stored_axis_m - modern_max_stored_axis_m; preserve positive, zero and negative values',
        'relative_change_percent': '100 * (LIA_max_stored_axis_m - modern_max_stored_axis_m) / LIA_max_stored_axis_m',
        'overall_length_weighted_percent': '100 * sum(LIA_max_stored_axis_m - modern_max_stored_axis_m) / sum(LIA_max_stored_axis_m), all654',
        'modern_axis_does_not_intersect_LIA_reference_is_not_disappearance_or_zero_length': True,
        'previous_485_positive_uncensored_subset_does_not_filter_new_metric': True,
        'all_roots_retained_and_LIA_modern_reference_choice_independent': True,
        'source_heterogeneity_branch_selection_head_crop_vector_mismatch_and_multipart_flags_retained': True,
        'complete_physical_boundary_to_boundary_glacier_length_claim': False,
        'same_branch_observed_terminus_retreat_claim': False,
        'two_endpoint_physical_simulations': 0,
        'new_route_generation': False,
        'head_or_tail_restoration_or_extrapolation': False,
        'LIA_saved_axes_recomputed': False,
        'source_files_old_reports_and_selected_area_volume_results_modified': False,
        'annual_rate_formal_CI_or_global_HMA_extrapolation': False,
        'independent_verification_required': True,
    },
    'YX selected654 paired length: user authorizes all-system stored-axis length changes with disclosed metric',
    'The user explicitly corrects the length reporting population: all654 retained systems should receive length calculations. Resolve the prior485-only limitation by adding a distinct paired stored-axis geometric length index, preserving the verified fixed-LIA-axis study unchanged. Reuse the already extracted695 LIA axes and extract frozen modern endpoint source paths with the same complete head-to-root arclength operator and actual connectors; for each endpoint independently choose its maximum across all components/roots. Compute signed LIA-minus-modern lengths and percentages for every654 system without a positive-only, single-root or common-axis-intersection sample filter. Disconnected modern outlines and the two prior no-intersection cases use their own saved modern paths rather than false zero lengths. This disclosed maximum-reference-axis length change is different from same-axis terminal retreat; initialized head cropping, incomplete exterior matching, changed dominant branch/component and the four candidate graph families remain explicit limitations and flags. Do not call it complete physical glacier length, observed historical terminus retreat or the prior485 metric. No source-axis regeneration, head/tail restoration, physical model rerun, area/volume change, missing-system substitution or formal191 acceptance change. If frozen paired modern source evidence is missing, resolve provenance before reporting all654 complete. Publish all654 CSV, per-root GIS, signed and mountain summaries in a new immutable supplement after independent unit/path/source/aggregation checks.',
    False, [],
    'Extract frozen modern endpoint axes once, reuse sealed LIA axes, calculate all654 signed paired maximum-reference-axis changes and independently audit source paths, units, sample coverage and report claims.',
    phase='all654-paired-length-authorized'))
