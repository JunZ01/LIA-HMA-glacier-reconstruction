function fit_fold_native_psm_v2(allowed_input_path, prior_input_path, replay_input_path, output_path, phyda_root, bearth_path, gpcc_path)
% Build a 500-member operator and close experiment-1 indices and archived R.

preflight_path = [output_path '.preflight_848.mat'];
fit_fold_native_psm_v1(allowed_input_path, prior_input_path, preflight_path, phyda_root, bearth_path, gpcc_path);
F = load(preflight_path);
P = load(prior_input_path, 'experiment1_sample_indices');
R = load(replay_input_path, 'sample_indices', 'HXb0', 'R0v_filled');
sample_indices = double(P.experiment1_sample_indices(:)');
replay_indices = double(R.sample_indices(:)');

assert(numel(sample_indices) == 500, 'experiment1-sample-count-not-500');
assert(numel(unique(sample_indices)) == 500, 'experiment1-sample-indices-not-unique');
assert(all(sample_indices >= 1 & sample_indices <= 848), 'experiment1-sample-index-range-failed');
assert(isequal(sample_indices, replay_indices), 'experiment1-replay-index-closure-failed');
experiment1_index_closure_exact = true;

hxb500 = F.hxb(:, :, sample_indices);
full_hxb500_replay_max_abs = max(abs(squeeze(hxb500(:, 1, :)) - double(R.HXb0)), [], 'all', 'omitnan');
assert(full_hxb500_replay_max_abs <= 1e-9, 'full-hxb500-replay-closure-failed');

archived_error_variance = var(double(R.R0v_filled), 0, 2, 'omitnan');
fitted_full_error_variance = F.residual_variance(:, 1);
full_R_max_abs = max(abs(fitted_full_error_variance - archived_error_variance), [], 'omitnan');
full_R_median_abs = median(abs(fitted_full_error_variance - archived_error_variance), 'omitnan');
full_R_finite_count = sum(isfinite(fitted_full_error_variance));
full_R_nonfinite_count = sum(~isfinite(fitted_full_error_variance));
positive_R = F.residual_variance(isfinite(F.residual_variance) & F.residual_variance > 0);
minimum_positive_R = min(positive_R);
nonfinite_R_for_eligible_count = sum(~isfinite(F.residual_variance(F.training_count >= 20)));
nonpositive_R_for_eligible_count = sum(F.residual_variance(F.training_count >= 20) <= 0);
assert(full_R_nonfinite_count == 0, 'full-R-has-nonfinite-values');
assert(nonfinite_R_for_eligible_count == 0, 'eligible-R-has-nonfinite-values');
assert(nonpositive_R_for_eligible_count == 0, 'eligible-R-has-nonpositive-values');

years = F.years;
blocks = F.blocks;
branch_outer = F.branch_outer;
branch_inner = F.branch_inner;
model_type = F.model_type;
b = F.b;
residual_variance = F.residual_variance;
training_count = F.training_count;
baseline_coefficient_max_abs = F.baseline_coefficient_max_abs;
baseline_hxb_max_abs = F.baseline_hxb_max_abs;
locked_1921_1939_accessed = false;
matlab_release = version('-release');
save(output_path, 'years', 'blocks', 'branch_outer', 'branch_inner', 'model_type', ...
    'hxb500', 'b', 'residual_variance', 'training_count', 'sample_indices', ...
    'experiment1_index_closure_exact', 'full_hxb500_replay_max_abs', ...
    'archived_error_variance', 'fitted_full_error_variance', 'full_R_max_abs', ...
    'full_R_median_abs', 'full_R_finite_count', 'full_R_nonfinite_count', ...
    'minimum_positive_R', 'nonfinite_R_for_eligible_count', ...
    'nonpositive_R_for_eligible_count', 'baseline_coefficient_max_abs', ...
    'baseline_hxb_max_abs', 'locked_1921_1939_accessed', 'matlab_release', '-v7.3');
fprintf('FOLD_NATIVE_PSM_V2_COMPLETE members=500 index_closure=1 hxb500_closure=%.17g R_max_abs=%.17g min_R=%.17g\n', ...
    full_hxb500_replay_max_abs, full_R_max_abs, minimum_positive_R);
end
