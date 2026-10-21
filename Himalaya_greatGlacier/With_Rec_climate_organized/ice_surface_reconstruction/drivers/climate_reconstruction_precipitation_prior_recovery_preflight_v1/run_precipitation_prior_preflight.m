function run_precipitation_prior_preflight(output_path, phyda_root, compact_prior_path)
% Rebuild the annual HMA precipitation prior twice with a common RNG state.

old_dir = pwd;
cleanup_dir = onCleanup(@() cd(old_dir));
old_rng = rng;
cleanup_rng = onCleanup(@() rng(old_rng));
cd(phyda_root);
addpath(genpath(phyda_root));

mdl_prior = 'cesm_lme010';
state_tp = 'p';
mon_avg_o = 1;
mon_avg_f = 12;
hma_lat_range = [24, 51];
hma_lon_range = [62, 110];
rng_seed = 4236;
[gcm_syr, ~] = modelyrs(mdl_prior);
member_years = 1001:1848;
p_yrs = member_years - gcm_syr + 1;

assert(isequal(p_yrs, 152:999), 'unexpected-model-year-indices');

rng(rng_seed, 'twister');
[prior_first, xmeta_first] = load_S_720360( ...
    mdl_prior, state_tp, mon_avg_o, mon_avg_f, p_yrs, hma_lat_range, hma_lon_range);
rng_state_after_first = rng;

rng(rng_seed, 'twister');
[prior_repeat, xmeta_repeat] = load_S_720360( ...
    mdl_prior, state_tp, mon_avg_o, mon_avg_f, p_yrs, hma_lat_range, hma_lon_range);
rng_state_after_repeat = rng;

assert(isequal(size(prior_first), [336, 848]), 'unexpected-prior-shape');
assert(isequal(size(prior_repeat), [336, 848]), 'unexpected-repeat-shape');
assert(all(isfinite(prior_first), 'all'), 'nonfinite-prior');
assert(all(isfinite(prior_repeat), 'all'), 'nonfinite-repeat');

repeat_difference = prior_repeat - prior_first;
repeat_max_abs_mm_day = max(abs(repeat_difference), [], 'all');
repeat_rmse_mm_day = sqrt(mean(repeat_difference.^2, 'all'));
repeat_changed_count = nnz(repeat_difference ~= 0);

assert(isequaln(xmeta_first, xmeta_repeat), 'xmeta-repeat-mismatch');

compact_data = load(compact_prior_path, 'prior_fields');
compact_export = compact_data.prior_fields.precipitation;
assert(isequal(size(compact_export), [336, 1001]), 'unexpected-compact-export-shape');
compact_overlap_mm_day = compact_export(:, 2:849) / 30;
compact_difference = prior_first - compact_overlap_mm_day;
compact_abs_difference = abs(compact_difference);
compact_overlap_zero_mask = compact_overlap_mm_day == 0;
prior_zero_mask = prior_first == 0;
both_nonzero = ~compact_overlap_zero_mask & ~prior_zero_mask;
exact_match = compact_difference == 0;

compact_overlap_max_abs_mm_day = max(compact_abs_difference, [], 'all');
compact_overlap_rmse_mm_day = sqrt(mean(compact_difference.^2, 'all'));
compact_overlap_exact_count = nnz(exact_match);
compact_overlap_exact_fraction = compact_overlap_exact_count / numel(exact_match);
compact_overlap_changed_count = nnz(~exact_match);
compact_zero_mask_mismatch_count = nnz(xor(compact_overlap_zero_mask, prior_zero_mask));
compact_nonzero_max_abs_mm_day = max(compact_abs_difference(both_nonzero), [], 'all');
compact_nonzero_rmse_mm_day = sqrt(mean(compact_difference(both_nonzero).^2, 'all'));
compact_nonzero_changed_count = nnz(compact_difference(both_nonzero) ~= 0);
compact_nonzero_count = nnz(both_nonzero);

positive_pair = prior_first > 0 & compact_overlap_mm_day > 0;
unit_ratio = compact_export(:, 2:849);
unit_ratio = unit_ratio(positive_pair) ./ prior_first(positive_pair);
unit_ratio_median = median(unit_ratio);
unit_ratio_p01 = prctile(unit_ratio, 1);
unit_ratio_p99 = prctile(unit_ratio, 99);

prior_min_mm_day = min(prior_first, [], 'all');
prior_mean_mm_day = mean(prior_first, 'all');
prior_median_mm_day = median(prior_first, 'all');
prior_max_mm_day = max(prior_first, [], 'all');
prior_zero_count = nnz(prior_first == 0);

lat = xmeta_first{1}.lat(:);
lon = xmeta_first{1}.lon(:);
id_X = xmeta_first{1}.id_X;
varnm = xmeta_first{1}.varnm;
varnmlng = xmeta_first{1}.varnmlng;

precipitation_prior_336x848_mm_day = prior_first;
save(output_path, ...
    'precipitation_prior_336x848_mm_day', 'member_years', 'p_yrs', ...
    'mdl_prior', 'state_tp', 'mon_avg_o', 'mon_avg_f', ...
    'hma_lat_range', 'hma_lon_range', 'rng_seed', ...
    'rng_state_after_first', 'rng_state_after_repeat', ...
    'lat', 'lon', 'id_X', 'varnm', 'varnmlng', ...
    'repeat_max_abs_mm_day', 'repeat_rmse_mm_day', 'repeat_changed_count', ...
    'compact_overlap_max_abs_mm_day', 'compact_overlap_rmse_mm_day', ...
    'compact_overlap_exact_count', 'compact_overlap_exact_fraction', ...
    'compact_overlap_changed_count', 'compact_zero_mask_mismatch_count', ...
    'compact_nonzero_max_abs_mm_day', 'compact_nonzero_rmse_mm_day', ...
    'compact_nonzero_changed_count', 'compact_nonzero_count', ...
    'unit_ratio_median', 'unit_ratio_p01', 'unit_ratio_p99', ...
    'prior_min_mm_day', 'prior_mean_mm_day', 'prior_median_mm_day', ...
    'prior_max_mm_day', 'prior_zero_count', '-v7.3');

disp('PRECIPITATION_PRIOR_PREFLIGHT_COMPLETE');
end

