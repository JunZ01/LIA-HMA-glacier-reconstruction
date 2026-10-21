function export_round134_netp_mjj_exact_prior_v1(output_path, phyda_root, annual_prior_path)
% Export exact annual and May-July precipitation priors via the production loader.

old_dir = pwd;
cleanup_dir = onCleanup(@() cd(old_dir));
old_rng = rng;
cleanup_rng = onCleanup(@() rng(old_rng));
cd(phyda_root);
addpath(genpath(phyda_root));

mdl_prior = 'cesm_lme010';
hma_lat_range = [24, 51];
hma_lon_range = [62, 110];
rng_seed = 4236;
[gcm_syr, ~] = modelyrs(mdl_prior);
member_years = 1001:1848;
p_yrs = member_years - gcm_syr + 1;

rng(rng_seed, 'twister');
[annual_precipitation, xmeta_p] = load_S_720360( ...
    mdl_prior, 'p', 1, 12, p_yrs, hma_lat_range, hma_lon_range);
rng(rng_seed, 'twister');
[mjj_precipitation, ~] = load_S_720360( ...
    mdl_prior, 'p', 5, 7, p_yrs, hma_lat_range, hma_lon_range);

assert(isequal(size(annual_precipitation), [336, 848]));
assert(isequal(size(mjj_precipitation), [336, 848]));
assert(all(isfinite(annual_precipitation), 'all'));
assert(all(isfinite(mjj_precipitation), 'all'));

archived = load(annual_prior_path, 'precipitation_prior_336x848_mm_day');
annual_difference = annual_precipitation - archived.precipitation_prior_336x848_mm_day;
annual_closure_max_abs = max(abs(annual_difference), [], 'all');
annual_closure_rmse = sqrt(mean(annual_difference.^2, 'all'));
annual_closure_changed_count = nnz(annual_difference ~= 0);
assert(annual_closure_max_abs == 0, 'annual-prior-exact-closure-failed');

lat = xmeta_p{1}.lat(:);
lon = xmeta_p{1}.lon(:);

save(output_path, ...
    'annual_precipitation', 'mjj_precipitation', 'member_years', 'p_yrs', ...
    'lat', 'lon', 'mdl_prior', 'hma_lat_range', 'hma_lon_range', 'rng_seed', ...
    'annual_closure_max_abs', 'annual_closure_rmse', ...
    'annual_closure_changed_count', '-v7.3');

disp('ROUND134_NETP_MJJ_EXACT_PRIOR_V1_COMPLETE');
end

