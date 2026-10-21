function export_round129_exact_seasonal_prior_v3(output_path, phyda_root, annual_prior_path)
% Export exact annual and seasonal HMA priors through the production loader.

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
winter_start_p_yrs = p_yrs - 1;

rng(rng_seed, 'twister');
[annual_precipitation, xmeta_p] = load_S_720360( ...
    mdl_prior, 'p', 1, 12, p_yrs, hma_lat_range, hma_lon_range);
rng(rng_seed, 'twister');
[jja_precipitation, ~] = load_S_720360( ...
    mdl_prior, 'p', 6, 8, p_yrs, hma_lat_range, hma_lon_range);
rng(rng_seed, 'twister');
[ondjfma_precipitation, ~] = load_S_720360( ...
    mdl_prior, 'p', 10, 4, winter_start_p_yrs, hma_lat_range, hma_lon_range);
rng(rng_seed, 'twister');
[annual_temperature, xmeta_t] = load_S_720360( ...
    mdl_prior, 't', 1, 12, p_yrs, hma_lat_range, hma_lon_range);
rng(rng_seed, 'twister');
[jja_temperature, ~] = load_S_720360( ...
    mdl_prior, 't', 6, 8, p_yrs, hma_lat_range, hma_lon_range);

assert(isequal(size(annual_precipitation), [336, 848]));
assert(isequal(size(jja_precipitation), [336, 848]));
assert(isequal(size(ondjfma_precipitation), [336, 848]));
assert(isequal(size(annual_temperature), [336, 848]));
assert(isequal(size(jja_temperature), [336, 848]));
assert(all(isfinite(annual_precipitation), 'all'));
assert(all(isfinite(jja_precipitation), 'all'));
assert(all(isfinite(ondjfma_precipitation), 'all'));
assert(all(isfinite(annual_temperature), 'all'));
assert(all(isfinite(jja_temperature), 'all'));

archived = load(annual_prior_path, 'precipitation_prior_336x848_mm_day');
annual_difference = annual_precipitation - archived.precipitation_prior_336x848_mm_day;
annual_closure_max_abs = max(abs(annual_difference), [], 'all');
annual_closure_rmse = sqrt(mean(annual_difference.^2, 'all'));
annual_closure_changed_count = nnz(annual_difference ~= 0);
assert(annual_closure_max_abs == 0, 'annual-prior-exact-closure-failed');

lat = xmeta_p{1}.lat(:);
lon = xmeta_p{1}.lon(:);
assert(isequal(lat, xmeta_t{1}.lat(:)));
assert(isequal(lon, xmeta_t{1}.lon(:)));

save(output_path, ...
    'annual_precipitation', 'jja_precipitation', 'ondjfma_precipitation', ...
    'annual_temperature', 'jja_temperature', 'member_years', 'p_yrs', ...
    'winter_start_p_yrs', 'lat', 'lon', 'mdl_prior', 'hma_lat_range', ...
    'hma_lon_range', 'rng_seed', 'annual_closure_max_abs', ...
    'annual_closure_rmse', 'annual_closure_changed_count', '-v7.3');

disp('ROUND129_EXACT_SEASONAL_PRIOR_V3_COMPLETE');
end
