function export_climate_reconstruction_round03_production_proxy9_v1(output_path, phyda_root)
% Export the proxy9 operator exactly as the archived production driver builds it.

old_dir = pwd;
cleanup_dir = onCleanup(@() cd(old_dir));
old_rng = rng;
cleanup_rng = onCleanup(@() rng(old_rng));
cd(phyda_root);
addpath(genpath(phyda_root));

rng(4236, 'twister');
mdl_prior = 'cesm_lme010';
mon_avg_o = 1;
mon_avg_f = 12;
itr = 1;
[gcm_syr, ~] = modelyrs(mdl_prior);
p_yrs = (1001:1848) - gcm_syr + 1;
clb_prd = [1940, 2000];
hma_lat_range = [24, 51];
hma_lon_range = [62, 110];

proxy = load_prxs_v5(9, 'y', 'ta', itr, mon_avg_o, mon_avg_f, ...
    mdl_prior, p_yrs, clb_prd, 7, 3, 'bi', hma_lat_range, hma_lon_range);
assert(numel(proxy) == 297, 'unexpected-proxy9-count');

years = (1940:2000)';
hxb = nan(297, 848);
observations = nan(61, 297);
residual_matrix = nan(297, 61);
coefficients = nan(297, 3);
for ii = 1:297
    item = proxy{ii};
    hxb(ii, :) = item.HXb(:)';
    item_years = round(item.age_yng(:));
    [present, positions] = ismember(years, item_years);
    assert(all(present), 'allowed-year-missing-from-proxy-axis');
    item_data = item.data(:);
    observations(:, ii) = item_data(positions);
    residual_matrix(ii, :) = item.Rvec(:)';
    coef = item.b(:);
    coefficients(ii, 1:numel(coef)) = coef;
end

save(output_path, 'years', 'hxb', 'observations', 'residual_matrix', ...
    'coefficients', 'p_yrs', '-v7.3');
disp('ROUND03_PRODUCTION_PROXY9_EXPORT_COMPLETE');
end
