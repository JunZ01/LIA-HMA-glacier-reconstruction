function diagnose_record205_fold3_v2(output_path, phyda_root, proxy9_path)
% Diagnose the sole rank-deficient fit with explicit column-vector inputs.

old_dir = pwd;
cleanup_dir = onCleanup(@() cd(old_dir));
old_rng = rng;
cleanup_rng = onCleanup(@() rng(old_rng));
cd(phyda_root);
addpath(genpath(phyda_root));

archived = load(proxy9_path, 'proxy_data');
item = archived.proxy_data.full_proxy{205};
year = (1000:2000)';
item_year = round(item.age_yng(:));
[present, positions] = ismember(year, item_year);
assert(all(present), 'reconstruction-year-missing-from-archive-axis');
item_data = item.data(:);
proxy = item_data(positions);
plat = item.location(1);
plon = item.location(2);

[O_t, xlat, xlon, x_yrs] = load_obs_S('bearth', 1, 12);
[O_p, xlat2, xlon2, x_yrs2] = load_obs_S('gpcp', 1, 12);
pl_o1 = nearest_latlon(xlat, xlon, plat, plon);
pl_o2 = nearest_latlon(xlat2, xlon2, plat, plon);
O_p = O_p / 30;
[~, ref_idx_obs_p] = intersect(x_yrs2, 1961:1990);
O_p_anom = O_p - nanmean(O_p(:, ref_idx_obs_p), 2);

train_years = setdiff((1901:1980)', (1941:1960)');
[present_t, obs_train_t] = ismember(train_years, x_yrs(:));
[present_p, obs_train_p] = ismember(train_years, x_yrs2(:));
[present_y, proxy_train] = ismember(train_years, year);
assert(all(present_t) && all(present_p) && all(present_y), 'training-year-mapping-failed');

train_proxy = proxy(proxy_train);
train_temp = O_t(pl_o1, obs_train_t)';
train_prcp = O_p_anom(pl_o2, obs_train_p)';
train_proxy = train_proxy(:);
train_temp = train_temp(:);
train_prcp = train_prcp(:);
valid = isfinite(train_proxy) & isfinite(train_temp) & isfinite(train_prcp);
valid_years = train_years(valid);
proxy_values = train_proxy(valid);
temp_values = train_temp(valid);
prcp_values = train_prcp(valid);
X = [ones(sum(valid), 1), temp_values, prcp_values];
singular_values = svd(X, 0);
design_rank = rank(X);
condition_number = cond(X);
temp_std = std(temp_values);
prcp_std = std(prcp_values);
temp_range = range(temp_values);
prcp_range = range(prcp_values);
temp_prcp_correlation = corr(temp_values, prcp_values, 'rows', 'complete');
constant_temp = temp_range <= eps(max(1, max(abs(temp_values))));
constant_prcp = prcp_range <= eps(max(1, max(abs(prcp_values))));

save(output_path, 'valid_years', 'proxy_values', 'temp_values', 'prcp_values', ...
    'singular_values', 'design_rank', 'condition_number', 'temp_std', ...
    'prcp_std', 'temp_range', 'prcp_range', 'temp_prcp_correlation', ...
    'constant_temp', 'constant_prcp', '-v7.3');
fprintf('RECORD=205 TRAIN=%d YEARS=%d-%d RANK=%d COND=%.17g\n', ...
    sum(valid), min(valid_years), max(valid_years), design_rank, condition_number);
fprintf('TEMP_STD=%.17g TEMP_RANGE=%.17g CONSTANT_TEMP=%d\n', temp_std, temp_range, constant_temp);
fprintf('PRCP_STD=%.17g PRCP_RANGE=%.17g CONSTANT_PRCP=%d\n', prcp_std, prcp_range, constant_prcp);
fprintf('TEMP_PRCP_CORRELATION=%.17g\n', temp_prcp_correlation);
fprintf('SINGULAR_VALUES=%s\n', mat2str(singular_values, 17));
disp('PROXY9_RECORD205_FOLD3_DIAGNOSIS_V2_COMPLETE');
end
