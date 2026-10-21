function run_audit(output_path, phyda_root, proxy9_path, blocked_cv_path)
% Compare observed and predicted proxy behavior in the held-out 1981-2000 block.

old_dir = pwd;
cleanup_dir = onCleanup(@() cd(old_dir));
cd(phyda_root);
addpath(genpath(phyda_root));

archived = load(proxy9_path, 'proxy_data');
full_proxy = archived.proxy_data.full_proxy;
assert(numel(full_proxy) == 297, 'unexpected-record-count');
year = (1000:2000)';
prxs = nan(numel(year), 297);
plat = nan(1, 297);
plon = nan(1, 297);
for ii = 1:297
    item = full_proxy{ii};
    item_year = round(item.age_yng(:));
    [present, positions] = ismember(year, item_year);
    assert(all(present), 'reconstruction-year-missing-from-archive-axis');
    prxs(:, ii) = item.data(positions);
    plat(ii) = item.location(1);
    plon(ii) = item.location(2);
end

[O_t, xlat, xlon, x_yrs] = load_obs_S('bearth', 1, 12);
[O_p, xlat2, xlon2, x_yrs2] = load_obs_S('gpcp', 1, 12);
pl_o1 = nearest_latlon(xlat, xlon, plat, plon);
pl_o2 = nearest_latlon(xlat2, xlon2, plat, plon);
O_p = O_p / 30;
[~, ref_idx_obs_p] = intersect(x_yrs2, 1961:1990);
O_p_anom = O_p - nanmean(O_p(:, ref_idx_obs_p), 2);

cv = load(blocked_cv_path, 'b');
coefficients = cv.b;
test_years = (1981:2000)';
[present_t, obs_t] = ismember(test_years, x_yrs(:));
[present_p, obs_p] = ismember(test_years, x_yrs2(:));
[present_y, proxy_y] = ismember(test_years, year);
assert(all(present_t) && all(present_p) && all(present_y), 'test-year-mapping-failed');

nrecord = 297;
observed_count = zeros(nrecord, 1);
proxy_trend = nan(nrecord, 1);
local_temperature_trend = nan(nrecord, 1);
local_precipitation_trend = nan(nrecord, 1);
proxy_temperature_correlation = nan(nrecord, 1);
proxy_precipitation_correlation = nan(nrecord, 1);
archived_prediction_trend = nan(nrecord, 1);
fold5_prediction_trend = nan(nrecord, 1);
archived_residual_trend = nan(nrecord, 1);
fold5_residual_trend = nan(nrecord, 1);
archived_prediction_correlation = nan(nrecord, 1);
fold5_prediction_correlation = nan(nrecord, 1);
proxy_values = nan(nrecord, numel(test_years));
temperature_values = nan(nrecord, numel(test_years));
precipitation_values = nan(nrecord, numel(test_years));
archived_prediction = nan(nrecord, numel(test_years));
fold5_prediction = nan(nrecord, numel(test_years));

for kk = 1:nrecord
    pd = prxs(proxy_y, kk);
    ot = O_t(pl_o1(kk), obs_t)';
    op = O_p_anom(pl_o2(kk), obs_p)';
    if kk <= 290
        valid = isfinite(pd) & isfinite(ot) & isfinite(op);
        archived_coef = squeeze(coefficients(kk, 1, :));
        fold5_coef = squeeze(coefficients(kk, 7, :));
        archived_pred_all = archived_coef(1) + archived_coef(2) * ot + archived_coef(3) * op;
        fold5_pred_all = fold5_coef(1) + fold5_coef(2) * ot + fold5_coef(3) * op;
    else
        valid = isfinite(pd) & isfinite(ot);
        archived_coef = squeeze(coefficients(kk, 1, 1:2));
        fold5_coef = squeeze(coefficients(kk, 7, 1:2));
        archived_pred_all = archived_coef(1) + archived_coef(2) * ot;
        fold5_pred_all = fold5_coef(1) + fold5_coef(2) * ot;
    end
    observed_count(kk) = sum(valid);
    proxy_values(kk, :) = pd;
    temperature_values(kk, :) = ot;
    precipitation_values(kk, :) = op;
    archived_prediction(kk, :) = archived_pred_all;
    fold5_prediction(kk, :) = fold5_pred_all;
    if sum(valid) < 3
        continue;
    end
    x = test_years(valid);
    proxy_trend(kk) = 100 * polyfit(x, pd(valid), 1) * [1; 0];
    local_temperature_trend(kk) = 100 * polyfit(x, ot(valid), 1) * [1; 0];
    if kk <= 290
        local_precipitation_trend(kk) = 100 * polyfit(x, op(valid), 1) * [1; 0];
        proxy_precipitation_correlation(kk) = corr(pd(valid), op(valid), 'rows', 'complete');
    end
    proxy_temperature_correlation(kk) = corr(pd(valid), ot(valid), 'rows', 'complete');
    archived_prediction_trend(kk) = 100 * polyfit(x, archived_pred_all(valid), 1) * [1; 0];
    fold5_prediction_trend(kk) = 100 * polyfit(x, fold5_pred_all(valid), 1) * [1; 0];
    archived_residual = pd(valid) - archived_pred_all(valid);
    fold5_residual = pd(valid) - fold5_pred_all(valid);
    archived_residual_trend(kk) = 100 * polyfit(x, archived_residual, 1) * [1; 0];
    fold5_residual_trend(kk) = 100 * polyfit(x, fold5_residual, 1) * [1; 0];
    if std(pd(valid)) > 0 && std(archived_pred_all(valid)) > 0
        archived_prediction_correlation(kk) = corr(pd(valid), archived_pred_all(valid), 'rows', 'complete');
    end
    if std(pd(valid)) > 0 && std(fold5_pred_all(valid)) > 0
        fold5_prediction_correlation(kk) = corr(pd(valid), fold5_pred_all(valid), 'rows', 'complete');
    end
end

save(output_path, 'test_years', 'observed_count', 'proxy_trend', ...
    'local_temperature_trend', 'local_precipitation_trend', ...
    'proxy_temperature_correlation', 'proxy_precipitation_correlation', ...
    'archived_prediction_trend', 'fold5_prediction_trend', ...
    'archived_residual_trend', 'fold5_residual_trend', ...
    'archived_prediction_correlation', 'fold5_prediction_correlation', ...
    'proxy_values', 'temperature_values', 'precipitation_values', ...
    'archived_prediction', 'fold5_prediction', '-v7.3');
disp('PROXY9_FOLD5_RESPONSE_DRIFT_V1_COMPLETE');
end
