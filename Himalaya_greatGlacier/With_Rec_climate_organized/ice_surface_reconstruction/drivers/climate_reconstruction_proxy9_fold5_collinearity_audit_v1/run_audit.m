function run_audit(output_path, phyda_root, proxy9_path)
% Audit bivariate predictor conditioning for fold-5 and archived windows.

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
pl_o1 = nearest_latlon(xlat, xlon, plat(1:290), plon(1:290));
pl_o2 = nearest_latlon(xlat2, xlon2, plat(1:290), plon(1:290));
O_p = O_p / 30;
[~, ref_idx_obs_p] = intersect(x_yrs2, 1961:1990);
O_p_anom = O_p - nanmean(O_p(:, ref_idx_obs_p), 2);

windows = [1901 1980; 1940 2000];
nwin = size(windows, 1);
effective_count = zeros(290, nwin);
predictor_correlation = nan(290, nwin);
vif = nan(290, nwin);
condition_number = nan(290, nwin);
temperature_std = nan(290, nwin);
precipitation_std = nan(290, nwin);

for ww = 1:nwin
    train_years = (windows(ww, 1):windows(ww, 2))';
    [present_t, obs_t] = ismember(train_years, x_yrs(:));
    [present_p, obs_p] = ismember(train_years, x_yrs2(:));
    [present_y, proxy_y] = ismember(train_years, year);
    assert(all(present_t) && all(present_p) && all(present_y), 'window-year-mapping-failed');
    for kk = 1:290
        pd = prxs(proxy_y, kk);
        ot = O_t(pl_o1(kk), obs_t)';
        op = O_p_anom(pl_o2(kk), obs_p)';
        valid = isfinite(pd) & isfinite(ot) & isfinite(op);
        effective_count(kk, ww) = sum(valid);
        if sum(valid) < 3
            continue;
        end
        tv = ot(valid);
        pv = op(valid);
        predictor_correlation(kk, ww) = corr(tv, pv, 'rows', 'complete');
        vif(kk, ww) = 1 / max(1e-12, 1 - predictor_correlation(kk, ww)^2);
        temperature_std(kk, ww) = std(tv);
        precipitation_std(kk, ww) = std(pv);
        if temperature_std(kk, ww) > 0 && precipitation_std(kk, ww) > 0
            X = [(tv - mean(tv)) / temperature_std(kk, ww), ...
                 (pv - mean(pv)) / precipitation_std(kk, ww)];
            condition_number(kk, ww) = cond(X);
        end
    end
end

save(output_path, 'windows', 'effective_count', 'predictor_correlation', ...
    'vif', 'condition_number', 'temperature_std', 'precipitation_std', '-v7.3');
disp('PROXY9_FOLD5_COLLINEARITY_AUDIT_V1_COMPLETE');
end
