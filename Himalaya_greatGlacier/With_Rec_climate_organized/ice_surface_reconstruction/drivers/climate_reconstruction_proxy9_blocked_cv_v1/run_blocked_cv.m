function run_blocked_cv(output_path, phyda_root, proxy9_path, baseline_v4_path)
% Fit annual proxy9 PSMs with five contiguous temporal holdout blocks.

old_dir = pwd;
cleanup_dir = onCleanup(@() cd(old_dir));
old_rng = rng;
cleanup_rng = onCleanup(@() rng(old_rng));
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

mdl_prior = 'cesm_lme010';
[gcm_syr, ~] = modelyrs(mdl_prior);
p_yrs = (1001:1848) - gcm_syr + 1;
rng_seed = 0;
rng(rng_seed, 'twister');
[M_t, xmeta] = load_S(mdl_prior, 't', 1, 12, p_yrs);
[M_p, ~] = load_S(mdl_prior, 'p', 1, 12, p_yrs);
pl_m = nearest_latlon(xmeta{1}.lat, xmeta{1}.lon, plat, plon);
M_t_centered = M_t(pl_m, :) - nanmean(M_t(pl_m, :), 2);
M_p_centered = M_p(pl_m, :) - nanmean(M_p(pl_m, :), 2);

O_p = O_p / 30;
[~, ref_idx_obs_p] = intersect(x_yrs2, 1961:1990);
O_p_anom = O_p - nanmean(O_p(:, ref_idx_obs_p), 2);

test_blocks = [1901 1920; 1921 1940; 1941 1960; 1961 1980; 1981 2000];
nfold = size(test_blocks, 1);
nbranch = 2 + nfold;
% Branch 1 repeats 1940-2000; branch 2 uses all 1901-2000; 3:7 are folds.
hxb = nan(297, nbranch, 848);
b = nan(297, nbranch, 3);
training_count = zeros(297, nbranch);
holdout_count = zeros(297, nfold);
holdout_rmse = nan(297, nfold);
holdout_correlation = nan(297, nfold);

for branch = 1:nbranch
    if branch == 1
        train_years = (1940:2000)';
        test_years = [];
    elseif branch == 2
        train_years = (1901:2000)';
        test_years = [];
    else
        fold = branch - 2;
        test_years = (test_blocks(fold, 1):test_blocks(fold, 2))';
        train_years = setdiff((1901:2000)', test_years);
    end

    [present_t, obs_train_t] = ismember(train_years, x_yrs(:));
    [present_p, obs_train_p] = ismember(train_years, x_yrs2(:));
    [present_proxy, proxy_train] = ismember(train_years, year);
    assert(all(present_t) && all(present_p) && all(present_proxy), 'training-year-mapping-failed');
    temp_mean = nanmean(O_t(pl_o1, obs_train_t), 2);
    prcp_mean = nanmean(O_p_anom(pl_o2, obs_train_p), 2);
    M_tbc = M_t_centered + temp_mean;
    M_pbc = M_p_centered + prcp_mean;

    if ~isempty(test_years)
        [present_tt, obs_test_t] = ismember(test_years, x_yrs(:));
        [present_tp, obs_test_p] = ismember(test_years, x_yrs2(:));
        [present_ty, proxy_test] = ismember(test_years, year);
        assert(all(present_tt) && all(present_tp) && all(present_ty), 'test-year-mapping-failed');
    end

    for kk = 1:297
        pd = prxs(proxy_train, kk);
        ot = O_t(pl_o1(kk), obs_train_t)';
        if kk <= 290
            op = O_p_anom(pl_o2(kk), obs_train_p)';
            valid = isfinite(pd) & isfinite(ot) & isfinite(op);
            training_count(kk, branch) = sum(valid);
            if sum(valid) < 20
                continue;
            end
            X = [ones(sum(valid), 1), ot(valid), op(valid)];
            coef = X \ pd(valid);
            b(kk, branch, :) = coef;
            hxb(kk, branch, :) = coef(1) + coef(2) * M_tbc(kk, :) + coef(3) * M_pbc(kk, :);
            if ~isempty(test_years)
                pd_test = prxs(proxy_test, kk);
                ot_test = O_t(pl_o1(kk), obs_test_t)';
                op_test = O_p_anom(pl_o2(kk), obs_test_p)';
                valid_test = isfinite(pd_test) & isfinite(ot_test) & isfinite(op_test);
                prediction = coef(1) + coef(2) * ot_test(valid_test) + coef(3) * op_test(valid_test);
            end
        else
            valid = isfinite(pd) & isfinite(ot);
            training_count(kk, branch) = sum(valid);
            if sum(valid) < 20
                continue;
            end
            X = [ones(sum(valid), 1), ot(valid)];
            coef = X \ pd(valid);
            b(kk, branch, 1:2) = coef;
            hxb(kk, branch, :) = coef(1) + coef(2) * M_tbc(kk, :);
            if ~isempty(test_years)
                pd_test = prxs(proxy_test, kk);
                ot_test = O_t(pl_o1(kk), obs_test_t)';
                valid_test = isfinite(pd_test) & isfinite(ot_test);
                prediction = coef(1) + coef(2) * ot_test(valid_test);
            end
        end
        if ~isempty(test_years)
            observed = pd_test(valid_test);
            holdout_count(kk, fold) = sum(valid_test);
            if numel(observed) >= 3
                holdout_rmse(kk, fold) = sqrt(mean((observed - prediction).^2));
                if std(observed) > 0 && std(prediction) > 0
                    holdout_correlation(kk, fold) = corr(observed, prediction, 'rows', 'complete');
                end
            end
        end
    end
end

baseline = load(baseline_v4_path, 'hxb', 'b');
baseline_hxb = squeeze(baseline.hxb(:, 3, :));
baseline_b = squeeze(baseline.b(:, 3, :));
baseline_hxb_max_abs = max(abs(squeeze(hxb(:, 1, :)) - baseline_hxb), [], 'all', 'omitnan');
baseline_coefficient_max_abs = max(abs(squeeze(b(:, 1, :)) - baseline_b), [], 'all', 'omitnan');
assert(baseline_hxb_max_abs <= 1e-10, 'baseline-hxb-closure-failed');
assert(baseline_coefficient_max_abs <= 1e-10, 'baseline-coefficient-closure-failed');

save(output_path, 'year', 'p_yrs', 'rng_seed', 'test_blocks', 'hxb', 'b', ...
    'training_count', 'holdout_count', 'holdout_rmse', 'holdout_correlation', ...
    'baseline_hxb_max_abs', 'baseline_coefficient_max_abs', '-v7.3');
disp('PROXY9_BLOCKED_CV_V1_COMPLETE');
end
