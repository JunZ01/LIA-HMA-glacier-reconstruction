function fit_climate_reconstruction_round03_blocked_psm_v1(input_path, output_path, phyda_root)
% Fit proxy PSMs on five allowed 1940-2000 temporal training folds.

old_dir = pwd;
cleanup_dir = onCleanup(@() cd(old_dir));
old_rng = rng;
cleanup_rng = onCleanup(@() rng(old_rng));
cd(phyda_root);
addpath(genpath(phyda_root));

source = load(input_path);
years = round(source.years(:));
observations = source.observations;
locations = source.locations;
archived_hxb = source.archived_hxb;
archived_b = source.archived_b;
model_type = round(source.model_type(:));
nproxy = size(observations, 2);
assert(isequal(years, (1940:2000)'), 'unexpected-allowed-year-axis');
assert(~any(years >= 1921 & years <= 1939), 'locked-period-requested');
assert(size(observations, 1) == 61, 'unexpected-observation-year-count');
assert(size(archived_hxb, 2) == 848, 'unexpected-archived-hxb-width');

plat = locations(:, 1)';
plon = locations(:, 2)';
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
M_t_centered = M_t(pl_m, :) - mean(M_t(pl_m, :), 2, 'omitnan');
M_p_centered = M_p(pl_m, :) - mean(M_p(pl_m, :), 2, 'omitnan');

O_p = O_p / 30;
[~, ref_idx_obs_p] = intersect(x_yrs2, 1961:1990);
O_p_anom = O_p - mean(O_p(:, ref_idx_obs_p), 2, 'omitnan');
[present_t, obs_all_t] = ismember(years, x_yrs(:));
[present_p, obs_all_p] = ismember(years, x_yrs2(:));
assert(all(present_t) && all(present_p), 'instrumental-year-mapping-failed');

test_blocks = [1940 1951; 1952 1963; 1964 1975; 1976 1987; 1988 2000];
nbranch = 1 + size(test_blocks, 1);
hxb = nan(nproxy, nbranch, 848);
b = nan(nproxy, nbranch, 3);
residual_variance = nan(nproxy, nbranch);
training_count = zeros(nproxy, nbranch);
holdout_count = zeros(nproxy, size(test_blocks, 1));
holdout_rmse = nan(nproxy, size(test_blocks, 1));
holdout_correlation = nan(nproxy, size(test_blocks, 1));

for branch = 1:nbranch
    if branch == 1
        train_mask = true(size(years));
        test_mask = false(size(years));
    else
        block = test_blocks(branch - 1, :);
        test_mask = years >= block(1) & years <= block(2);
        train_mask = ~test_mask;
    end
    temp_mean = mean(O_t(pl_o1, obs_all_t(train_mask)), 2, 'omitnan');
    prcp_mean = mean(O_p_anom(pl_o2, obs_all_p(train_mask)), 2, 'omitnan');
    M_tbc = M_t_centered + temp_mean;
    M_pbc = M_p_centered + prcp_mean;

    for kk = 1:nproxy
        pd = observations(train_mask, kk);
        ot = O_t(pl_o1(kk), obs_all_t(train_mask))';
        if model_type(kk) == 2
            op = O_p_anom(pl_o2(kk), obs_all_p(train_mask))';
            valid = isfinite(pd) & isfinite(ot) & isfinite(op);
            training_count(kk, branch) = sum(valid);
            if sum(valid) < 20
                continue;
            end
            X = [ones(sum(valid), 1), ot(valid), op(valid)];
            coef = X \ pd(valid);
            b(kk, branch, :) = coef;
            hxb(kk, branch, :) = coef(1) + coef(2) * M_tbc(kk, :) + coef(3) * M_pbc(kk, :);
            residual = pd(valid) - X * coef;
            if any(test_mask)
                pd_test = observations(test_mask, kk);
                ot_test = O_t(pl_o1(kk), obs_all_t(test_mask))';
                op_test = O_p_anom(pl_o2(kk), obs_all_p(test_mask))';
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
            residual = pd(valid) - X * coef;
            if any(test_mask)
                pd_test = observations(test_mask, kk);
                ot_test = O_t(pl_o1(kk), obs_all_t(test_mask))';
                valid_test = isfinite(pd_test) & isfinite(ot_test);
                prediction = coef(1) + coef(2) * ot_test(valid_test);
            end
        end
        residual_variance(kk, branch) = var(residual, 0, 'omitnan');
        if any(test_mask)
            observed = pd_test(valid_test);
            holdout_count(kk, branch - 1) = sum(valid_test);
            if numel(observed) >= 3
                holdout_rmse(kk, branch - 1) = sqrt(mean((observed - prediction).^2, 'omitnan'));
                if std(observed, 'omitnan') > 0 && std(prediction, 'omitnan') > 0
                    holdout_correlation(kk, branch - 1) = corr(observed, prediction, 'rows', 'complete');
                end
            end
        end
    end
end

baseline_hxb_max_abs = max(abs(squeeze(hxb(:, 1, :)) - archived_hxb), [], 'all', 'omitnan');
baseline_b_max_abs = max(abs(squeeze(b(:, 1, :)) - archived_b), [], 'all', 'omitnan');
assert(baseline_hxb_max_abs <= 1e-9, 'baseline-hxb-closure-failed');
assert(baseline_b_max_abs <= 1e-9, 'baseline-b-closure-failed');
assert(all(isfinite(residual_variance(training_count >= 20))), 'finite-residual-variance-failed');

save(output_path, 'years', 'test_blocks', 'p_yrs', 'rng_seed', 'model_type', ...
    'hxb', 'b', 'residual_variance', 'training_count', 'holdout_count', ...
    'holdout_rmse', 'holdout_correlation', 'baseline_hxb_max_abs', ...
    'baseline_b_max_abs', '-v7.3');
disp('ROUND03_BLOCKED_PSM_COMPLETE');
end
