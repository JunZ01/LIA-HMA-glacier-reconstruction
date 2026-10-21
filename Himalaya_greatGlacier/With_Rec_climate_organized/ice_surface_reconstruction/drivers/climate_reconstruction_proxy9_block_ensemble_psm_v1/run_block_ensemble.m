function run_block_ensemble(output_path, phyda_root, proxy9_path)
% Fit local 20-year PSMs and construct rank-gated block ensembles.

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
    item_data = item.data(:);
    [present, positions] = ismember(year, item_year);
    assert(all(present), 'reconstruction-year-missing-from-archive-axis');
    prxs(:, ii) = item_data(positions);
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

blocks = [1901 1920; 1921 1940; 1941 1960; 1961 1980];
nblock = size(blocks, 1);
local_hxb = nan(297, nblock, 848);
local_b = nan(297, nblock, 3);
local_training_count = zeros(297, nblock);
local_design_rank = nan(297, nblock);
local_condition_number = nan(297, nblock);
local_eligible = false(297, nblock);

for bb = 1:nblock
    train_years = (blocks(bb, 1):blocks(bb, 2))';
    [present_t, obs_train_t] = ismember(train_years, x_yrs(:));
    [present_p, obs_train_p] = ismember(train_years, x_yrs2(:));
    [present_y, proxy_train] = ismember(train_years, year);
    assert(all(present_t) && all(present_p) && all(present_y), 'local-training-year-mapping-failed');
    temp_mean = nanmean(O_t(pl_o1, obs_train_t), 2);
    prcp_mean = nanmean(O_p_anom(pl_o2, obs_train_p), 2);
    M_tbc = M_t_centered + temp_mean;
    M_pbc = M_p_centered + prcp_mean;

    for kk = 1:297
        pd = prxs(proxy_train, kk);
        ot = O_t(pl_o1(kk), obs_train_t)';
        if kk <= 290
            op = O_p_anom(pl_o2(kk), obs_train_p)';
            valid = isfinite(pd) & isfinite(ot) & isfinite(op);
            X = [ones(sum(valid), 1), ot(valid), op(valid)];
        else
            valid = isfinite(pd) & isfinite(ot);
            X = [ones(sum(valid), 1), ot(valid)];
        end
        local_training_count(kk, bb) = sum(valid);
        if sum(valid) < 20
            continue;
        end
        singular_values = svd(X, 0);
        tolerance = max(size(X)) * eps(max(singular_values));
        local_design_rank(kk, bb) = sum(singular_values > tolerance);
        if min(singular_values) == 0
            local_condition_number(kk, bb) = Inf;
        else
            local_condition_number(kk, bb) = max(singular_values) / min(singular_values);
        end
        if local_design_rank(kk, bb) < size(X, 2)
            continue;
        end
        coef = X \ pd(valid);
        local_b(kk, bb, 1:numel(coef)) = coef;
        if kk <= 290
            local_hxb(kk, bb, :) = coef(1) + coef(2) * M_tbc(kk, :) + coef(3) * M_pbc(kk, :);
        else
            local_hxb(kk, bb, :) = coef(1) + coef(2) * M_tbc(kk, :);
        end
        local_eligible(kk, bb) = true;
    end
end

% Methods: 1=mean, 2=median. Pooled controls come from the verified blocked-CV file.
crossfit_hxb = nan(297, 2, nblock, 848);
crossfit_available_count = zeros(297, nblock);
crossfit_fallback_required = false(297, nblock);
holdout_count = zeros(297, nblock);
holdout_rmse = nan(297, 2, nblock);
holdout_correlation = nan(297, 2, nblock);

for fold = 1:nblock
    train_blocks = setdiff(1:nblock, fold);
    test_years = (blocks(fold, 1):blocks(fold, 2))';
    [present_tt, obs_test_t] = ismember(test_years, x_yrs(:));
    [present_tp, obs_test_p] = ismember(test_years, x_yrs2(:));
    [present_ty, proxy_test] = ismember(test_years, year);
    assert(all(present_tt) && all(present_tp) && all(present_ty), 'local-test-year-mapping-failed');

    for kk = 1:297
        available = train_blocks(local_eligible(kk, train_blocks));
        crossfit_available_count(kk, fold) = numel(available);
        if numel(available) < 2
            crossfit_fallback_required(kk, fold) = true;
            continue;
        end
        values = squeeze(local_hxb(kk, available, :));
        crossfit_hxb(kk, 1, fold, :) = mean(values, 1);
        crossfit_hxb(kk, 2, fold, :) = median(values, 1);

        pd_test = prxs(proxy_test, kk);
        ot_test = O_t(pl_o1(kk), obs_test_t)';
        if kk <= 290
            op_test = O_p_anom(pl_o2(kk), obs_test_p)';
            valid_test = isfinite(pd_test) & isfinite(ot_test) & isfinite(op_test);
        else
            valid_test = isfinite(pd_test) & isfinite(ot_test);
        end
        observed = pd_test(valid_test);
        holdout_count(kk, fold) = numel(observed);
        if numel(observed) < 3
            continue;
        end
        local_predictions = nan(numel(observed), numel(available));
        for jj = 1:numel(available)
            coef = squeeze(local_b(kk, available(jj), :));
            if kk <= 290
                local_predictions(:, jj) = coef(1) + coef(2) * ot_test(valid_test) + coef(3) * op_test(valid_test);
            else
                local_predictions(:, jj) = coef(1) + coef(2) * ot_test(valid_test);
            end
        end
        predictions = [mean(local_predictions, 2), median(local_predictions, 2)];
        for mm = 1:2
            holdout_rmse(kk, mm, fold) = sqrt(mean((observed - predictions(:, mm)).^2));
            if std(observed) > 0 && std(predictions(:, mm)) > 0
                holdout_correlation(kk, mm, fold) = corr(observed, predictions(:, mm), 'rows', 'complete');
            end
        end
    end
end

production_hxb = nan(297, 2, 848);
production_available_count = sum(local_eligible, 2);
production_fallback_required = production_available_count < 2;
for kk = 1:297
    available = find(local_eligible(kk, :));
    if numel(available) < 2
        continue;
    end
    values = squeeze(local_hxb(kk, available, :));
    production_hxb(kk, 1, :) = mean(values, 1);
    production_hxb(kk, 2, :) = median(values, 1);
end

save(output_path, 'year', 'p_yrs', 'rng_seed', 'blocks', 'local_hxb', ...
    'local_b', 'local_training_count', 'local_design_rank', ...
    'local_condition_number', 'local_eligible', 'crossfit_hxb', ...
    'crossfit_available_count', 'crossfit_fallback_required', ...
    'holdout_count', 'holdout_rmse', 'holdout_correlation', ...
    'production_hxb', 'production_available_count', ...
    'production_fallback_required', '-v7.3');
fprintf('LOCAL_ELIGIBLE_FITS=%d OF %d\n', sum(local_eligible, 'all'), numel(local_eligible));
fprintf('CROSSFIT_FALLBACK_RECORD_FOLDS=%d\n', sum(crossfit_fallback_required, 'all'));
fprintf('PRODUCTION_FALLBACK_RECORDS=%d\n', sum(production_fallback_required));
disp('PROXY9_BLOCK_ENSEMBLE_PSM_V1_COMPLETE');
end
