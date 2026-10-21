function run_temperature_only_cv(output_path, phyda_root, proxy9_path)
% Fit annual temperature-only PSMs under the existing five blocked folds.

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
pl_o = nearest_latlon(xlat, xlon, plat, plon);
mdl_prior = 'cesm_lme010';
[gcm_syr, ~] = modelyrs(mdl_prior);
p_yrs = (1001:1848) - gcm_syr + 1;
rng_seed = 0;
rng(rng_seed, 'twister');
[M_t, xmeta] = load_S(mdl_prior, 't', 1, 12, p_yrs);
pl_m = nearest_latlon(xmeta{1}.lat, xmeta{1}.lon, plat, plon);
M_t_centered = M_t(pl_m, :) - nanmean(M_t(pl_m, :), 2);

test_blocks = [1901 1920; 1921 1940; 1941 1960; 1961 1980; 1981 2000];
nfold = size(test_blocks, 1);
nbranch = 1 + nfold;
% Branch 1 is full 1901-2000; branches 2:6 are blocked folds.
hxb = nan(297, nbranch, 848);
b = nan(297, nbranch, 2);
training_count = zeros(297, nbranch);
holdout_count = zeros(297, nfold);
holdout_rmse = nan(297, nfold);
holdout_correlation = nan(297, nfold);

for branch = 1:nbranch
    if branch == 1
        train_years = (1901:2000)';
        test_years = [];
    else
        fold = branch - 1;
        test_years = (test_blocks(fold, 1):test_blocks(fold, 2))';
        train_years = setdiff((1901:2000)', test_years);
    end
    [present_t, obs_train] = ismember(train_years, x_yrs(:));
    [present_y, proxy_train] = ismember(train_years, year);
    assert(all(present_t) && all(present_y), 'training-year-mapping-failed');
    temp_mean = nanmean(O_t(pl_o, obs_train), 2);
    M_tbc = M_t_centered + temp_mean;
    if ~isempty(test_years)
        [present_tt, obs_test] = ismember(test_years, x_yrs(:));
        [present_ty, proxy_test] = ismember(test_years, year);
        assert(all(present_tt) && all(present_ty), 'test-year-mapping-failed');
    end
    for kk = 1:297
        pd = prxs(proxy_train, kk);
        ot = O_t(pl_o(kk), obs_train)';
        valid = isfinite(pd) & isfinite(ot);
        training_count(kk, branch) = sum(valid);
        if sum(valid) < 20
            continue;
        end
        X = [ones(sum(valid), 1), ot(valid)];
        coef = X \ pd(valid);
        b(kk, branch, :) = coef;
        hxb(kk, branch, :) = coef(1) + coef(2) * M_tbc(kk, :);
        if ~isempty(test_years)
            pd_test = prxs(proxy_test, kk);
            ot_test = O_t(pl_o(kk), obs_test)';
            valid_test = isfinite(pd_test) & isfinite(ot_test);
            observed = pd_test(valid_test);
            prediction = coef(1) + coef(2) * ot_test(valid_test);
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

save(output_path, 'year', 'p_yrs', 'rng_seed', 'test_blocks', 'hxb', 'b', ...
    'training_count', 'holdout_count', 'holdout_rmse', 'holdout_correlation', '-v7.3');
disp('PROXY9_TEMPERATURE_ONLY_BLOCKED_CV_V1_COMPLETE');
end
