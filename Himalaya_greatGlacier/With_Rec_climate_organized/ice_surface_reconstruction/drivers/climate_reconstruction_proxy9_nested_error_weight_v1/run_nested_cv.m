function run_nested_cv(output_path, phyda_root, proxy9_path)
% Estimate record-wise PSM error from four inner folds within 1901-1980.

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
O_p = O_p / 30;
[~, ref_idx_obs_p] = intersect(x_yrs2, 1961:1990);
O_p_anom = O_p - nanmean(O_p(:, ref_idx_obs_p), 2);

inner_blocks = [1901 1920; 1921 1940; 1941 1960; 1961 1980];
nfold = size(inner_blocks, 1);
squared_error_sum = zeros(297, 1);
prediction_count = zeros(297, 1);
valid_fold_count = zeros(297, 1);
fold_mse = nan(297, nfold);
fold_correlation = nan(297, nfold);
training_count = zeros(297, nfold);

for fold = 1:nfold
    test_years = (inner_blocks(fold, 1):inner_blocks(fold, 2))';
    train_years = setdiff((1901:1980)', test_years);
    [present_t, obs_train_t] = ismember(train_years, x_yrs(:));
    [present_p, obs_train_p] = ismember(train_years, x_yrs2(:));
    [present_y, proxy_train] = ismember(train_years, year);
    [present_tt, obs_test_t] = ismember(test_years, x_yrs(:));
    [present_tp, obs_test_p] = ismember(test_years, x_yrs2(:));
    [present_ty, proxy_test] = ismember(test_years, year);
    assert(all(present_t) && all(present_p) && all(present_y), 'inner-training-year-mapping-failed');
    assert(all(present_tt) && all(present_tp) && all(present_ty), 'inner-test-year-mapping-failed');

    for kk = 1:297
        train_proxy = prxs(proxy_train, kk);
        train_temp = O_t(pl_o1(kk), obs_train_t)';
        test_proxy = prxs(proxy_test, kk);
        test_temp = O_t(pl_o1(kk), obs_test_t)';
        if kk <= 290
            train_prcp = O_p_anom(pl_o2(kk), obs_train_p)';
            test_prcp = O_p_anom(pl_o2(kk), obs_test_p)';
            valid_train = isfinite(train_proxy) & isfinite(train_temp) & isfinite(train_prcp);
            valid_test = isfinite(test_proxy) & isfinite(test_temp) & isfinite(test_prcp);
            training_count(kk, fold) = sum(valid_train);
            if sum(valid_train) < 20
                continue;
            end
            X = [ones(sum(valid_train), 1), train_temp(valid_train), train_prcp(valid_train)];
            coef = X \ train_proxy(valid_train);
            prediction = coef(1) + coef(2) * test_temp(valid_test) + coef(3) * test_prcp(valid_test);
        else
            valid_train = isfinite(train_proxy) & isfinite(train_temp);
            valid_test = isfinite(test_proxy) & isfinite(test_temp);
            training_count(kk, fold) = sum(valid_train);
            if sum(valid_train) < 20
                continue;
            end
            X = [ones(sum(valid_train), 1), train_temp(valid_train)];
            coef = X \ train_proxy(valid_train);
            prediction = coef(1) + coef(2) * test_temp(valid_test);
        end
        observed = test_proxy(valid_test);
        if numel(observed) < 3
            continue;
        end
        errors = observed - prediction;
        fold_mse(kk, fold) = mean(errors.^2);
        squared_error_sum(kk) = squared_error_sum(kk) + sum(errors.^2);
        prediction_count(kk) = prediction_count(kk) + numel(errors);
        valid_fold_count(kk) = valid_fold_count(kk) + 1;
        if std(observed) > 0 && std(prediction) > 0
            fold_correlation(kk, fold) = corr(observed, prediction, 'rows', 'complete');
        end
    end
end

nested_mse = squared_error_sum ./ prediction_count;
nested_mse(prediction_count == 0) = NaN;
save(output_path, 'inner_blocks', 'nested_mse', 'prediction_count', ...
    'valid_fold_count', 'fold_mse', 'fold_correlation', 'training_count', '-v7.3');
disp('PROXY9_NESTED_ERROR_WEIGHT_V1_COMPLETE');
end
