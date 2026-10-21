function run_nested_cv_v2(output_path, phyda_root, proxy9_path)
% Estimate nested-CV PSM error while explicitly excluding rank-deficient fits.

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
test_count = zeros(297, nfold);
design_columns = zeros(297, nfold);
design_rank = nan(297, nfold);
design_rank_tolerance = nan(297, nfold);
design_condition_number = nan(297, nfold);
rank_deficient = false(297, nfold);
insufficient_training = false(297, nfold);
insufficient_test = false(297, nfold);

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
            X = [ones(sum(valid_train), 1), train_temp(valid_train), train_prcp(valid_train)];
            Xtest = [ones(sum(valid_test), 1), test_temp(valid_test), test_prcp(valid_test)];
        else
            valid_train = isfinite(train_proxy) & isfinite(train_temp);
            valid_test = isfinite(test_proxy) & isfinite(test_temp);
            X = [ones(sum(valid_train), 1), train_temp(valid_train)];
            Xtest = [ones(sum(valid_test), 1), test_temp(valid_test)];
        end
        training_count(kk, fold) = sum(valid_train);
        test_count(kk, fold) = sum(valid_test);
        design_columns(kk, fold) = size(X, 2);
        if training_count(kk, fold) < 20
            insufficient_training(kk, fold) = true;
            continue;
        end

        singular_values = svd(X, 0);
        max_singular = max(singular_values);
        tolerance = max(size(X)) * eps(max_singular);
        matrix_rank = sum(singular_values > tolerance);
        design_rank(kk, fold) = matrix_rank;
        design_rank_tolerance(kk, fold) = tolerance;
        if min(singular_values) == 0
            design_condition_number(kk, fold) = Inf;
        else
            design_condition_number(kk, fold) = max_singular / min(singular_values);
        end
        if matrix_rank < size(X, 2)
            rank_deficient(kk, fold) = true;
            continue;
        end
        if test_count(kk, fold) < 3
            insufficient_test(kk, fold) = true;
            continue;
        end

        coef = X \ train_proxy(valid_train);
        prediction = Xtest * coef;
        observed = test_proxy(valid_test);
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
eligible_nested_mse = nested_mse;
eligible_nested_mse(valid_fold_count < 3) = NaN;
rank_deficient_linear_index = find(rank_deficient);
[rank_deficient_record, rank_deficient_fold] = ind2sub(size(rank_deficient), rank_deficient_linear_index);

assert(all(isfinite(eligible_nested_mse)), 'record-with-fewer-than-three-valid-inner-folds');
assert(min(valid_fold_count) >= 3, 'valid-inner-fold-gate-failed');
save(output_path, 'inner_blocks', 'nested_mse', 'eligible_nested_mse', ...
    'prediction_count', 'valid_fold_count', 'fold_mse', 'fold_correlation', ...
    'training_count', 'test_count', 'design_columns', 'design_rank', ...
    'design_rank_tolerance', 'design_condition_number', 'rank_deficient', ...
    'rank_deficient_record', 'rank_deficient_fold', 'insufficient_training', ...
    'insufficient_test', '-v7.3');
fprintf('RANK_DEFICIENT_FITS=%d\n', numel(rank_deficient_record));
for ii = 1:numel(rank_deficient_record)
    rr = rank_deficient_record(ii);
    ff = rank_deficient_fold(ii);
    fprintf('RANK_DEFICIENT_RECORD=%d FOLD=%d TRAIN=%d RANK=%d COLS=%d COND=%.17g\n', ...
        rr, ff, training_count(rr, ff), design_rank(rr, ff), ...
        design_columns(rr, ff), design_condition_number(rr, ff));
end
fprintf('MIN_VALID_FOLDS=%d\n', min(valid_fold_count));
disp('PROXY9_NESTED_ERROR_WEIGHT_V2_COMPLETE');
end
