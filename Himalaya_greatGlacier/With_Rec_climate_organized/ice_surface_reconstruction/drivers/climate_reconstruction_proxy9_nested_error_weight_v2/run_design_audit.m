function run_design_audit(output_path, phyda_root, proxy9_path)
% Save inner-fold design diagnostics without constructing reliability weights.

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
training_count = zeros(297, nfold);
test_count = zeros(297, nfold);
design_columns = zeros(297, nfold);
design_rank = nan(297, nfold);
design_rank_tolerance = nan(297, nfold);
design_condition_number = nan(297, nfold);
rank_deficient = false(297, nfold);
insufficient_training = false(297, nfold);
insufficient_test = false(297, nfold);
eligible_fold = false(297, nfold);

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
        else
            valid_train = isfinite(train_proxy) & isfinite(train_temp);
            valid_test = isfinite(test_proxy) & isfinite(test_temp);
            X = [ones(sum(valid_train), 1), train_temp(valid_train)];
        end
        training_count(kk, fold) = sum(valid_train);
        test_count(kk, fold) = sum(valid_test);
        design_columns(kk, fold) = size(X, 2);
        insufficient_training(kk, fold) = training_count(kk, fold) < 20;
        insufficient_test(kk, fold) = test_count(kk, fold) < 3;
        if insufficient_training(kk, fold)
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
        rank_deficient(kk, fold) = matrix_rank < size(X, 2);
        eligible_fold(kk, fold) = ~rank_deficient(kk, fold) && ~insufficient_test(kk, fold);
    end
end

eligible_fold_count = sum(eligible_fold, 2);
save(output_path, 'inner_blocks', 'training_count', 'test_count', ...
    'design_columns', 'design_rank', 'design_rank_tolerance', ...
    'design_condition_number', 'rank_deficient', 'insufficient_training', ...
    'insufficient_test', 'eligible_fold', 'eligible_fold_count', '-v7.3');

[rr, ff] = find(rank_deficient);
fprintf('RANK_DEFICIENT_FITS=%d\n', numel(rr));
for ii = 1:numel(rr)
    fprintf('RANK_DEFICIENT_RECORD=%d FOLD=%d TRAIN=%d TEST=%d RANK=%d COLS=%d COND=%.17g\n', ...
        rr(ii), ff(ii), training_count(rr(ii), ff(ii)), test_count(rr(ii), ff(ii)), ...
        design_rank(rr(ii), ff(ii)), design_columns(rr(ii), ff(ii)), ...
        design_condition_number(rr(ii), ff(ii)));
end
problem_records = find(eligible_fold_count < 3);
fprintf('RECORDS_BELOW_THREE_ELIGIBLE_FOLDS=%d\n', numel(problem_records));
for ii = 1:numel(problem_records)
    rr = problem_records(ii);
    fprintf('INSUFFICIENT_RECORD=%d ELIGIBLE=%d TRAIN_COUNTS=%s TEST_COUNTS=%s\n', ...
        rr, eligible_fold_count(rr), mat2str(training_count(rr, :)), ...
        mat2str(test_count(rr, :)));
end
disp('PROXY9_NESTED_ERROR_WEIGHT_V2_DESIGN_AUDIT_COMPLETE');
end
