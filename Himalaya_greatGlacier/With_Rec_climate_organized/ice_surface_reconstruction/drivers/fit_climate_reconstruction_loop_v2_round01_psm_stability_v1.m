function fit_climate_reconstruction_loop_v2_round01_psm_stability_v1(input_path, baseline_path, output_path, phyda_root)
% Fit nested-CV ridge and precipitation-only proxy9 PSM variants.

old_dir = pwd;
cleanup_dir = onCleanup(@() cd(old_dir));
old_rng = rng;
cleanup_rng = onCleanup(@() rng(old_rng));
cd(phyda_root);
addpath(genpath(phyda_root));

s = load(input_path);
base = load(baseline_path, 'hxb', 'residual_variance', 'training_count');
years = round(s.years(:));
observations = s.observations;
locations = s.locations;
model_type = round(s.model_type(:));
nproxy = size(observations, 2);
assert(isequal(years, (1940:2000)'), 'unexpected-allowed-year-axis');
assert(~any(years >= 1921 & years <= 1939), 'locked-period-requested');
assert(nproxy == 297, 'unexpected-proxy-count');

plat = locations(:, 1)';
plon = locations(:, 2)';
[O_t, xlat, xlon, x_yrs] = load_obs_S('bearth', 1, 12);
[O_p, xlat2, xlon2, x_yrs2] = load_obs_S('gpcp', 1, 12);
pl_o1 = nearest_latlon(xlat, xlon, plat, plon);
pl_o2 = nearest_latlon(xlat2, xlon2, plat, plon);
[gcm_syr, ~] = modelyrs('cesm_lme010');
p_yrs = (1001:1848) - gcm_syr + 1;
rng_seed = 0;
rng(rng_seed, 'twister');
[M_t, xmeta] = load_S('cesm_lme010', 't', 1, 12, p_yrs);
[M_p, ~] = load_S('cesm_lme010', 'p', 1, 12, p_yrs);
pl_m = nearest_latlon(xmeta{1}.lat, xmeta{1}.lon, plat, plon);
M_t_centered = M_t(pl_m, :) - mean(M_t(pl_m, :), 2, 'omitnan');
M_p_centered = M_p(pl_m, :) - mean(M_p(pl_m, :), 2, 'omitnan');
O_p = O_p / 30;
[~, ref_idx] = intersect(x_yrs2, 1961:1990);
O_p_anom = O_p - mean(O_p(:, ref_idx), 2, 'omitnan');
[ok_t, obs_t] = ismember(years, x_yrs(:));
[ok_p, obs_p] = ismember(years, x_yrs2(:));
assert(all(ok_t) && all(ok_p), 'instrumental-year-mapping-failed');

test_blocks = [1940 1951; 1952 1963; 1964 1975; 1976 1987; 1988 2000];
lambda_grid = [0 0.01 0.1 1 10 100];
nbranch = 6;
ridge_hxb = nan(nproxy, nbranch, 848);
precip_only_hxb = nan(nproxy, nbranch, 848);
ridge_residual_variance = nan(nproxy, nbranch);
precip_only_residual_variance = nan(nproxy, nbranch);
ridge_coefficients = nan(nproxy, nbranch, 3);
precip_only_coefficients = nan(nproxy, nbranch, 2);
ridge_lambda = nan(nproxy, nbranch);
ridge_training_count = zeros(nproxy, nbranch);
precip_only_training_count = zeros(nproxy, nbranch);
ridge_inner_validation_count = zeros(nproxy, nbranch);
ridge_holdout_rmse = nan(nproxy, 5);
precip_only_holdout_rmse = nan(nproxy, 5);

for branch = 1:nbranch
    if branch == 1
        train = true(size(years));
        test = false(size(years));
    else
        block = test_blocks(branch - 1, :);
        test = years >= block(1) & years <= block(2);
        train = ~test;
    end
    temp_mean = mean(O_t(pl_o1, obs_t(train)), 2, 'omitnan');
    prcp_mean = mean(O_p_anom(pl_o2, obs_p(train)), 2, 'omitnan');
    M_tbc = M_t_centered + temp_mean;
    M_pbc = M_p_centered + prcp_mean;

    for kk = 1:nproxy
        if model_type(kk) ~= 2
            ridge_hxb(kk, branch, :) = base.hxb(kk, branch, :);
            precip_only_hxb(kk, branch, :) = base.hxb(kk, branch, :);
            ridge_residual_variance(kk, branch) = base.residual_variance(kk, branch);
            precip_only_residual_variance(kk, branch) = base.residual_variance(kk, branch);
            ridge_training_count(kk, branch) = base.training_count(kk, branch);
            precip_only_training_count(kk, branch) = base.training_count(kk, branch);
            continue;
        end

        pd = observations(:, kk);
        ot = O_t(pl_o1(kk), obs_t)';
        op = O_p_anom(pl_o2(kk), obs_p)';
        valid_outer = train & isfinite(pd) & isfinite(ot) & isfinite(op);
        if sum(valid_outer) < 20
            continue;
        end

        lambda_scores = nan(size(lambda_grid));
        lambda_counts = zeros(size(lambda_grid));
        for ll = 1:numel(lambda_grid)
            squared_error = 0;
            validation_count = 0;
            for inner = 1:size(test_blocks, 1)
                inner_test = train & years >= test_blocks(inner, 1) & years <= test_blocks(inner, 2) ...
                    & isfinite(pd) & isfinite(ot) & isfinite(op);
                inner_train = train & ~inner_test & isfinite(pd) & isfinite(ot) & isfinite(op);
                if sum(inner_train) < 20 || sum(inner_test) < 3
                    continue;
                end
                coef_inner = ridge_fit([ot(inner_train), op(inner_train)], pd(inner_train), lambda_grid(ll));
                prediction = coef_inner(1) + coef_inner(2) * ot(inner_test) + coef_inner(3) * op(inner_test);
                squared_error = squared_error + sum((pd(inner_test) - prediction).^2, 'omitnan');
                validation_count = validation_count + sum(inner_test);
            end
            if validation_count > 0
                lambda_scores(ll) = squared_error / validation_count;
                lambda_counts(ll) = validation_count;
            end
        end
        available = find(isfinite(lambda_scores));
        if isempty(available)
            selected_lambda = 1;
            inner_count = 0;
        else
            [~, local_min] = min(lambda_scores(available));
            selected_index = available(local_min);
            selected_lambda = lambda_grid(selected_index);
            inner_count = lambda_counts(selected_index);
        end
        ridge_coef = ridge_fit([ot(valid_outer), op(valid_outer)], pd(valid_outer), selected_lambda);
        ridge_prediction = ridge_coef(1) + ridge_coef(2) * ot(valid_outer) + ridge_coef(3) * op(valid_outer);
        ridge_residual = pd(valid_outer) - ridge_prediction;
        ridge_hxb(kk, branch, :) = ridge_coef(1) + ridge_coef(2) * M_tbc(kk, :) + ridge_coef(3) * M_pbc(kk, :);
        ridge_residual_variance(kk, branch) = var(ridge_residual, 0, 'omitnan');
        ridge_coefficients(kk, branch, :) = ridge_coef;
        ridge_lambda(kk, branch) = selected_lambda;
        ridge_training_count(kk, branch) = sum(valid_outer);
        ridge_inner_validation_count(kk, branch) = inner_count;

        valid_precip = train & isfinite(pd) & isfinite(op);
        if sum(valid_precip) >= 20
            precip_coef = [ones(sum(valid_precip), 1), op(valid_precip)] \ pd(valid_precip);
            precip_prediction = precip_coef(1) + precip_coef(2) * op(valid_precip);
            precip_residual = pd(valid_precip) - precip_prediction;
            precip_only_hxb(kk, branch, :) = precip_coef(1) + precip_coef(2) * M_pbc(kk, :);
            precip_only_residual_variance(kk, branch) = var(precip_residual, 0, 'omitnan');
            precip_only_coefficients(kk, branch, :) = precip_coef;
            precip_only_training_count(kk, branch) = sum(valid_precip);
        end

        if branch > 1
            valid_test = test & isfinite(pd) & isfinite(ot) & isfinite(op);
            if sum(valid_test) >= 3
                prediction = ridge_coef(1) + ridge_coef(2) * ot(valid_test) + ridge_coef(3) * op(valid_test);
                ridge_holdout_rmse(kk, branch - 1) = sqrt(mean((pd(valid_test) - prediction).^2, 'omitnan'));
                if all(isfinite(precip_only_coefficients(kk, branch, :)), 'all')
                    precip_coef = squeeze(precip_only_coefficients(kk, branch, :));
                    prediction = precip_coef(1) + precip_coef(2) * op(valid_test);
                    precip_only_holdout_rmse(kk, branch - 1) = sqrt(mean((pd(valid_test) - prediction).^2, 'omitnan'));
                end
            end
        end
    end
end

assert(all(isfinite(ridge_residual_variance(ridge_training_count >= 20))), 'ridge-residual-variance-failed');
assert(all(isfinite(precip_only_residual_variance(precip_only_training_count >= 20))), 'precip-residual-variance-failed');
save(output_path, 'years', 'test_blocks', 'lambda_grid', 'p_yrs', 'rng_seed', ...
    'model_type', 'ridge_hxb', 'precip_only_hxb', ...
    'ridge_residual_variance', 'precip_only_residual_variance', ...
    'ridge_coefficients', 'precip_only_coefficients', 'ridge_lambda', ...
    'ridge_training_count', 'precip_only_training_count', ...
    'ridge_inner_validation_count', 'ridge_holdout_rmse', ...
    'precip_only_holdout_rmse', '-v7.3');
disp('LOOP_V2_ROUND01_PSM_STABILITY_COMPLETE');
end

function coef = ridge_fit(predictors, response, lambda)
mu = mean(predictors, 1, 'omitnan');
sigma = std(predictors, 0, 1, 'omitnan');
assert(all(isfinite(sigma)) && all(sigma > 0), 'ridge-predictor-scale-failed');
standardized = (predictors - mu) ./ sigma;
response_mean = mean(response, 'omitnan');
beta = (standardized' * standardized + lambda * eye(size(standardized, 2))) ...
    \ (standardized' * (response - response_mean));
raw = beta ./ sigma';
intercept = response_mean - mu * raw;
coef = [intercept; raw];
end
