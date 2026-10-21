function fit_climate_reconstruction_loop_v2_round05_cross_product_psm_v1(proxy_path, output_path, phyda_root)
% Fit identical blocked bivariate PSMs with GPCC and CRU precipitation.

old_dir = pwd;
cleanup_dir = onCleanup(@() cd(old_dir));
cd(phyda_root);
addpath(genpath(phyda_root));

s = load(proxy_path, 'years', 'observations', 'locations', 'model_type');
years = round(s.years(:));
observations = s.observations;
locations = s.locations;
model_type = round(s.model_type(:));
assert(isequal(years, (1940:2000)'), 'unexpected-allowed-year-axis');
assert(~any(years >= 1921 & years <= 1939), 'locked-period-requested');
assert(isequal(size(observations), [61 297]), 'unexpected-observation-shape');
assert(sum(model_type == 2) == 290, 'unexpected-precipitation-proxy-count');

plat = locations(:, 1)';
plon = locations(:, 2)';
[O_t, xlat_t, xlon_t, years_t] = load_obs_S('bearth', 1, 12);
[O_gpcc, xlat_g, xlon_g, years_g] = load_obs_S('gpcp', 1, 12);
close all force;
[O_cru, xlat_c, xlon_c, years_c] = load_obs_S('cru_ts4_pre', 1, 12);
close all force;

site_t = nearest_latlon(xlat_t, xlon_t, plat, plon);
site_g = nearest_latlon(xlat_g, xlon_g, plat, plon);
site_c = nearest_latlon(xlat_c, xlon_c, plat, plon);
[ok_t, index_t] = ismember(years, years_t(:));
[ok_g, index_g] = ismember(years, years_g(:));
[ok_c, index_c] = ismember(years, years_c(:));
assert(all(ok_t) && all(ok_g) && all(ok_c), 'instrumental-year-mapping-failed');

O_gpcc = O_gpcc / 30;
O_cru = O_cru / 30;
[~, ref_g] = intersect(years_g(:), (1961:1990)');
[~, ref_c] = intersect(years_c(:), (1961:1990)');
O_gpcc = O_gpcc - mean(O_gpcc(:, ref_g), 2, 'omitnan');
O_cru = O_cru - mean(O_cru(:, ref_c), 2, 'omitnan');

temperature = O_t(site_t, index_t)';
precipitation = nan(61, 297, 2);
precipitation(:, :, 1) = O_gpcc(site_g, index_g)';
precipitation(:, :, 2) = O_cru(site_c, index_c)';
product_names = {'GPCC_v2022', 'CRU_TS4_04'};
test_blocks = [1940 1951; 1952 1963; 1964 1975; 1976 1987; 1988 2000];
nbranch = 6;
nproduct = 2;
nproxy = size(observations, 2);

coefficients = nan(nproxy, nbranch, 3, nproduct);
standardized_temperature_effect = nan(nproxy, nbranch, nproduct);
standardized_precipitation_effect = nan(nproxy, nbranch, nproduct);
residual_variance = nan(nproxy, nbranch, nproduct);
training_count = zeros(nproxy, nbranch, nproduct);
holdout_rmse = nan(nproxy, 5, nproduct, nproduct);
holdout_count = zeros(nproxy, 5, nproduct, nproduct);

for branch = 1:nbranch
    if branch == 1
        train = true(size(years));
        test = false(size(years));
    else
        block = test_blocks(branch - 1, :);
        test = years >= block(1) & years <= block(2);
        train = ~test;
    end
    for kk = 1:nproxy
        if model_type(kk) ~= 2
            continue;
        end
        pd = observations(:, kk);
        ot = temperature(:, kk);
        for fit_product = 1:nproduct
            op_fit = precipitation(:, kk, fit_product);
            valid_train = train & isfinite(pd) & isfinite(ot) & isfinite(op_fit);
            if sum(valid_train) < 20
                continue;
            end
            predictors = [ones(sum(valid_train), 1), ot(valid_train), op_fit(valid_train)];
            coef = predictors \ pd(valid_train);
            prediction = predictors * coef;
            residual = pd(valid_train) - prediction;
            coefficients(kk, branch, :, fit_product) = coef;
            standardized_temperature_effect(kk, branch, fit_product) = coef(2) * std(ot(valid_train), 0, 'omitnan');
            standardized_precipitation_effect(kk, branch, fit_product) = coef(3) * std(op_fit(valid_train), 0, 'omitnan');
            residual_variance(kk, branch, fit_product) = var(residual, 0, 'omitnan');
            training_count(kk, branch, fit_product) = sum(valid_train);

            if branch > 1
                for eval_product = 1:nproduct
                    op_eval = precipitation(:, kk, eval_product);
                    valid_test = test & isfinite(pd) & isfinite(ot) & isfinite(op_eval);
                    if sum(valid_test) < 3
                        continue;
                    end
                    test_prediction = coef(1) + coef(2) * ot(valid_test) + coef(3) * op_eval(valid_test);
                    holdout_rmse(kk, branch - 1, fit_product, eval_product) = ...
                        sqrt(mean((pd(valid_test) - test_prediction).^2, 'omitnan'));
                    holdout_count(kk, branch - 1, fit_product, eval_product) = sum(valid_test);
                end
            end
        end
    end
end

assert(all(isfinite(residual_variance(training_count >= 20))), 'residual-variance-failed');
save(output_path, 'years', 'test_blocks', 'product_names', 'model_type', ...
    'temperature', 'precipitation', 'coefficients', ...
    'standardized_temperature_effect', 'standardized_precipitation_effect', ...
    'residual_variance', 'training_count', 'holdout_rmse', 'holdout_count', ...
    'site_t', 'site_g', 'site_c', '-v7.3');
disp('LOOP_V2_ROUND05_CROSS_PRODUCT_PSM_COMPLETE');
end
