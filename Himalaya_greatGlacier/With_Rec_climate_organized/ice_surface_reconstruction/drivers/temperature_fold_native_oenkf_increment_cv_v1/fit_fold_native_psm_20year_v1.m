function fit_fold_native_psm_20year_v1(allowed_input_path, prior_input_path, replay_input_path, model_predictor_input_path, output_path, phyda_root, bearth_path, gpcc_path)
% Build full and three 20-year holdout PSM operators with production RegEM R.

old_dir = pwd;
cleanup_dir = onCleanup(@() cd(old_dir));
cd(phyda_root);
addpath(phyda_root);
addpath(fullfile(phyda_root, 'input-recon', 'RegEM-master'));

S = load(allowed_input_path);
P = load(prior_input_path, 'temperature_prior_336x848', 'experiment1_sample_indices');
R = load(replay_input_path, 'sample_indices', 'HXb0', 'R0v_filled');
years = double(S.years(:));
assert(isequal(years, (1940:2000)'), 'unexpected-allowed-year-axis');
assert(~any(years >= 1921 & years <= 1939), 'locked-period-requested');
assert(size(P.temperature_prior_336x848, 2) == 848, 'unexpected-prior-member-count');

sample_indices = double(P.experiment1_sample_indices(:)');
replay_indices = double(R.sample_indices(:)');
assert(numel(sample_indices) == 500, 'experiment1-sample-count-not-500');
assert(numel(unique(sample_indices)) == 500, 'experiment1-sample-indices-not-unique');
assert(all(sample_indices >= 1 & sample_indices <= 848), 'experiment1-sample-index-range-failed');
assert(isequal(sample_indices, replay_indices), 'experiment1-replay-index-closure-failed');
experiment1_index_closure_exact = true;

[O_t, tlat, tlon] = read_bearth_allowed(bearth_path);
[O_p, plat, plon] = read_gpcc_allowed(gpcc_path);
assert(size(O_t, 2) == numel(years) && size(O_p, 2) == numel(years), 'allowed-observation-year-count-failed');

locations = double(S.locations);
proxy_lat = locations(:, 1)';
proxy_lon = locations(:, 2)';
obs_t_index = nearest_latlon(tlat, tlon, proxy_lat, proxy_lon);
obs_p_index = nearest_latlon(plat, plon, proxy_lat, proxy_lon);
proxy_t = O_t(obs_t_index, :);
proxy_p = O_p(obs_p_index, :);

archived_b = double(S.archived_b);
archived_hxb = double(S.archived_hxb);
model_type = double(S.model_type(:));
nproxy = size(S.observations, 2);
assert(size(archived_hxb, 1) == nproxy && size(archived_hxb, 2) == 848, 'archived-hxb-shape-failed');
assert(isequal(size(R.HXb0), [nproxy 500]), 'replay-hxb-shape-failed');
assert(size(R.R0v_filled, 1) == nproxy, 'replay-R-shape-failed');
M = load(model_predictor_input_path, 'model_t_848', 'model_p_848', 'source_hxb_closure_max_abs');
model_t = double(M.model_t_848);
model_p = double(M.model_p_848);
source_hxb_closure_max_abs = double(M.source_hxb_closure_max_abs);
assert(isequal(size(model_t), [nproxy 848]), 'model-t-shape-failed');
assert(isequal(size(model_p), [nproxy 848]), 'model-p-shape-failed');
assert(source_hxb_closure_max_abs <= 1e-9, 'proxy-model-predictor-source-closure-failed');

blocks = [1940 1959; 1960 1979; 1980 2000];
branch_outer = [0; 1; 2; 3];
branch_inner = zeros(4, 1);
hxb = nan(nproxy, 4, 848);
b = nan(nproxy, 4, 3);
residual_variance = nan(nproxy, 4);
training_count = zeros(nproxy, 4);
observations = double(S.observations);

for branch = 1:4
    train = true(numel(years), 1);
    outer = branch_outer(branch);
    if outer > 0
        train = train & ~(years >= blocks(outer, 1) & years <= blocks(outer, 2));
    end

    rvec = zeros(nproxy, sum(train));
    rvec(model_type ~= 2, :) = NaN;
    for proxy = 1:nproxy
        response = observations(train, proxy);
        temperature = proxy_t(proxy, train)';
        if model_type(proxy) == 2
            precipitation = proxy_p(proxy, train)';
            valid = isfinite(response) & isfinite(temperature) & isfinite(precipitation);
            if sum(valid) < 20
                continue;
            end
            X = [ones(sum(valid), 1), temperature(valid), precipitation(valid)];
            coefficient = X \ response(valid);
            b(proxy, branch, :) = coefficient;
            hxb(proxy, branch, :) = coefficient(1) + coefficient(2) * model_t(proxy, :) + coefficient(3) * model_p(proxy, :);
        else
            valid = isfinite(response) & isfinite(temperature);
            if sum(valid) < 20
                continue;
            end
            X = [ones(sum(valid), 1), temperature(valid)];
            coefficient = X \ response(valid);
            b(proxy, branch, 1:2) = coefficient;
            hxb(proxy, branch, :) = coefficient(1) + coefficient(2) * model_t(proxy, :);
        end
        residual = response(valid) - X * coefficient;
        training_count(proxy, branch) = sum(valid);
        rvec(proxy, find(valid)) = residual;
    end
    rvec_filled = regem(rvec);
    if branch == 1
        full_rvec_filled_max_abs = max(abs(rvec_filled - double(R.R0v_filled)), [], 'all', 'omitnan');
    end
    branch_R = nancov(rvec_filled');
    residual_variance(:, branch) = diag(branch_R);
end
assert(full_rvec_filled_max_abs <= 1e-12, 'full-regem-rvec-replay-closure-failed');

refit_full_coefficient_max_abs = max(abs(squeeze(b(:, 1, :)) - archived_b), [], 'all', 'omitnan');
refit_full_hxb_max_abs = max(abs(squeeze(hxb(:, 1, :)) - archived_hxb), [], 'all', 'omitnan');
archived_error_variance = var(double(R.R0v_filled), 0, 2, 'omitnan');
refit_full_R_max_abs = max(abs(residual_variance(:, 1) - archived_error_variance), [], 'omitnan');
refit_full_R_median_abs = median(abs(residual_variance(:, 1) - archived_error_variance), 'omitnan');
assert(refit_full_coefficient_max_abs <= 1e-9, 'refit-full-coefficient-closure-failed');
assert(refit_full_hxb_max_abs <= 1e-9, 'refit-full-hxb-before-overlay-closure-failed');

b(:, 1, :) = reshape(archived_b, nproxy, 1, 3);
hxb(:, 1, :) = reshape(archived_hxb, nproxy, 1, 848);
residual_variance(:, 1) = archived_error_variance;
baseline_coefficient_max_abs = max(abs(squeeze(b(:, 1, :)) - archived_b), [], 'all', 'omitnan');
baseline_hxb_max_abs = max(abs(squeeze(hxb(:, 1, :)) - archived_hxb), [], 'all', 'omitnan');
hxb500 = hxb(:, :, sample_indices);
full_hxb500_replay_max_abs = max(abs(squeeze(hxb500(:, 1, :)) - double(R.HXb0)), [], 'all', 'omitnan');
full_R_max_abs = max(abs(residual_variance(:, 1) - archived_error_variance), [], 'omitnan');
full_R_median_abs = median(abs(residual_variance(:, 1) - archived_error_variance), 'omitnan');

assert(baseline_coefficient_max_abs <= 1e-12, 'archived-full-coefficient-closure-failed');
assert(baseline_hxb_max_abs <= 1e-12, 'archived-full-hxb848-closure-failed');
assert(full_hxb500_replay_max_abs <= 1e-9, 'archived-full-hxb500-replay-closure-failed');
assert(full_R_max_abs <= 1e-12, 'archived-full-R-replay-closure-failed');

positive_R = residual_variance(isfinite(residual_variance) & residual_variance > 0);
minimum_positive_R = min(positive_R);
full_R_finite_count = sum(isfinite(residual_variance(:, 1)));
full_R_nonfinite_count = sum(~isfinite(residual_variance(:, 1)));
eligible = training_count >= 20;
nonfinite_R_for_eligible_count = sum(~isfinite(residual_variance(eligible)));
nonpositive_R_for_eligible_count = sum(residual_variance(eligible) <= 0);
assert(full_R_nonfinite_count == 0, 'full-R-has-nonfinite-values');
assert(nonfinite_R_for_eligible_count == 0, 'eligible-R-has-nonfinite-values');
assert(nonpositive_R_for_eligible_count == 0, 'eligible-R-has-nonpositive-values');

locked_1921_1939_accessed = false;
matlab_release = version('-release');
save(output_path, 'years', 'blocks', 'branch_outer', 'branch_inner', 'model_type', ...
    'hxb500', 'b', 'residual_variance', 'training_count', 'sample_indices', ...
    'experiment1_index_closure_exact', 'full_hxb500_replay_max_abs', ...
    'archived_error_variance', 'full_R_max_abs', 'full_R_median_abs', ...
    'full_R_finite_count', 'full_R_nonfinite_count', 'minimum_positive_R', ...
    'nonfinite_R_for_eligible_count', 'nonpositive_R_for_eligible_count', ...
    'baseline_coefficient_max_abs', 'baseline_hxb_max_abs', ...
    'refit_full_coefficient_max_abs', 'refit_full_hxb_max_abs', ...
    'refit_full_R_max_abs', 'refit_full_R_median_abs', ...
    'source_hxb_closure_max_abs', 'full_rvec_filled_max_abs', ...
    'locked_1921_1939_accessed', 'matlab_release', '-v7.3');
fprintf(['FOLD_NATIVE_PSM_20YEAR_V1_COMPLETE proxies=%d members=500 ' ...
    'hxb500_closure=%.17g R_closure=%.17g min_R=%.17g\n'], ...
    nproxy, full_hxb500_replay_max_abs, full_R_max_abs, minimum_positive_R);
end

function [annual, lat, lon] = read_bearth_allowed(path)
info = ncinfo(path, 'temperature');
start_month = (1940 - 1850) * 12 + 1;
count_month = (2000 - 1940 + 1) * 12;
raw = ncread(path, 'temperature', [1 1 start_month], [info.Size(1) info.Size(2) count_month]);
raw = permute(raw, [2 1 3]);
lon = double(ncread(path, 'longitude'));
lat = double(ncread(path, 'latitude'));
[raw, lon] = regrid2gcm(raw, lon);
annual3 = reshape(mean(reshape(raw, size(raw, 1), size(raw, 2), 12, 61), 3, 'omitnan'), size(raw, 1), size(raw, 2), 61);
annual3(annual3 == 0) = NaN;
annual = reshape(annual3, size(annual3, 1) * size(annual3, 2), 61);
lat = lat(:);
lon = lon(:);
end

function [annual, lat, lon] = read_gpcc_allowed(path)
info = ncinfo(path, 'precip');
start_month = (1940 - 1891) * 12 + 1;
count_month = (2000 - 1940 + 1) * 12;
raw = ncread(path, 'precip', [1 1 start_month], [info.Size(1) info.Size(2) count_month]);
raw = permute(raw, [2 1 3]);
raw = flip(raw, 1);
lon = double(ncread(path, 'lon'));
lat = flip(double(ncread(path, 'lat')));
[raw, lon] = regrid2gcm(raw, lon);
annual3 = reshape(mean(reshape(raw, size(raw, 1), size(raw, 2), 12, 61), 3, 'omitnan'), size(raw, 1), size(raw, 2), 61);
annual3(annual3 == 0) = NaN;
annual = reshape(annual3, size(annual3, 1) * size(annual3, 2), 61) / 30;
reference = (1961:1990) - 1940 + 1;
annual = annual - mean(annual(:, reference), 2, 'omitnan');
lat = lat(:);
lon = lon(:);
end
