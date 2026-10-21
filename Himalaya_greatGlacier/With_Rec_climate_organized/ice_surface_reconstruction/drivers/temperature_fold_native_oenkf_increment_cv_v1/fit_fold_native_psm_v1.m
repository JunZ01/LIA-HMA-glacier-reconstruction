function fit_fold_native_psm_v1(allowed_input_path, prior_input_path, output_path, phyda_root, bearth_path, gpcc_path)
% Fit full, outer, and nested-inner PSM branches using only 1940-2000.

old_dir = pwd;
cleanup_dir = onCleanup(@() cd(old_dir));
cd(phyda_root);
addpath(phyda_root);

S = load(allowed_input_path);
P = load(prior_input_path, 'temperature_prior_336x848');
years = double(S.years(:));
assert(isequal(years, (1940:2000)'), 'unexpected-allowed-year-axis');
assert(~any(years >= 1921 & years <= 1939), 'locked-period-requested');
assert(size(P.temperature_prior_336x848, 2) == 848, 'unexpected-prior-member-count');

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

state_lat = linspace(-89, 89, 90);
state_lon = linspace(-179, 179, 180);
state_lat = state_lat(state_lat >= 24 & state_lat <= 51);
state_lon = state_lon(state_lon >= 62 & state_lon <= 110);
[~, lat_index] = min(abs(state_lat(:) - proxy_lat), [], 1);
[~, lon_index] = min(abs(state_lon(:) - proxy_lon), [], 1);
state_index = sub2ind([numel(state_lat), numel(state_lon)], lat_index, lon_index);

temperature_prior = double(P.temperature_prior_336x848);
model_t_centered = temperature_prior(state_index, :) - mean(temperature_prior(state_index, :), 2, 'omitnan');
full_t_mean = mean(proxy_t, 2, 'omitnan');
full_p_mean = mean(proxy_p, 2, 'omitnan');
model_t_full = model_t_centered + full_t_mean;

archived_b = double(S.archived_b);
archived_hxb = double(S.archived_hxb);
model_type = double(S.model_type(:));
nproxy = size(S.observations, 2);
assert(size(archived_hxb, 1) == nproxy && size(archived_hxb, 2) == 848, 'archived-hxb-shape-failed');

model_p_centered = nan(nproxy, 848);
bivariate = model_type == 2;
assert(all(abs(archived_b(bivariate, 3)) > 1e-12), 'near-zero-archived-precipitation-coefficient');
model_p_full = (archived_hxb(bivariate, :) - archived_b(bivariate, 1) ...
    - archived_b(bivariate, 2) .* model_t_full(bivariate, :)) ./ archived_b(bivariate, 3);
model_p_centered(bivariate, :) = model_p_full - full_p_mean(bivariate);

blocks = [1940 1951; 1952 1963; 1964 1975; 1976 1987; 1988 2000];
branch_outer = zeros(26, 1);
branch_inner = zeros(26, 1);
cursor = 1;
for outer = 1:5
    cursor = cursor + 1;
    branch_outer(cursor) = outer;
end
for outer = 1:5
    for inner = 1:5
        if inner == outer
            continue;
        end
        cursor = cursor + 1;
        branch_outer(cursor) = outer;
        branch_inner(cursor) = inner;
    end
end
assert(cursor == 26, 'branch-count-failed');

hxb = nan(nproxy, 26, 848);
b = nan(nproxy, 26, 3);
residual_variance = nan(nproxy, 26);
training_count = zeros(nproxy, 26);
observations = double(S.observations);

for branch = 1:26
    train = true(numel(years), 1);
    outer = branch_outer(branch);
    inner = branch_inner(branch);
    if outer > 0
        train = train & ~(years >= blocks(outer, 1) & years <= blocks(outer, 2));
    end
    if inner > 0
        train = train & ~(years >= blocks(inner, 1) & years <= blocks(inner, 2));
    end
    branch_t_mean = mean(proxy_t(:, train), 2, 'omitnan');
    branch_p_mean = mean(proxy_p(:, train), 2, 'omitnan');
    model_t = model_t_centered + branch_t_mean;
    model_p = model_p_centered + branch_p_mean;

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
        residual_variance(proxy, branch) = var(residual, 0, 'omitnan');
    end
end

coefficient_delta = squeeze(b(:, 1, :)) - archived_b;
baseline_coefficient_max_abs = max(abs(coefficient_delta), [], 'all', 'omitnan');
baseline_hxb_max_abs = max(abs(squeeze(hxb(:, 1, :)) - archived_hxb), [], 'all', 'omitnan');
assert(baseline_coefficient_max_abs <= 1e-9, 'allowed-full-coefficient-closure-failed');
assert(baseline_hxb_max_abs <= 1e-9, 'allowed-full-hxb-closure-failed');
assert(all(isfinite(residual_variance(training_count >= 20))), 'finite-residual-variance-failed');

locked_1921_1939_accessed = false;
matlab_release = version('-release');
save(output_path, 'years', 'blocks', 'branch_outer', 'branch_inner', 'model_type', ...
    'hxb', 'b', 'residual_variance', 'training_count', ...
    'baseline_coefficient_max_abs', 'baseline_hxb_max_abs', ...
    'locked_1921_1939_accessed', 'matlab_release', '-v7.3');
fprintf('FOLD_NATIVE_PSM_COMPLETE proxies=%d coefficient_closure=%.17g hxb_closure=%.17g\n', ...
    nproxy, baseline_coefficient_max_abs, baseline_hxb_max_abs);
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
