function replay_enkf_fixed_experiment_matlab_v1(input_path, output_path, mode, prior_cache_path)
% Exact MATLAB replay of proxy6 / EnKF / experiment 1.

if nargin < 3 || isempty(mode)
    mode = 'preflight';
end
if nargin < 4
    prior_cache_path = '';
end

source_root = 'D:/Users/JUNZ/MATLAB/Projects/untitled/PHYDA-v1-master';
addpath(source_root);
addpath(fullfile(source_root, 'input-recon', 'RegEM-master'));

S = load(input_path);
if strcmpi(mode, 'preflight')
    columns = round(S.preflight_columns(:)');
elseif strcmpi(mode, 'full')
    columns = 1:1001;
else
    error('Unknown replay mode: %s', mode);
end

if ~isempty(prior_cache_path) && exist(prior_cache_path, 'file')
    P = load(prior_cache_path, 'Xb_o_full');
    Xb_o_full = P.Xb_o_full;
else
    Xb_o_full = build_merged_cru_prior(source_root);
end
if ~isequal(size(Xb_o_full), [336, 848])
    error('Unexpected prior shape: %s', mat2str(size(Xb_o_full)));
end

sample_indices = round(S.sample_indices(:));
Xb = Xb_o_full(:, sample_indices);
nrun = length(columns);
Xa_m = zeros(336, nrun);
Xa_sigma = zeros(336, nrun);
Xa_prct = zeros(336, 3, nrun);
active_proxy_count = zeros(1, nrun);

run_timer = tic;
for jj = 1:nrun
    t = columns(jj);
    ni = find(~isnan(S.y0(:, t)));
    active_proxy_count(jj) = length(ni);
    y = S.y0(ni, t);
    HXb = S.HXb0(ni, :);
    residual_subset = S.R0v_filled(ni, :);
    if exist('nancov', 'file')
        R = diag(diag(nancov(residual_subset')));
    else
        R = diag(var(residual_subset, 0, 2));
    end
    [Xa_current, Xa_mean] = M_update(Xb, HXb, y, R, 1);
    Xa_m(:, jj) = Xa_mean;
    Xa_sigma(:, jj) = std(Xa_current, 0, 2);
    Xa_prct(:, :, jj) = prctile(Xa_current, [5, 50, 95], 2);
    if mod(jj, 100) == 0 || jj == nrun
        fprintf('Replay progress: %d/%d\n', jj, nrun);
    end
end
runtime_seconds = toc(run_timer);

target_mean = S.target_mean(:, columns);
target_sigma = S.target_sigma(:, columns);
target_prct = S.target_prct(:, :, columns);
mean_abs_diff = abs(Xa_m - target_mean);
sigma_abs_diff = abs(Xa_sigma - target_sigma);
prct_abs_diff = abs(Xa_prct - target_prct);
mean_max_abs_diff_by_year = max(mean_abs_diff, [], 1);
sigma_max_abs_diff_by_year = max(sigma_abs_diff, [], 1);
prct_max_abs_diff_by_year = squeeze(max(max(prct_abs_diff, [], 1), [], 2))';
mean_global_max_abs_diff = max(mean_abs_diff(:));
sigma_global_max_abs_diff = max(sigma_abs_diff(:));
prct_global_max_abs_diff = max(prct_abs_diff(:));
run_years = columns + 999;

save(output_path, 'mode', 'columns', 'run_years', 'active_proxy_count', ...
    'Xb_o_full', 'Xa_m', 'Xa_sigma', 'Xa_prct', ...
    'mean_max_abs_diff_by_year', 'sigma_max_abs_diff_by_year', ...
    'prct_max_abs_diff_by_year', 'mean_global_max_abs_diff', ...
    'sigma_global_max_abs_diff', 'prct_global_max_abs_diff', ...
    'runtime_seconds', '-v7');

fprintf('MEAN_GLOBAL_MAX_ABS_DIFF=%.17g\n', mean_global_max_abs_diff);
fprintf('SIGMA_GLOBAL_MAX_ABS_DIFF=%.17g\n', sigma_global_max_abs_diff);
fprintf('PRCT_GLOBAL_MAX_ABS_DIFF=%.17g\n', prct_global_max_abs_diff);
fprintf('RUNTIME_SECONDS=%.3f\n', runtime_seconds);
end


function Xb_o_full = build_merged_cru_prior(source_root)
coord_file = fullfile(source_root, 'input-recon', ...
    'b.e11.BLMTRC5CN.f19_g16.010.cam.h0.TREFHT.085001-184912.nc');
data_file = 'D:/VM/gcm_7/cesm/cesm_TREFHT_merged.nc';
hma_lat_range = [24, 51];
hma_lon_range = [62, 110];

lat_orig = ncread(coord_file, 'lat');
lon_orig_raw = ncread(coord_file, 'lon');
lon_orig = lon_orig_raw;
lon_orig(lon_orig > 180) = lon_orig(lon_orig > 180) - 360;
[lon_orig, lon_sort_idx] = sort(lon_orig);

new_lon = linspace(-179, 179, 180);
new_lat = linspace(-89, 89, 90);
lat_hma_idx = find(lat_orig >= hma_lat_range(1) - 1 & ...
    lat_orig <= hma_lat_range(2) + 2);
lon_hma_idx = find(lon_orig >= hma_lon_range(1) - 1 & ...
    lon_orig <= hma_lon_range(2) + 1);
lat_hma = lat_orig(lat_hma_idx);
lon_hma = lon_orig(lon_hma_idx);
ob_lat_hma_idx = find(new_lat >= hma_lat_range(1) & new_lat <= hma_lat_range(2));
ob_lon_hma_idx = find(new_lon >= hma_lon_range(1) & new_lon <= hma_lon_range(2));
ob_lat_hma = new_lat(ob_lat_hma_idx);
ob_lon_hma = new_lon(ob_lon_hma_idx);
[LON_hma, LAT_hma] = meshgrid(ob_lon_hma, ob_lat_hma);

source_lon_ids = lon_sort_idx(lon_hma_idx);
if any(diff(source_lon_ids) ~= 1) || any(diff(lat_hma_idx) ~= 1)
    error('Expected contiguous HMA source indices.');
end
Xv_mon = ncread(data_file, 'TREFHT', ...
    [source_lon_ids(1), lat_hma_idx(1), 1], ...
    [length(source_lon_ids), length(lat_hma_idx), Inf]);
Xv_hma = permute(Xv_mon, [2, 1, 3]);
[~, ~, nt] = size(Xv_hma);
Xv_hma_hr = nan(length(ob_lat_hma), length(ob_lon_hma), nt);
for tt = 1:nt
    Xv_hma_hr(:, :, tt) = interp2(lon_hma, lat_hma, ...
        Xv_hma(:, :, tt), LON_hma, LAT_hma, 'linear');
end
clear Xv_mon Xv_hma

old_dir = pwd;
cleanup = onCleanup(@() cd(old_dir));
cd(source_root);
X_clim1 = load_obsclim('CRUCL11_tas', new_lat, new_lon);
X_clim = X_clim1(ob_lat_hma_idx, ob_lon_hma_idx - 90, :);

[nlat, nlon, nmonth] = size(Xv_hma_hr);
Xv_mon12 = reshape(Xv_hma_hr, nlat, nlon, 12, nmonth / 12);
Xv_ns = bsxfun(@minus, Xv_mon12, mean(Xv_mon12, 4));
Xv_bc = bsxfun(@plus, Xv_ns, X_clim);
Xv_sub_bc = reshape(Xv_bc, nlat, nlon, nmonth);
Xv_ann = mon2ann(Xv_sub_bc, 1, 12);
Xb_o_full = reshape(Xv_ann(:, :, 152:999), 336, 848);
clear cleanup
end
