function replay_precipitation_enkf_exp1(input_path, prior_path, output_path)
% Replay proxy6 / precipitation / EnKF / experiment 1 for all 1001 years.

source_root = 'D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master';
addpath(source_root);

S = load(input_path);
P = load(prior_path, 'precipitation_prior_336x848_mm_day');
Xb_o_full = P.precipitation_prior_336x848_mm_day;
assert(isequal(size(Xb_o_full), [336, 848]), 'unexpected-prior-shape');
assert(all(isfinite(Xb_o_full), 'all'), 'nonfinite-prior');

sample_indices = round(S.sample_indices(:));
assert(length(sample_indices) == 500, 'unexpected-sample-count');
assert(length(unique(sample_indices)) == 500, 'duplicate-sample-index');
Xb = Xb_o_full(:, sample_indices);

nrun = 1001;
Xa_m = zeros(336, nrun);
Xa_sigma = zeros(336, nrun);
Xa_prct = zeros(336, 3, nrun);
active_proxy_count = zeros(1, nrun);

run_timer = tic;
for t = 1:nrun
    ni = find(~isnan(S.y0(:, t)));
    active_proxy_count(t) = length(ni);
    y = S.y0(ni, t);
    HXb = S.HXb0(ni, :);
    residual_subset = S.R0v_filled(ni, :);
    if exist('nancov', 'file')
        R = diag(diag(nancov(residual_subset')));
    else
        R = diag(var(residual_subset, 0, 2));
    end
    [Xa_current, Xa_mean] = M_update(Xb, HXb, y, R, 1);
    Xa_m(:, t) = Xa_mean;
    Xa_sigma(:, t) = std(Xa_current, 0, 2);
    Xa_prct(:, :, t) = prctile(Xa_current, [5, 50, 95], 2);
    if mod(t, 100) == 0 || t == nrun
        fprintf('Replay progress: %d/%d\n', t, nrun);
    end
end
runtime_seconds = toc(run_timer);

mean_abs_diff = abs(Xa_m - S.target_mean);
sigma_abs_diff = abs(Xa_sigma - S.target_sigma);
prct_abs_diff = abs(Xa_prct - S.target_prct);
mean_max_abs_diff_by_year = max(mean_abs_diff, [], 1);
sigma_max_abs_diff_by_year = max(sigma_abs_diff, [], 1);
prct_max_abs_diff_by_year = squeeze(max(max(prct_abs_diff, [], 1), [], 2))';
mean_global_max_abs_diff = max(mean_abs_diff, [], 'all');
sigma_global_max_abs_diff = max(sigma_abs_diff, [], 'all');
prct_global_max_abs_diff = max(prct_abs_diff, [], 'all');
mean_global_rmse = sqrt(mean(mean_abs_diff.^2, 'all'));
sigma_global_rmse = sqrt(mean(sigma_abs_diff.^2, 'all'));
prct_global_rmse = sqrt(mean(prct_abs_diff.^2, 'all'));
run_years = 1000:2000;

save(output_path, 'run_years', 'active_proxy_count', ...
    'Xa_m', 'Xa_sigma', 'Xa_prct', ...
    'mean_max_abs_diff_by_year', 'sigma_max_abs_diff_by_year', ...
    'prct_max_abs_diff_by_year', 'mean_global_max_abs_diff', ...
    'sigma_global_max_abs_diff', 'prct_global_max_abs_diff', ...
    'mean_global_rmse', 'sigma_global_rmse', 'prct_global_rmse', ...
    'runtime_seconds', '-v7.3');

fprintf('MEAN_GLOBAL_MAX_ABS_DIFF=%.17g\n', mean_global_max_abs_diff);
fprintf('SIGMA_GLOBAL_MAX_ABS_DIFF=%.17g\n', sigma_global_max_abs_diff);
fprintf('PRCT_GLOBAL_MAX_ABS_DIFF=%.17g\n', prct_global_max_abs_diff);
fprintf('RUNTIME_SECONDS=%.3f\n', runtime_seconds);
disp('PRECIPITATION_ENKF_EXP1_REPLAY_COMPLETE');
end

