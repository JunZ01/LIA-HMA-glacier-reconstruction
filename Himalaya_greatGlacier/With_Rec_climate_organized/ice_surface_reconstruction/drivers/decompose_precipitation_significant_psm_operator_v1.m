function decompose_precipitation_significant_psm_operator_v1(replay_input_path, prior_path, component_path, baseline_path, output_path)
% Frozen-background EnKF decomposition of significant bivariate PSM anomalies.

source_root = 'D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master';
addpath(source_root);
S = load(replay_input_path);
P = load(prior_path, 'precipitation_prior_336x848_mm_day', 'lat', 'lon');
C = load(component_path);
B = load(baseline_path, 'run_years', 'total_increment');
Xb_full = P.precipitation_prior_336x848_mm_day;
assert(isequal(size(Xb_full), [336, 848]), 'unexpected-prior-shape');
sample_indices = round(S.sample_indices(:));
assert(length(sample_indices) == 500 && length(unique(sample_indices)) == 500, 'invalid-sample-indices');
Xb = Xb_full(:, sample_indices);
Xb_m = mean(Xb, 2);
Xb_p = Xb - Xb_m;
n_ens = size(Xb, 2);

selected = round(C.selected_proxy_indices(:));
assert(length(selected) == 65 && length(unique(selected)) == 65, 'unexpected-selected-count');
assert(isequal(size(C.temperature_anomaly), [65, n_ens]), 'invalid-temperature-anomaly-shape');
assert(isequal(size(C.precipitation_anomaly), [65, n_ens]), 'invalid-precipitation-anomaly-shape');
assert(isequal(size(C.full_anomaly), [65, n_ens]), 'invalid-full-anomaly-shape');

H_base = S.HXb0;
selected_mean = mean(H_base(selected, :), 2);
H_combined = H_base;
H_temperature = H_base;
H_precipitation = H_base;
H_zero = H_base;
H_combined(selected, :) = selected_mean + C.full_anomaly;
H_temperature(selected, :) = selected_mean + C.temperature_anomaly;
H_precipitation(selected, :) = selected_mean + C.precipitation_anomaly;
H_zero(selected, :) = selected_mean;
combined_closure = max(abs(H_combined - H_base), [], 'all');
assert(combined_closure <= 1e-12, 'combined-operator-closure-failed');
mean_invariance = [max(abs(mean(H_base, 2) - mean(H_combined, 2))), max(abs(mean(H_base, 2) - mean(H_temperature, 2))), max(abs(mean(H_base, 2) - mean(H_precipitation, 2))), max(abs(mean(H_base, 2) - mean(H_zero, 2)))];
assert(max(mean_invariance) <= 1e-12, 'operator-mean-invariance-failed');

weight_grid = repmat(cosd(P.lat(:)), 1, length(P.lon));
state_weights = weight_grid(:) ./ sum(weight_grid(:));
prior_hma_mean = state_weights' * Xb_m;
run_years = 1901:1920;
assert(isequal(round(B.run_years(:))', run_years), 'baseline-year-mismatch');
branch_names = {'baseline', 'combined_rebuild', 'temperature_only', 'precipitation_only', 'zero_selected_anomaly'};
branch_hxb = {H_base, H_combined, H_temperature, H_precipitation, H_zero};
nbranch = length(branch_names);
all_proxy_count = size(S.y0, 1);
nrun = length(run_years);
branch_total_increment = zeros(nrun, nbranch);
branch_selected_contribution = zeros(nrun, nbranch);
branch_nonselected_contribution = zeros(nrun, nbranch);
branch_closure_error = zeros(nrun, nbranch);
branch_gain_weight = nan(all_proxy_count, nrun, nbranch);
branch_innovation = nan(all_proxy_count, nrun, nbranch);
branch_contribution = zeros(all_proxy_count, nrun, nbranch);

for bb = 1:nbranch
    H_all = branch_hxb{bb};
    for yy = 1:nrun
        t = run_years(yy) - 999;
        ni = find(~isnan(S.y0(:, t)));
        y = S.y0(ni, t);
        HXb = H_all(ni, :);
        HXb_m = mean(HXb, 2);
        HXb_p = HXb - HXb_m;
        residual_subset = S.R0v_filled(ni, :);
        if exist('nancov', 'file')
            R = diag(diag(nancov(residual_subset')));
        else
            R = diag(var(residual_subset, 0, 2));
        end
        K = ((Xb_p * HXb_p') ./ (n_ens - 1)) / (((HXb_p * HXb_p') ./ (n_ens - 1)) + R);
        current_innovation = y - HXb_m;
        current_gain_weight = state_weights' * K;
        current_contribution = current_gain_weight(:) .* current_innovation(:);
        branch_total_increment(yy, bb) = state_weights' * (Xb_m + K * current_innovation) - prior_hma_mean;
        branch_closure_error(yy, bb) = sum(current_contribution) - branch_total_increment(yy, bb);
        selected_local = ismember(ni, selected);
        branch_selected_contribution(yy, bb) = sum(current_contribution(selected_local));
        branch_nonselected_contribution(yy, bb) = sum(current_contribution(~selected_local));
        branch_gain_weight(ni, yy, bb) = current_gain_weight(:);
        branch_innovation(ni, yy, bb) = current_innovation(:);
        branch_contribution(ni, yy, bb) = current_contribution(:);
    end
end

assert(max(abs(branch_closure_error), [], 'all') <= 1e-12, 'branch-increment-closure-failed');
baseline_closure = max(abs(branch_total_increment(:, 1) - B.total_increment(:)));
assert(baseline_closure <= 1e-12, 'archived-baseline-closure-failed');
innovation_invariance = max(abs(branch_innovation - branch_innovation(:, :, 1)), [], 'all');
assert(innovation_invariance <= 1e-12, 'innovation-invariance-failed');
save(output_path, 'run_years', 'branch_names', 'selected', 'prior_hma_mean', 'branch_total_increment', 'branch_selected_contribution', 'branch_nonselected_contribution', 'branch_closure_error', 'branch_gain_weight', 'branch_innovation', 'branch_contribution', 'combined_closure', 'mean_invariance', 'baseline_closure', 'innovation_invariance', 'state_weights', '-v7.3');
fprintf('COMBINED_CLOSURE=%.17g\n', combined_closure);
fprintf('BASELINE_CLOSURE=%.17g\n', baseline_closure);
fprintf('INNOVATION_INVARIANCE=%.17g\n', innovation_invariance);
disp('SIGNIFICANT_PSM_OPERATOR_DECOMPOSITION_COMPLETE');
end
