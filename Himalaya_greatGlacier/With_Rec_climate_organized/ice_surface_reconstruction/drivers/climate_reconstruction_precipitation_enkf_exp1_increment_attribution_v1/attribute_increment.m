function attribute_increment(replay_input_path, prior_path, output_path)
% Attribute proxy6 EnKF experiment-1 HMA-mean increments for 1901-1920.

S = load(replay_input_path);
P = load(prior_path, 'precipitation_prior_336x848_mm_day', 'lat', 'lon');
Xb_o_full = P.precipitation_prior_336x848_mm_day;
assert(isequal(size(Xb_o_full), [336, 848]), 'unexpected-prior-shape');
assert(length(P.lat) == 14 && length(P.lon) == 24, 'unexpected-prior-grid');

sample_indices = round(S.sample_indices(:));
Xb = Xb_o_full(:, sample_indices);
Xb_m = mean(Xb, 2);
Xb_p = Xb - Xb_m;
n_ens = size(Xb, 2);

latitude_weights = cosd(P.lat(:));
weight_grid = repmat(latitude_weights, 1, length(P.lon));
state_weights = weight_grid(:);
state_weights = state_weights ./ sum(state_weights);
prior_hma_mean = state_weights' * Xb_m;

run_years = 1901:1920;
nrun = length(run_years);
all_proxy_count = size(S.y0, 1);
posterior_hma_mean = zeros(nrun, 1);
total_increment = zeros(nrun, 1);
contribution_sum = zeros(nrun, 1);
closure_error = zeros(nrun, 1);
active_proxy_count = zeros(nrun, 1);
gain_weight = nan(all_proxy_count, nrun);
innovation = nan(all_proxy_count, nrun);
contribution = zeros(all_proxy_count, nrun);

for yy = 1:nrun
    year = run_years(yy);
    t = year - 999;
    ni = find(~isnan(S.y0(:, t)));
    active_proxy_count(yy) = length(ni);
    y = S.y0(ni, t);
    HXb = S.HXb0(ni, :);
    HXb_m = mean(HXb, 2);
    HXb_p = HXb - HXb_m;
    residual_subset = S.R0v_filled(ni, :);
    if exist('nancov', 'file')
        R = diag(diag(nancov(residual_subset')));
    else
        R = diag(var(residual_subset, 0, 2));
    end
    PbHT = (Xb_p * HXb_p') ./ (n_ens - 1);
    HPbHTR = (HXb_p * HXb_p') ./ (n_ens - 1) + R;
    K = PbHT / HPbHTR;
    current_innovation = y - HXb_m;
    current_gain_weight = state_weights' * K;
    current_contribution = current_gain_weight(:) .* current_innovation(:);
    Xa_m = Xb_m + K * current_innovation;

    posterior_hma_mean(yy) = state_weights' * Xa_m;
    total_increment(yy) = posterior_hma_mean(yy) - prior_hma_mean;
    contribution_sum(yy) = sum(current_contribution);
    closure_error(yy) = contribution_sum(yy) - total_increment(yy);
    gain_weight(ni, yy) = current_gain_weight(:);
    innovation(ni, yy) = current_innovation(:);
    contribution(ni, yy) = current_contribution(:);
end

assert(max(abs(closure_error)) <= 1e-12, 'increment-attribution-closure-failed');
save(output_path, 'run_years', 'prior_hma_mean', 'posterior_hma_mean', ...
    'total_increment', 'contribution_sum', 'closure_error', ...
    'active_proxy_count', 'gain_weight', 'innovation', 'contribution', ...
    'state_weights', '-v7.3');
fprintf('MAX_CLOSURE_ERROR=%.17g\n', max(abs(closure_error)));
disp('PRECIPITATION_INCREMENT_ATTRIBUTION_COMPLETE');
end
