function replay_precipitation_pakistan_weight_sensitivity_v1(replay_path, prior_path, group_path, baseline_path, output_path)
% Frozen precipitation-EnKF sensitivity to Pakistan proxy group information weight.
S = load(replay_path);
P = load(prior_path, 'precipitation_prior_336x848_mm_day', 'lat', 'lon');
G = load(group_path);
B = load(baseline_path, 'run_years', 'total_increment');
group = round(G.pakistan_proxy_indices(:));
design_effect = G.design_effect(1);
assert(length(group) == 14 && design_effect > 1 && design_effect < 14, 'group-weight-contract-failed');

Xall = P.precipitation_prior_336x848_mm_day;
samples = round(S.sample_indices(:));
Xb = Xall(:, samples);
Xmean = mean(Xb, 2);
Xp = Xb - Xmean;
nens = size(Xb, 2);
latw = repmat(cosd(P.lat(:)), 1, length(P.lon));
weights = latw(:) ./ sum(latw(:));
prior_hma_mean = weights' * Xmean;

years = 1901:1920;
assert(isequal(round(B.run_years(:))', years), 'year-contract-failed');
names = {'baseline','pakistan_ablation','pakistan_design_effect_R'};
total = zeros(length(years), length(names));
posterior_mean = zeros(size(Xb, 1), length(years), length(names));
update_closure = zeros(length(years), length(names));
available_group_count = zeros(length(years), 1);
zero_gain_ablation_equivalence = zeros(length(years), 1);

for yy = 1:length(years)
    t = years(yy) - 999;
    ni = find(~isnan(S.y0(:, t)));
    available_group_count(yy) = sum(ismember(ni, group));
    assert(available_group_count(yy) == 14, 'incomplete-pakistan-target-period');
    y = S.y0(ni, t);
    HX = S.HXb0(ni, :);
    residual_subset = S.R0v_filled(ni, :);
    Rdiag = diag(nancov(residual_subset'));

    [posterior_mean(:,yy,1), update_closure(yy,1)] = update_branch(Xmean, Xp, HX, y, Rdiag, weights, prior_hma_mean, nens);

    keep = ~ismember(ni, group);
    [posterior_mean(:,yy,2), update_closure(yy,2)] = update_branch(Xmean, Xp, HX(keep,:), y(keep), Rdiag(keep), weights, prior_hma_mean, nens);

    weighted_Rdiag = Rdiag;
    weighted_Rdiag(ismember(ni, group)) = weighted_Rdiag(ismember(ni, group)) .* design_effect;
    [posterior_mean(:,yy,3), update_closure(yy,3)] = update_branch(Xmean, Xp, HX, y, weighted_Rdiag, weights, prior_hma_mean, nens);

    Hzero = HX;
    group_rows = ismember(ni, group);
    Hzero(group_rows,:) = repmat(mean(Hzero(group_rows,:), 2), 1, nens);
    [zero_posterior, ~] = update_branch(Xmean, Xp, Hzero, y, Rdiag, weights, prior_hma_mean, nens);
    zero_gain_ablation_equivalence(yy) = max(abs(zero_posterior - posterior_mean(:,yy,2)));
end

for bb = 1:length(names)
    total(:,bb) = squeeze(weights' * posterior_mean(:,:,bb))' - prior_hma_mean;
end
baseline_closure = max(abs(total(:,1) - B.total_increment(:)));
max_update_closure = max(abs(update_closure), [], 'all');
max_zero_gain_ablation_equivalence = max(zero_gain_ablation_equivalence);
assert(baseline_closure <= 1e-12, 'baseline-closure-failed');
assert(max_update_closure <= 1e-12, 'update-closure-failed');
assert(max_zero_gain_ablation_equivalence <= 1e-12, 'ablation-equivalence-failed');
save(output_path, 'years', 'names', 'group', 'design_effect', 'prior_hma_mean', 'total', 'posterior_mean', 'weights', 'available_group_count', 'update_closure', 'baseline_closure', 'zero_gain_ablation_equivalence', 'max_zero_gain_ablation_equivalence', '-v7.3');
fprintf('BASELINE_CLOSURE=%.17g\n', baseline_closure);
fprintf('MAX_UPDATE_CLOSURE=%.17g\n', max_update_closure);
fprintf('ABLATION_EQUIVALENCE=%.17g\n', max_zero_gain_ablation_equivalence);
disp('PAKISTAN_WEIGHT_SENSITIVITY_COMPLETE');
end

function [posterior, closure] = update_branch(Xmean, Xp, HX, y, Rdiag, weights, prior_hma_mean, nens)
HXmean = mean(HX, 2);
HXp = HX - HXmean;
R = diag(Rdiag);
K = ((Xp * HXp') ./ (nens - 1)) / (((HXp * HXp') ./ (nens - 1)) + R);
innovation = y - HXmean;
posterior = Xmean + K * innovation;
hma_increment = weights' * posterior - prior_hma_mean;
closure = hma_increment - sum((weights' * K)'.* innovation);
end
