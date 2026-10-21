function replay_precipitation_positive_b2_source_block_ablation_v1(replay_path, prior_path, group_path, baseline_path, prior_full_ablation_path, output_path)
% Frozen precipitation-EnKF ablation of three mutually exclusive positive-b2 blocks.
S = load(replay_path);
P = load(prior_path, 'precipitation_prior_336x848_mm_day', 'lat', 'lon');
G = load(group_path);
B = load(baseline_path, 'run_years', 'total_increment');
F = load(prior_full_ablation_path, 'years', 'names', 'total');

pakistan = round(G.pakistan_proxy_indices(:));
shao = round(G.shao_proxy_indices(:));
other = round(G.other_positive_b2_proxy_indices(:));
positive = round(G.positive_b2_proxy_indices(:));
assert(length(pakistan) == 14 && length(shao) == 13 && length(other) == 33 && length(positive) == 60, 'group-count-contract-failed');
assert(isempty(intersect(pakistan, shao)) && isempty(intersect(pakistan, other)) && isempty(intersect(shao, other)), 'group-overlap-contract-failed');
assert(isequal(sort([pakistan; shao; other]), sort(positive)), 'group-union-contract-failed');

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
names = {'baseline','pakistan_ablation','shao_ablation','other_positive_b2_ablation','all_positive_b2_ablation'};
groups = {[], pakistan, shao, other, positive};
total = zeros(length(years), length(names));
posterior_mean = zeros(size(Xb, 1), length(years), length(names));
update_closure = zeros(length(years), length(names));
available_group_count = zeros(length(years), length(names));
zero_gain_ablation_equivalence = zeros(length(years), length(names));

for yy = 1:length(years)
    t = years(yy) - 999;
    ni = find(~isnan(S.y0(:, t)));
    y = S.y0(ni, t);
    HX = S.HXb0(ni, :);
    Rdiag = diag(nancov(S.R0v_filled(ni, :)'));
    [posterior_mean(:,yy,1), update_closure(yy,1)] = update_branch(Xmean, Xp, HX, y, Rdiag, weights, prior_hma_mean, nens);

    for bb = 2:length(names)
        group_rows = ismember(ni, groups{bb});
        available_group_count(yy,bb) = sum(group_rows);
        keep = ~group_rows;
        [posterior_mean(:,yy,bb), update_closure(yy,bb)] = update_branch(Xmean, Xp, HX(keep,:), y(keep), Rdiag(keep), weights, prior_hma_mean, nens);

        Hzero = HX;
        Hzero(group_rows,:) = repmat(mean(Hzero(group_rows,:), 2), 1, nens);
        [zero_posterior, ~] = update_branch(Xmean, Xp, Hzero, y, Rdiag, weights, prior_hma_mean, nens);
        zero_gain_ablation_equivalence(yy,bb) = max(abs(zero_posterior - posterior_mean(:,yy,bb)));
    end
end

for bb = 1:length(names)
    total(:,bb) = squeeze(weights' * posterior_mean(:,:,bb))' - prior_hma_mean;
end
baseline_closure = max(abs(total(:,1) - B.total_increment(:)));
full_index = find(strcmp(F.names, 'positive_b2_full_zero'));
assert(length(full_index) == 1 && isequal(round(F.years(:))', years), 'prior-full-ablation-contract-failed');
prior_full_ablation_closure = max(abs(total(:,5) - F.total(:,full_index)));
max_update_closure = max(abs(update_closure), [], 'all');
max_zero_gain_ablation_equivalence = max(zero_gain_ablation_equivalence, [], 'all');
assert(baseline_closure <= 1e-12, 'baseline-closure-failed');
assert(prior_full_ablation_closure <= 2e-12, 'prior-full-ablation-closure-failed');
assert(max_update_closure <= 1e-12, 'update-closure-failed');
assert(max_zero_gain_ablation_equivalence <= 2e-12, 'ablation-equivalence-failed');

save(output_path, 'years', 'names', 'pakistan', 'shao', 'other', 'positive', 'prior_hma_mean', 'total', 'posterior_mean', 'weights', 'available_group_count', 'update_closure', 'baseline_closure', 'prior_full_ablation_closure', 'zero_gain_ablation_equivalence', 'max_zero_gain_ablation_equivalence', '-v7.3');
fprintf('BASELINE_CLOSURE=%.17g\n', baseline_closure);
fprintf('PRIOR_FULL_ABLATION_CLOSURE=%.17g\n', prior_full_ablation_closure);
fprintf('MAX_UPDATE_CLOSURE=%.17g\n', max_update_closure);
fprintf('MAX_ABLATION_EQUIVALENCE=%.17g\n', max_zero_gain_ablation_equivalence);
disp('POSITIVE_B2_SOURCE_BLOCK_ABLATION_COMPLETE');
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
