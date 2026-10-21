function run_precipitation_oof_admissibility_operator_sensitivity_v1(replay_path, prior_path, group_path, baseline_path, output_path)
% Frozen proxy-operator anomaly sensitivity for OOF PSM admissibility.

addpath('D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master');
S = load(replay_path);
P = load(prior_path, 'precipitation_prior_336x848_mm_day', 'lat', 'lon');
G = load(group_path);
B = load(baseline_path, 'run_years', 'total_increment');
sample_indices = round(S.sample_indices(:));
Xb = P.precipitation_prior_336x848_mm_day(:, sample_indices);
Xb_m = mean(Xb, 2);
Xb_p = Xb - Xb_m;
n_ens = size(Xb, 2);
H_base = S.HXb0;
assert(isequal(size(H_base), [202, 500]), 'unexpected-HXb-shape');
tree = round(G.tree_indices(:));
admissible = round(G.admissible_indices(:));
not_admissible = round(G.not_admissible_indices(:));
significant = round(G.archived_significant_indices(:));
assert(length(tree) == 201 && length(admissible) == 27 && length(not_admissible) == 174 && length(significant) == 100, 'group-count-conflict');
assert(isempty(intersect(admissible, not_admissible)) && isequal(sort([admissible; not_admissible]), tree), 'group-partition-conflict');

branch_names = {'baseline', 'oof_admissible_only', 'oof_not_admissible_only', 'archived_significant_only', 'zero_all_tree'};
zero_rows = {[], not_admissible, admissible, setdiff(tree, significant), tree};
nbranch = length(branch_names);
branch_hxb = cell(1, nbranch);
baseline_mean = mean(H_base, 2);
for bb = 1:nbranch
    H = H_base;
    rows = zero_rows{bb};
    if ~isempty(rows)
        H(rows, :) = repmat(baseline_mean(rows), 1, n_ens);
    end
    branch_hxb{bb} = H;
end
mean_invariance = zeros(1, nbranch);
for bb = 1:nbranch
    mean_invariance(bb) = max(abs(mean(branch_hxb{bb}, 2) - baseline_mean));
end
assert(max(mean_invariance) <= 1e-12, 'operator-mean-invariance-failed');

region_names = {'hma', 'west_lt80', 'central_80_90', 'east_ge90'};
region_lon_masks = {true(size(P.lon)), P.lon < 80, P.lon >= 80 & P.lon < 90, P.lon >= 90};
nregion = length(region_names);
weight_grid = repmat(cosd(P.lat(:)), 1, length(P.lon));
state_weights = zeros(size(Xb, 1), nregion);
for rr = 1:nregion
    mask_grid = repmat(region_lon_masks{rr}(:)', length(P.lat), 1);
    weights = weight_grid .* mask_grid;
    weights = weights ./ sum(weights, 'all');
    state_weights(:, rr) = weights(:);
end
prior_region_mean = state_weights' * Xb_m;

run_years = 1901:1920;
assert(isequal(round(B.run_years(:))', run_years), 'baseline-year-conflict');
nrun = length(run_years);
branch_total_increment = zeros(nrun, nbranch, nregion);
branch_tree_contribution = zeros(nrun, nbranch, nregion);
branch_glacier_contribution = zeros(nrun, nbranch, nregion);
branch_closure_error = zeros(nrun, nbranch, nregion);
branch_contribution = zeros(202, nrun, nbranch, nregion);
branch_innovation = nan(202, nrun, nbranch);

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
        innovation = y - HXb_m;
        branch_innovation(ni, yy, bb) = innovation;
        for rr = 1:nregion
            gain_weight = state_weights(:, rr)' * K;
            contribution = gain_weight(:) .* innovation(:);
            branch_contribution(ni, yy, bb, rr) = contribution;
            branch_total_increment(yy, bb, rr) = state_weights(:, rr)' * (Xb_m + K * innovation) - prior_region_mean(rr);
            branch_tree_contribution(yy, bb, rr) = sum(contribution(ismember(ni, tree)));
            branch_glacier_contribution(yy, bb, rr) = sum(contribution(ni == 202));
            branch_closure_error(yy, bb, rr) = sum(contribution) - branch_total_increment(yy, bb, rr);
        end
    end
end
assert(max(abs(branch_closure_error), [], 'all') <= 1e-12, 'increment-closure-failed');
baseline_hma_closure = max(abs(branch_total_increment(:, 1, 1) - B.total_increment(:)));
assert(baseline_hma_closure <= 1e-12, 'baseline-hma-closure-failed');
innovation_invariance = max(abs(branch_innovation - branch_innovation(:, :, 1)), [], 'all');
assert(innovation_invariance <= 1e-12, 'innovation-invariance-failed');
save(output_path, 'run_years', 'branch_names', 'region_names', 'tree', 'admissible', 'not_admissible', 'significant', 'prior_region_mean', 'branch_total_increment', 'branch_tree_contribution', 'branch_glacier_contribution', 'branch_closure_error', 'branch_contribution', 'mean_invariance', 'baseline_hma_closure', 'innovation_invariance', 'state_weights', '-v7.3');
fprintf('BASELINE_HMA_CLOSURE=%.17g\n', baseline_hma_closure);
fprintf('INNOVATION_INVARIANCE=%.17g\n', innovation_invariance);
disp('OOF_ADMISSIBILITY_OPERATOR_SENSITIVITY_COMPLETE');
end
