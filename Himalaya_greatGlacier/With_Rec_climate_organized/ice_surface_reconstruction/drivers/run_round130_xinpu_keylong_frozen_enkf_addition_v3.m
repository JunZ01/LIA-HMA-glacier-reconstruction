function run_round130_xinpu_keylong_frozen_enkf_addition_v3(replay_path, prior_path, seasonal_path, input_path, output_path)
% Run frozen 1000-1920 EnKF branches with Xinpu and Keylong additions.

S = load(replay_path);
P = load(prior_path, 'precipitation_prior_336x848_mm_day');
Q = load(seasonal_path);
I = load(input_path);
sample_indices = round(S.sample_indices(:));
Xb = P.precipitation_prior_336x848_mm_day(:, sample_indices);
years = round(I.years(:))';
assert(isequal(years, 1000:1920));
assert(size(I.y_new, 1) == 2 && size(I.y_new, 2) == length(years));
assert(all(sample_indices >= 1 & sample_indices <= 848));

nlat = length(Q.lat);
[~, xinpu_ilat] = min(abs(Q.lat(:) - 30.42));
[~, xinpu_ilon] = min(abs(Q.lon(:) - 95.07));
[~, keylong_ilat] = min(abs(Q.lat(:) - 32.75));
[~, keylong_ilon] = min(abs(Q.lon(:) - 76.62));
xinpu_cell = xinpu_ilat + (xinpu_ilon - 1) * nlat;
keylong_cell = keylong_ilat + (keylong_ilon - 1) * nlat;

xinpu_t = Q.jja_temperature(xinpu_cell, :); xinpu_t = xinpu_t - mean(xinpu_t);
xinpu_p = Q.jja_precipitation(xinpu_cell, :); xinpu_p = xinpu_p - mean(xinpu_p);
keylong_t = Q.jja_temperature(keylong_cell, :); keylong_t = keylong_t - mean(keylong_t);
keylong_p = Q.ondjfma_precipitation(keylong_cell, :); keylong_p = keylong_p - mean(keylong_p);

products = {'bea', 'cru'};
new_hxb = zeros(2, 500, 2);
new_rvar = zeros(2, 2);
for p = 1:2
    tag = products{p};
    xc = I.(['xinpu_coeff_' tag]); kc = I.(['keylong_coeff_' tag]);
    xfull = xc(1) + xc(2) * xinpu_t + xc(3) * xinpu_p;
    kfull = kc(1) + kc(2) * keylong_t + kc(3) * keylong_p;
    new_hxb(1, :, p) = xfull(sample_indices);
    new_hxb(2, :, p) = kfull(sample_indices);
    new_rvar(1, p) = I.(['xinpu_rvar_' tag]);
    new_rvar(2, p) = I.(['keylong_rvar_' tag]);
end

branch_names = {'baseline','xinpu_bea','xinpu_cru','keylong_bea','keylong_cru','both_bea','both_cru'};
branch_new_rows = {[],1,1,2,2,[1 2],[1 2]};
branch_product = [0,1,2,1,2,1,2];
nrun = length(years); nbranch = length(branch_names);
posterior_mean = zeros(336, nrun, nbranch);
posterior_sigma = zeros(336, nrun, nbranch);
active_existing_count = zeros(1, nrun);
active_new_count = zeros(nbranch, nrun);
innovation_z = nan(2, nrun, 2);

for p = 1:2
    for j = 1:2
        active = isfinite(I.y_new(j, :));
        variance = var(new_hxb(j, :, p), 0, 2) + new_rvar(j, p);
        innovation_z(j, active, p) = (I.y_new(j, active) - mean(new_hxb(j, :, p), 2)) / sqrt(variance);
    end
end

for t = 1:nrun
    ni = find(~isnan(S.y0(:, t)));
    active_existing_count(t) = length(ni);
    y_existing = S.y0(ni, t);
    h_existing = S.HXb0(ni, :);
    residual_subset = S.R0v_filled(ni, :);
    if exist('nancov', 'file')
        r_existing = diag(diag(nancov(residual_subset')));
    else
        r_existing = diag(var(residual_subset, 0, 2));
    end
    for b = 1:nbranch
        rows = branch_new_rows{b};
        p = branch_product(b);
        if isempty(rows)
            active_rows = [];
        else
            active_rows = rows(isfinite(I.y_new(rows, t)));
        end
        active_new_count(b, t) = length(active_rows);
        if isempty(active_rows)
            y = y_existing; h = h_existing; r = r_existing;
        else
            y = [y_existing; I.y_new(active_rows, t)];
            h = [h_existing; reshape(new_hxb(active_rows, :, p), length(active_rows), 500)];
            r = blkdiag(r_existing, diag(new_rvar(active_rows, p)));
        end
        [xa, xa_mean] = M_update(Xb, h, y, r, 1);
        posterior_mean(:, t, b) = xa_mean;
        posterior_sigma(:, t, b) = std(xa, 0, 2);
    end
    if mod(t, 100) == 0 || t == nrun
        fprintf('Round130 progress: %d/%d\n', t, nrun);
    end
end

baseline_max_abs_closure = max(abs(posterior_mean(:, :, 1) - S.target_mean(:, 1:nrun)), [], 'all');
fprintf('Baseline max closure: %.12g mm/day\n', baseline_max_abs_closure);
assert(baseline_max_abs_closure <= 0.05, 'baseline-replay-approved-gate-failed');
save(output_path, 'years', 'branch_names', 'posterior_mean', 'posterior_sigma', ...
    'active_existing_count', 'active_new_count', 'innovation_z', 'new_hxb', ...
    'new_rvar', 'xinpu_cell', 'keylong_cell', 'baseline_max_abs_closure', '-v7.3');
disp('ROUND130_XINPU_KEYLONG_FROZEN_ENKF_COMPLETE');
end


