function run_pooled_inner_controls(output_path, phyda_root, proxy9_path)
% Fit pooled 1901-1980 complement controls without outer-test leakage.

old_dir = pwd;
cleanup_dir = onCleanup(@() cd(old_dir));
old_rng = rng;
cleanup_rng = onCleanup(@() rng(old_rng));
cd(phyda_root);
addpath(genpath(phyda_root));

archived = load(proxy9_path, 'proxy_data');
full_proxy = archived.proxy_data.full_proxy;
assert(numel(full_proxy) == 297, 'unexpected-record-count');
year = (1000:2000)';
prxs = nan(numel(year), 297);
plat = nan(1, 297);
plon = nan(1, 297);
for ii = 1:297
    item = full_proxy{ii};
    item_year = round(item.age_yng(:));
    item_data = item.data(:);
    [present, positions] = ismember(year, item_year);
    assert(all(present), 'reconstruction-year-missing-from-archive-axis');
    prxs(:, ii) = item_data(positions);
    plat(ii) = item.location(1);
    plon(ii) = item.location(2);
end

[O_t, xlat, xlon, x_yrs] = load_obs_S('bearth', 1, 12);
[O_p, xlat2, xlon2, x_yrs2] = load_obs_S('gpcp', 1, 12);
pl_o1 = nearest_latlon(xlat, xlon, plat, plon);
pl_o2 = nearest_latlon(xlat2, xlon2, plat, plon);
O_p = O_p / 30;
[~, ref_idx_obs_p] = intersect(x_yrs2, 1961:1990);
O_p_anom = O_p - nanmean(O_p(:, ref_idx_obs_p), 2);

mdl_prior = 'cesm_lme010';
[gcm_syr, ~] = modelyrs(mdl_prior);
p_yrs = (1001:1848) - gcm_syr + 1;
rng_seed = 0;
rng(rng_seed, 'twister');
[M_t, xmeta] = load_S(mdl_prior, 't', 1, 12, p_yrs);
[M_p, ~] = load_S(mdl_prior, 'p', 1, 12, p_yrs);
pl_m = nearest_latlon(xmeta{1}.lat, xmeta{1}.lon, plat, plon);
M_t_centered = M_t(pl_m, :) - nanmean(M_t(pl_m, :), 2);
M_p_centered = M_p(pl_m, :) - nanmean(M_p(pl_m, :), 2);

blocks = [1901 1920; 1921 1940; 1941 1960; 1961 1980];
nfold = size(blocks, 1);
pooled_hxb = nan(297, nfold + 1, 848);
pooled_b = nan(297, nfold + 1, 3);
training_count = zeros(297, nfold + 1);
design_rank = nan(297, nfold + 1);
condition_number = nan(297, nfold + 1);
eligible = false(297, nfold + 1);

for branch = 1:(nfold + 1)
    if branch <= nfold
        test_years = (blocks(branch, 1):blocks(branch, 2))';
        train_years = setdiff((1901:1980)', test_years);
    else
        train_years = (1901:1980)';
    end
    [present_t, obs_train_t] = ismember(train_years, x_yrs(:));
    [present_p, obs_train_p] = ismember(train_years, x_yrs2(:));
    [present_y, proxy_train] = ismember(train_years, year);
    assert(all(present_t) && all(present_p) && all(present_y), 'pooled-training-year-mapping-failed');
    temp_mean = nanmean(O_t(pl_o1, obs_train_t), 2);
    prcp_mean = nanmean(O_p_anom(pl_o2, obs_train_p), 2);
    M_tbc = M_t_centered + temp_mean;
    M_pbc = M_p_centered + prcp_mean;

    for kk = 1:297
        pd = prxs(proxy_train, kk);
        ot = O_t(pl_o1(kk), obs_train_t)';
        if kk <= 290
            op = O_p_anom(pl_o2(kk), obs_train_p)';
            valid = isfinite(pd) & isfinite(ot) & isfinite(op);
            X = [ones(sum(valid), 1), ot(valid), op(valid)];
        else
            valid = isfinite(pd) & isfinite(ot);
            X = [ones(sum(valid), 1), ot(valid)];
        end
        training_count(kk, branch) = sum(valid);
        if sum(valid) < 20
            continue;
        end
        singular_values = svd(X, 0);
        tolerance = max(size(X)) * eps(max(singular_values));
        design_rank(kk, branch) = sum(singular_values > tolerance);
        if min(singular_values) == 0
            condition_number(kk, branch) = Inf;
        else
            condition_number(kk, branch) = max(singular_values) / min(singular_values);
        end
        if design_rank(kk, branch) < size(X, 2)
            continue;
        end
        coef = X \ pd(valid);
        pooled_b(kk, branch, 1:numel(coef)) = coef;
        if kk <= 290
            pooled_hxb(kk, branch, :) = coef(1) + coef(2) * M_tbc(kk, :) + coef(3) * M_pbc(kk, :);
        else
            pooled_hxb(kk, branch, :) = coef(1) + coef(2) * M_tbc(kk, :);
        end
        eligible(kk, branch) = true;
    end
end

save(output_path, 'year', 'p_yrs', 'rng_seed', 'blocks', 'pooled_hxb', ...
    'pooled_b', 'training_count', 'design_rank', 'condition_number', ...
    'eligible', '-v7.3');
fprintf('POOLED_COMPLEMENT_ELIGIBLE=%s\n', mat2str(sum(eligible(:, 1:4), 1)));
fprintf('POOLED_PRODUCTION_ELIGIBLE=%d\n', sum(eligible(:, 5)));
disp('PROXY9_BLOCK_ENSEMBLE_POOLED_CONTROLS_V1_COMPLETE');
end
