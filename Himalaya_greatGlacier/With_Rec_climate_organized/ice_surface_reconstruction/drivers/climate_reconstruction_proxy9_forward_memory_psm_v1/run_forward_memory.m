function run_forward_memory(output_path, phyda_root, proxy9_path)
% Build expanding and exponentially weighted forward-validation PSMs.

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

train_periods = [1901 1920; 1901 1940; 1901 1960; 1901 1980];
test_blocks = [1921 1940; 1941 1960; 1961 1980];
memory_years = [Inf 10 20 40];
nmethod = numel(memory_years);
nbranch = size(train_periods, 1);
hxb = nan(297, nmethod, nbranch, 848);
b = nan(297, nmethod, nbranch, 3);
training_count = zeros(297, nmethod, nbranch);
effective_sample_size = nan(297, nmethod, nbranch);
design_rank = nan(297, nmethod, nbranch);
condition_number = nan(297, nmethod, nbranch);
eligible = false(297, nmethod, nbranch);

for branch = 1:nbranch
    train_years = (train_periods(branch, 1):train_periods(branch, 2))';
    [present_t, obs_train_t] = ismember(train_years, x_yrs(:));
    [present_p, obs_train_p] = ismember(train_years, x_yrs2(:));
    [present_y, proxy_train] = ismember(train_years, year);
    assert(all(present_t) && all(present_p) && all(present_y), 'training-year-mapping-failed');

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
        valid_years = train_years(valid);
        y = pd(valid);
        if numel(y) < 20
            continue;
        end

        for mm = 1:nmethod
            if isinf(memory_years(mm))
                weights = ones(size(y));
            else
                weights = exp((valid_years - max(valid_years)) / memory_years(mm));
            end
            training_count(kk, mm, branch) = numel(y);
            effective_sample_size(kk, mm, branch) = sum(weights)^2 / sum(weights.^2);
            Xw = X .* sqrt(weights);
            yw = y .* sqrt(weights);
            singular_values = svd(Xw, 0);
            tolerance = max(size(Xw)) * eps(max(singular_values));
            design_rank(kk, mm, branch) = sum(singular_values > tolerance);
            if min(singular_values) == 0
                condition_number(kk, mm, branch) = Inf;
            else
                condition_number(kk, mm, branch) = max(singular_values) / min(singular_values);
            end
            if design_rank(kk, mm, branch) < size(Xw, 2)
                continue;
            end
            coef = Xw \ yw;
            b(kk, mm, branch, 1:numel(coef)) = coef;
            temp_mean = sum(weights .* ot(valid)) / sum(weights);
            M_tbc = M_t_centered(kk, :) + temp_mean;
            if kk <= 290
                prcp_mean = sum(weights .* op(valid)) / sum(weights);
                M_pbc = M_p_centered(kk, :) + prcp_mean;
                hxb(kk, mm, branch, :) = coef(1) + coef(2) * M_tbc + coef(3) * M_pbc;
            else
                hxb(kk, mm, branch, :) = coef(1) + coef(2) * M_tbc;
            end
            eligible(kk, mm, branch) = true;
        end
    end
end

save(output_path, 'year', 'p_yrs', 'rng_seed', 'train_periods', ...
    'test_blocks', 'memory_years', 'hxb', 'b', 'training_count', ...
    'effective_sample_size', 'design_rank', 'condition_number', ...
    'eligible', '-v7.3');
for branch = 1:nbranch
    fprintf('ELIGIBLE_BRANCH_%d=%s\n', branch, mat2str(sum(eligible(:, :, branch), 1)));
end
disp('PROXY9_FORWARD_MEMORY_PSM_V1_COMPLETE');
end
