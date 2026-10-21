function run_calibration_windows_v3(output_path, phyda_root, proxy9_path)
% Replay equal-length windows from archived record chronologies on 1000-2000.

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
archv_tp = cell(1, 297);
for ii = 1:297
    item = full_proxy{ii};
    item_year = round(item.age_yng(:));
    item_data = item.data(:);
    [present, positions] = ismember(item_year, year);
    assert(all(present), 'record-year-outside-reconstruction-axis');
    assert(numel(unique(item_year)) == numel(item_year), 'duplicate-record-year');
    prxs(positions, ii) = item_data;
    plat(ii) = item.location(1);
    plon(ii) = item.location(2);
    archv_tp{ii} = item.ptype;
end

mdl_prior = 'cesm_lme010';
[gcm_syr, ~] = modelyrs(mdl_prior);
p_yrs = (1001:1848) - gcm_syr + 1;
windows = [1901 1961; 1920 1980; 1940 2000];
nwin = size(windows, 1);
rng_seed = 0;
itr = 1;

hxb = nan(297, nwin, 848);
b = nan(297, nwin, 3);
pcr = nan(297, nwin);
pcr_t = nan(297, nwin);
pcr_p = nan(297, nwin);
pval = nan(297, nwin);
residual_variance = nan(297, nwin);
data_max_abs = nan(297, nwin);

for ww = 1:nwin
    clb_prd = windows(ww, :);
    rng(rng_seed, 'twister');
    tree = psm_uni(prxs(:, 1:290), plat(1:290), plon(1:290), 290, archv_tp(1:290), itr, ...
        mdl_prior, p_yrs, clb_prd, year, 1, 12, 'bearth', 'gpcp', 't', 'bi');
    other = psm_ul(prxs(:, 291:297), plat(291:297), plon(291:297), 7, archv_tp(291:297), itr, ...
        mdl_prior, p_yrs, clb_prd, year, 1, 12, 'bearth', 't');
    current = [tree, other];
    for ii = 1:297
        item = current{ii};
        hxb(ii, ww, :) = item.HXb(:);
        coef = item.b(:);
        b(ii, ww, 1:numel(coef)) = coef;
        pcr(ii, ww) = item.pcr;
        if isfield(item, 'pcr_t'); pcr_t(ii, ww) = item.pcr_t; end
        if isfield(item, 'pcr_p'); pcr_p(ii, ww) = item.pcr_p; end
        pval(ii, ww) = item.pval;
        residual_variance(ii, ww) = nanvar(item.Rvec, 0, 2);
        source = full_proxy{ii};
        [~, positions] = ismember(round(source.age_yng(:)), year);
        data_max_abs(ii, ww) = max(abs(item.data(positions) - source.data(:)));
    end
end

rng(rng_seed, 'twister');
tree_repeat = psm_uni(prxs(:, 1:290), plat(1:290), plon(1:290), 290, archv_tp(1:290), itr, ...
    mdl_prior, p_yrs, windows(3, :), year, 1, 12, 'bearth', 'gpcp', 't', 'bi');
other_repeat = psm_ul(prxs(:, 291:297), plat(291:297), plon(291:297), 7, archv_tp(291:297), itr, ...
    mdl_prior, p_yrs, windows(3, :), year, 1, 12, 'bearth', 't');
repeat = [tree_repeat, other_repeat];
baseline_repeat_hxb = nan(297, 848);
for ii = 1:297
    baseline_repeat_hxb(ii, :) = repeat{ii}.HXb(:)';
end

save(output_path, 'windows', 'rng_seed', 'p_yrs', 'hxb', 'b', 'pcr', ...
    'pcr_t', 'pcr_p', 'pval', 'residual_variance', 'data_max_abs', ...
    'baseline_repeat_hxb', '-v7.3');
disp('PROXY9_CALIBRATION_WINDOW_STABILITY_V3_COMPLETE');
end
