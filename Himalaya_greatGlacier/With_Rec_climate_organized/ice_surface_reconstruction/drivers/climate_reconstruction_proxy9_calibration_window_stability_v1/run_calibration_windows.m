function run_calibration_windows(output_path, selection_csv, phyda_root, proxy9_path, combined_path)
% Replay all proxy9 PSMs under three equal-length calibration windows.

old_dir = pwd;
cleanup_dir = onCleanup(@() cd(old_dir));
old_rng = rng;
cleanup_rng = onCleanup(@() rng(old_rng));
cd(phyda_root);
addpath(genpath(phyda_root));

selected = readtable(selection_csv, 'TextType', 'string');
assert(height(selected) == 297, 'unexpected-record-count');
assert(all(selected.proxy9_record_1based(:)' == 1:297), 'record-order-changed');
columns = selected.combined_column_1based(:)';
source = load(combined_path, 'combined6_data', 'p_lat', 'p_lon', 'archive', 'year');
archived = load(proxy9_path, 'proxy_data');

prxs = source.combined6_data(:, columns);
prxs = (prxs - nanmean(prxs, 1)) ./ nanstd(prxs, 0, 1);
plat = source.p_lat(columns);
plon = source.p_lon(columns);
archv_tp = source.archive(columns);
year = source.year(:);
for ii = 1:297
    assert(abs(plat(ii) - selected.latitude(ii)) < 1e-10, 'source-lat-changed');
    assert(abs(plon(ii) - selected.longitude(ii)) < 1e-10, 'source-lon-changed');
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
        baseline = archived.proxy_data.full_proxy{ii};
        data_max_abs(ii, ww) = max(abs(item.data(:) - baseline.data(:)));
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

save(output_path, 'windows', 'rng_seed', 'p_yrs', 'columns', 'hxb', 'b', 'pcr', ...
    'pcr_t', 'pcr_p', 'pval', 'residual_variance', 'data_max_abs', ...
    'baseline_repeat_hxb', '-v7.3');
disp('PROXY9_CALIBRATION_WINDOW_STABILITY_COMPLETE');
end
