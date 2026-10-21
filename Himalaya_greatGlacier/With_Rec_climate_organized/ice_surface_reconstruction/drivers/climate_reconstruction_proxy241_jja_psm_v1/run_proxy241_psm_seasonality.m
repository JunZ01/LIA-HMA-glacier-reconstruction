function run_proxy241_psm_seasonality(output_path, phyda_root, proxy9_path, combined_path)
% Rebuild proxy 241 with annual and JJA bivariate PSMs using archived code.

old_dir = pwd;
cleanup = onCleanup(@() cd(old_dir));
cd(phyda_root);
addpath(genpath(phyda_root));

source = load(combined_path, 'combined6_data', 'p_lat', 'p_lon', 'archive', 'year', 'dataName');
column = find(abs(source.p_lat - 34.65) < 1e-10 & abs(source.p_lon - 75.47) < 1e-10);
assert(isscalar(column), 'proxy241-source-column-not-unique');
assert(column == 253, 'proxy241-source-column-changed');

raw = source.combined6_data(:, column);
prxs = (raw - nanmean(raw, 1)) ./ nanstd(raw, 0, 1);
plat = source.p_lat(column);
plon = source.p_lon(column);
archv_tp = source.archive(column);
year = source.year(:);

mdl_prior = 'cesm_lme010';
[gcm_syr, ~] = modelyrs(mdl_prior);
p_yrs = (1001:1848) - gcm_syr + 1;
clb_prd = [1940 2000];
itr = 1;

annual_cell = psm_uni(prxs, plat, plon, 1, archv_tp, itr, ...
    mdl_prior, p_yrs, clb_prd, year, 1, 12, 'bearth', 'gpcp', 't', 'bi');
jja_cell = psm_uni(prxs, plat, plon, 1, archv_tp, itr, ...
    mdl_prior, p_yrs, clb_prd, year, 6, 8, 'bearth', 'gpcp', 't', 'bi');
annual = annual_cell{1};
jja = jja_cell{1};

archived = load(proxy9_path, 'proxy_data');
baseline = archived.proxy_data.full_proxy{241};

baseline_data = baseline.data(:)';
baseline_b = baseline.b(:)';
baseline_hxb = baseline.HXb(:)';
baseline_mt = baseline.M_t(:)';
baseline_mp = baseline.M_p(:)';
baseline_rvec = baseline.Rvec(:)';
baseline_r = baseline.R(:)';
baseline_pcr = baseline.pcr;
baseline_pcr_t = baseline.pcr_t;
baseline_pcr_p = baseline.pcr_p;
baseline_pval = baseline.pval;

annual_data = annual.data(:)';
annual_b = annual.b(:)';
annual_hxb = annual.HXb(:)';
annual_mt = annual.M_t(:)';
annual_mp = annual.M_p(:)';
annual_rvec = annual.Rvec(:)';
annual_r = annual.R(:)';
annual_pcr = annual.pcr;
annual_pcr_t = annual.pcr_t;
annual_pcr_p = annual.pcr_p;
annual_pval = annual.pval;

jja_data = jja.data(:)';
jja_b = jja.b(:)';
jja_hxb = jja.HXb(:)';
jja_mt = jja.M_t(:)';
jja_mp = jja.M_p(:)';
jja_rvec = jja.Rvec(:)';
jja_r = jja.R(:)';
jja_pcr = jja.pcr;
jja_pcr_t = jja.pcr_t;
jja_pcr_p = jja.pcr_p;
jja_pval = jja.pval;

save(output_path, 'column', 'year', 'p_yrs', 'clb_prd', ...
    'baseline_data', 'baseline_b', 'baseline_hxb', 'baseline_mt', 'baseline_mp', ...
    'baseline_rvec', 'baseline_r', 'baseline_pcr', 'baseline_pcr_t', 'baseline_pcr_p', 'baseline_pval', ...
    'annual_data', 'annual_b', 'annual_hxb', 'annual_mt', 'annual_mp', ...
    'annual_rvec', 'annual_r', 'annual_pcr', 'annual_pcr_t', 'annual_pcr_p', 'annual_pval', ...
    'jja_data', 'jja_b', 'jja_hxb', 'jja_mt', 'jja_mp', ...
    'jja_rvec', 'jja_r', 'jja_pcr', 'jja_pcr_t', 'jja_pcr_p', 'jja_pval', '-v7');
disp('PROXY241_SEASONALITY_COMPLETE');
end
