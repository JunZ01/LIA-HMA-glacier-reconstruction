function run_proxy271_psm_seasonality_delta_anchor(output_path, phyda_root, proxy9_path, combined_path)
% Build a common-random-number JJA-minus-annual PSM perturbation for proxy 271.

old_dir = pwd;
cleanup_dir = onCleanup(@() cd(old_dir));
old_rng = rng;
cleanup_rng = onCleanup(@() rng(old_rng));
cd(phyda_root);
addpath(genpath(phyda_root));

source = load(combined_path, 'combined6_data', 'p_lat', 'p_lon', 'archive', 'year', 'dataName');
column = find(abs(source.p_lat - 30.75) < 1e-10 & abs(source.p_lon - 77.80) < 1e-10);
assert(isscalar(column), 'proxy271-source-column-not-unique');
assert(column == 287, 'proxy271-source-column-changed');

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
rng_seed = 0;

rng(rng_seed, 'twister');
annual_cell = psm_uni(prxs, plat, plon, 1, archv_tp, itr, ...
    mdl_prior, p_yrs, clb_prd, year, 1, 12, 'bearth', 'gpcp', 't', 'bi');
rng(rng_seed, 'twister');
jja_cell = psm_uni(prxs, plat, plon, 1, archv_tp, itr, ...
    mdl_prior, p_yrs, clb_prd, year, 6, 8, 'bearth', 'gpcp', 't', 'bi');
rng(rng_seed, 'twister');
annual_repeat_cell = psm_uni(prxs, plat, plon, 1, archv_tp, itr, ...
    mdl_prior, p_yrs, clb_prd, year, 1, 12, 'bearth', 'gpcp', 't', 'bi');
annual = annual_cell{1};
jja = jja_cell{1};
annual_repeat = annual_repeat_cell{1};

archived = load(proxy9_path, 'proxy_data');
baseline = archived.proxy_data.full_proxy{271};

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

annual_repeat_hxb = annual_repeat.HXb(:)';
annual_repeat_mp = annual_repeat.M_p(:)';
jja_hxb_delta = jja_hxb - annual_hxb;
anchored_annual_hxb = baseline_hxb;
anchored_jja_hxb = baseline_hxb + jja_hxb_delta;

save(output_path, 'column', 'year', 'p_yrs', 'clb_prd', 'rng_seed', ...
    'baseline_data', 'baseline_b', 'baseline_hxb', 'baseline_mt', 'baseline_mp', ...
    'baseline_rvec', 'baseline_r', 'baseline_pcr', 'baseline_pcr_t', 'baseline_pcr_p', 'baseline_pval', ...
    'annual_data', 'annual_b', 'annual_hxb', 'annual_mt', 'annual_mp', ...
    'annual_rvec', 'annual_r', 'annual_pcr', 'annual_pcr_t', 'annual_pcr_p', 'annual_pval', ...
    'jja_data', 'jja_b', 'jja_hxb', 'jja_mt', 'jja_mp', ...
    'jja_rvec', 'jja_r', 'jja_pcr', 'jja_pcr_t', 'jja_pcr_p', 'jja_pval', ...
    'annual_repeat_hxb', 'annual_repeat_mp', 'jja_hxb_delta', ...
    'anchored_annual_hxb', 'anchored_jja_hxb', '-v7');
disp('PROXY271_SEASONALITY_DELTA_ANCHOR_COMPLETE');
end