function run_proxy271_jja_positive_constraint(output_path, phyda_root, proxy9_path, combined_path)
% Test a positive-temperature constrained JJA PSM for proxy9 record 271.

old_dir = pwd;
cleanup_dir = onCleanup(@() cd(old_dir));
old_rng = rng;
cleanup_rng = onCleanup(@() rng(old_rng));
cd(phyda_root);
addpath(genpath(phyda_root));

source = load(combined_path, 'combined6_data', 'p_lat', 'p_lon', 'archive', 'year');
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

% The unconstrained JJA optimum has b_T < 0. Under b_T >= 0, the quadratic
% least-squares optimum lies at b_T=0, then intercept and precipitation refit.
cal_years = clb_prd(1):clb_prd(2);
[~, proxy_idx] = intersect(year, cal_years);
% jja.M_t and jja.M_p are prior predictors, not observation calibration predictors.
% Recreate the observed JJA predictor rows exactly as psm_uni constructs them.
[O_t, xlat, xlon, x_yrs] = load_obs_S('bearth', 6, 8);
[O_p, xlat2, xlon2, x_yrs2] = load_obs_S('gpcp', 6, 8);
mask = sum(~isnan(O_t), 2); mask(mask == 0) = NaN; mask(~isnan(mask)) = 1;
pl_o1 = nearest_latlon(xlat, xlon, plat, plon, mask);
pl_o2 = nearest_latlon(xlat2, xlon2, plat, plon, mask);
[~, io_t] = intersect(x_yrs, cal_years);
[~, io_p] = intersect(x_yrs2, cal_years);
ref_period = 1961:1990;
[~, ref_idx_obs_p] = intersect(x_yrs2, ref_period);
O_p = O_p / 30;
O_p_ref = nanmean(O_p(:, ref_idx_obs_p), 2);
O_p_anom = O_p - O_p_ref;
o_t = O_t(pl_o1, io_t)';
o_p = O_p_anom(pl_o2, io_p)';
pd = prxs(proxy_idx);
valid = ~isnan(o_t) & ~isnan(o_p) & ~isnan(pd);
assert(sum(valid) >= 20, 'proxy271-constrained-jja-valid-count');

X_unconstrained = [ones(sum(valid), 1), o_t(valid), o_p(valid)];
y_unconstrained = pd(valid);
b_unconstrained_check = X_unconstrained \ y_unconstrained;
assert(abs(b_unconstrained_check(2) - jja_b(2)) < 1e-10, 'proxy271-jja-unconstrained-coefficient-mismatch');
X_boundary = [ones(sum(valid), 1), o_p(valid)];
b_boundary = X_boundary \ y_unconstrained;
constrained_b = [b_boundary(1); 0; b_boundary(2)];
y_pred_constrained = X_unconstrained * constrained_b;
constrained_rvec = zeros(size(valid));
constrained_rvec(valid) = y_unconstrained - y_pred_constrained;
[constrained_pcr, constrained_pval] = corr(y_unconstrained, y_pred_constrained, 'rows', 'complete');
constrained_pcr_t = corr(y_unconstrained, o_t(valid), 'rows', 'complete');
constrained_pcr_p = corr(y_unconstrained, o_p(valid), 'rows', 'complete');
constrained_hxb = constrained_b(1) + constrained_b(2) * jja_mt + constrained_b(3) * jja_mp;

annual_repeat_hxb = annual_repeat.HXb(:)';
annual_repeat_mp = annual_repeat.M_p(:)';
constrained_jja_hxb_delta = constrained_hxb - annual_hxb;
anchored_annual_hxb = baseline_hxb;
anchored_constrained_jja_hxb = baseline_hxb + constrained_jja_hxb_delta;

save(output_path, 'column', 'year', 'p_yrs', 'clb_prd', 'rng_seed', ...
    'baseline_data', 'baseline_b', 'baseline_hxb', 'baseline_mt', 'baseline_mp', ...
    'baseline_rvec', 'baseline_r', 'baseline_pcr', 'baseline_pcr_t', 'baseline_pcr_p', 'baseline_pval', ...
    'annual_data', 'annual_b', 'annual_hxb', 'annual_mt', 'annual_mp', ...
    'annual_rvec', 'annual_r', 'annual_pcr', 'annual_pcr_t', 'annual_pcr_p', 'annual_pval', ...
    'jja_data', 'jja_b', 'jja_hxb', 'jja_mt', 'jja_mp', ...
    'jja_rvec', 'jja_r', 'jja_pcr', 'jja_pcr_t', 'jja_pcr_p', 'jja_pval', ...
    'annual_repeat_hxb', 'annual_repeat_mp', 'constrained_b', 'constrained_hxb', ...
    'constrained_rvec', 'constrained_pcr', 'constrained_pcr_t', 'constrained_pcr_p', 'constrained_pval', ...
    'constrained_jja_hxb_delta', 'anchored_annual_hxb', 'anchored_constrained_jja_hxb', '-v7');
disp('PROXY271_POSITIVE_CONSTRAINT_COMPLETE');
end