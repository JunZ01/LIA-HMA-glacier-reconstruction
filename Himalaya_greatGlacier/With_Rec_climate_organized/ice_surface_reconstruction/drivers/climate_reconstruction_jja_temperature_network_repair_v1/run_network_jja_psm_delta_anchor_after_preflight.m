function run_network_jja_psm_delta_anchor_after_preflight(output_path, selection_csv, phyda_root, proxy9_path, combined_path, first_row, last_row)
% Compute post-preflight JJA-minus-annual PSM batches.
% The first five records established exact annual-repeat closure. Each record
% here still resets the same RNG before annual and JJA calls.

old_dir = pwd;
cleanup_dir = onCleanup(@() cd(old_dir));
old_rng = rng;
cleanup_rng = onCleanup(@() rng(old_rng));
cd(phyda_root);
addpath(genpath(phyda_root));

selected = readtable(selection_csv, 'TextType', 'string');
assert(first_row >= 1 && last_row <= height(selected) && first_row <= last_row, 'invalid-batch-range');
selected = selected(first_row:last_row, :);
record_ids = selected.proxy9_record_1based(:)';
columns = selected.combined_column_1based(:)';

source = load(combined_path, 'combined6_data', 'p_lat', 'p_lon', 'archive', 'year');
archived = load(proxy9_path, 'proxy_data');
mdl_prior = 'cesm_lme010';
[gcm_syr, ~] = modelyrs(mdl_prior);
p_yrs = (1001:1848) - gcm_syr + 1;
clb_prd = [1940 2000];
itr = 1;
rng_seed = 0;
n = numel(record_ids);

baseline_hxb = nan(n, 848);
annual_hxb = nan(n, 848);
jja_hxb = nan(n, 848);
anchored_jja_hxb = nan(n, 848);
baseline_b = nan(n, 3);
annual_b = nan(n, 3);
jja_b = nan(n, 3);
annual_pcr = nan(n, 1);
jja_pcr = nan(n, 1);
annual_pval = nan(n, 1);
jja_pval = nan(n, 1);
annual_data_max_abs = nan(n, 1);

for ii = 1:n
    record_id = record_ids(ii);
    column = columns(ii);
    assert(abs(source.p_lat(column) - selected.latitude(ii)) < 1e-10, 'source-lat-changed');
    assert(abs(source.p_lon(column) - selected.longitude(ii)) < 1e-10, 'source-lon-changed');
    raw = source.combined6_data(:, column);
    prxs = (raw - nanmean(raw, 1)) ./ nanstd(raw, 0, 1);
    plat = source.p_lat(column);
    plon = source.p_lon(column);
    archv_tp = source.archive(column);
    year = source.year(:);

    rng(rng_seed, 'twister');
    annual_cell = psm_uni(prxs, plat, plon, 1, archv_tp, itr, mdl_prior, p_yrs, clb_prd, year, 1, 12, 'bearth', 'gpcp', 't', 'bi');
    rng(rng_seed, 'twister');
    jja_cell = psm_uni(prxs, plat, plon, 1, archv_tp, itr, mdl_prior, p_yrs, clb_prd, year, 6, 8, 'bearth', 'gpcp', 't', 'bi');

    baseline = archived.proxy_data.full_proxy{record_id};
    annual = annual_cell{1};
    jja = jja_cell{1};
    baseline_hxb(ii, :) = baseline.HXb(:)';
    annual_hxb(ii, :) = annual.HXb(:)';
    jja_hxb(ii, :) = jja.HXb(:)';
    anchored_jja_hxb(ii, :) = baseline.HXb(:)' + (jja.HXb(:)' - annual.HXb(:)');
    baseline_b(ii, :) = baseline.b(:)';
    annual_b(ii, :) = annual.b(:)';
    jja_b(ii, :) = jja.b(:)';
    annual_pcr(ii) = annual.pcr;
    jja_pcr(ii) = jja.pcr;
    annual_pval(ii) = annual.pval;
    jja_pval(ii) = jja.pval;
    annual_data_max_abs(ii) = max(abs(annual.data(:)' - baseline.data(:)'));
end

save(output_path, 'first_row', 'last_row', 'record_ids', 'columns', 'rng_seed', 'p_yrs', 'clb_prd', ...
    'baseline_hxb', 'annual_hxb', 'jja_hxb', 'anchored_jja_hxb', ...
    'baseline_b', 'annual_b', 'jja_b', 'annual_pcr', 'jja_pcr', 'annual_pval', 'jja_pval', ...
    'annual_data_max_abs', '-v7');
disp('NETWORK_JJA_TEMPERATURE_POST_PREFLIGHT_BATCH_COMPLETE');
end
