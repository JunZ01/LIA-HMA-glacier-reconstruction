function export_climate_reconstruction_round03_production_proxy9_v2(output_path, phyda_root, combined_path)
% Rebuild the production proxy9 operator with its original RNG and PSM choices.

old_dir = pwd;
cleanup_dir = onCleanup(@() cd(old_dir));
old_rng = rng;
cleanup_rng = onCleanup(@() rng(old_rng));
cd(phyda_root);
addpath(genpath(phyda_root));

rng(4236, 'twister');
source = load(combined_path);
required = {'combined6_data', 'p_lat', 'p_lon', 'archive', 'year'};
for ii = 1:numel(required)
    assert(isfield(source, required{ii}), ['missing-source-variable-' required{ii}]);
end

prx_i = source.combined6_data;
p_lat = source.p_lat;
p_lon = source.p_lon;
archive = source.archive;
year = source.year(:)';
prx_im = bsxfun(@minus, prx_i, nanmean(prx_i, 1));
prx_d = bsxfun(@rdivide, prx_im, nanstd(prx_im, 0, 1));

mdl_prior = 'cesm_lme010';
[gcm_syr, ~] = modelyrs(mdl_prior);
p_yrs = (1001:1848) - gcm_syr + 1;
clb_prd = [1940, 2000];
itr = 1;

tree_idx = find(ismember(archive, {'tree', 'Tree Rings'}));
tree = psm_uni(prx_d(:, tree_idx), p_lat(tree_idx), p_lon(tree_idx), ...
    numel(tree_idx), archive(tree_idx), itr, mdl_prior, p_yrs, clb_prd, ...
    year, 1, 12, 'bearth', 'gpcp', 't', 'bi');

other_idx = find(~ismember(archive, {'coral', 'tree', 'sclerosponge', ...
    'marine sediment', 'Corals and Sclerosponges', 'Tree Rings', 'Bivalve'}));
other_data = prx_d(:, other_idx);
other_lat = p_lat(other_idx);
other_lon = p_lon(other_idx);
other_archive = archive(other_idx);
outliers = find(max(abs(other_data)) > 7.5);
other_data(:, outliers) = [];
other_lat(outliers) = [];
other_lon(outliers) = [];
other_archive(outliers) = [];
other = psm_ul(other_data, other_lat, other_lon, numel(other_lon), ...
    other_archive, itr, mdl_prior, p_yrs, clb_prd, year, 1, 12, 'bearth', 't');

proxy = [tree, other];
assert(numel(proxy) == 297, 'unexpected-proxy9-count');
years = (1940:2000)';
hxb = nan(297, 848);
observations = nan(61, 297);
residual_matrix = nan(297, 61);
coefficients = nan(297, 3);
for ii = 1:297
    item = proxy{ii};
    hxb(ii, :) = item.HXb(:)';
    item_years = round(item.age_yng(:));
    [present, positions] = ismember(years, item_years);
    assert(all(present), 'allowed-year-missing-from-proxy-axis');
    item_data = item.data(:);
    observations(:, ii) = item_data(positions);
    residual_matrix(ii, :) = item.Rvec(:)';
    coef = item.b(:);
    coefficients(ii, 1:numel(coef)) = coef;
end

save(output_path, 'years', 'hxb', 'observations', 'residual_matrix', ...
    'coefficients', 'p_yrs', '-v7.3');
disp('ROUND03_PRODUCTION_PROXY9_EXPORT_V2_COMPLETE');
end
