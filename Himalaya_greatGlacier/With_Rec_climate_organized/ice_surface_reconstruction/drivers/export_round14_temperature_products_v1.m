function export_round14_temperature_products_v1(network_path, output_path, phyda_root)
% Export record-season Berkeley Earth and CRU TS temperatures.

old_dir = pwd;
cleanup_dir = onCleanup(@() cd(old_dir));
cd(phyda_root);
addpath(genpath(phyda_root));
s = load(network_path, 'years', 'observations', 'locations', 'identity_ids');
years = round(s.years(:));
assert(isequal(years, (1940:2000)'), 'unexpected-year-axis');
assert(~any(years >= 1921 & years <= 1939), 'locked-period-requested');
plat = s.locations(:, 1)'; plon = s.locations(:, 2)';

products = {'bearth', 'cru_ts4_tmp'};
season_starts = [1 6 4 4];
season_ends = [12 8 6 4];
temperature = nan(61, size(s.locations, 1), 4, 2);
sites = nan(size(s.locations, 1), 4, 2);
for product = 1:2
    for season = 1:4
        [source, lat, lon, source_years] = load_obs_S(products{product}, season_starts(season), season_ends(season));
        close all force;
        site = nearest_latlon(lat, lon, plat, plon);
        [ok, indices] = ismember(years, source_years(:));
        assert(all(ok), 'instrumental-year-mapping-failed');
        temperature(:, :, season, product) = source(site, indices)';
        sites(:, season, product) = site(:);
    end
end
observations = s.observations;
locations = s.locations;
identity_ids = s.identity_ids;
season_names = {'ANN', 'JJA', 'AMJ', 'APR'};
product_names = {'Berkeley_Earth', 'CRU_TS4_04'};
save(output_path, 'years', 'observations', 'locations', 'identity_ids', ...
    'temperature', 'sites', 'season_names', 'product_names', '-v7.3');
disp('LOOP_V2_ROUND14_TEMPERATURE_PRODUCTS_COMPLETE');
end
