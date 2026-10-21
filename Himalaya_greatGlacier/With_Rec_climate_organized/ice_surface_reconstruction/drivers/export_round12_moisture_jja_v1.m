function export_round12_moisture_jja_v1(network_path, output_path, phyda_root)
% Export JJA instrumental predictors for the frozen expanded moisture network.

old_dir = pwd;
cleanup_dir = onCleanup(@() cd(old_dir));
cd(phyda_root);
addpath(genpath(phyda_root));

s = load(network_path, 'years', 'observations', 'locations', 'identity_ids');
years = round(s.years(:));
assert(isequal(years, (1940:2000)'), 'unexpected-year-axis');
assert(~any(years >= 1921 & years <= 1939), 'locked-period-requested');
assert(isequal(size(s.observations), [61 22]), 'unexpected-network-shape');

plat = s.locations(:, 1)';
plon = s.locations(:, 2)';
[O_t, lat_t, lon_t, years_t] = load_obs_S('bearth', 6, 8);
[O_g, lat_g, lon_g, years_g] = load_obs_S('gpcp', 6, 8);
close all force;
[O_c, lat_c, lon_c, years_c] = load_obs_S('cru_ts4_pre', 6, 8);
close all force;

site_t = nearest_latlon(lat_t, lon_t, plat, plon);
site_g = nearest_latlon(lat_g, lon_g, plat, plon);
site_c = nearest_latlon(lat_c, lon_c, plat, plon);
[ok_t, index_t] = ismember(years, years_t(:));
[ok_g, index_g] = ismember(years, years_g(:));
[ok_c, index_c] = ismember(years, years_c(:));
assert(all(ok_t) && all(ok_g) && all(ok_c), 'instrumental-year-mapping-failed');

O_g = O_g / 30;
O_c = O_c / 30;
[~, ref_g] = intersect(years_g(:), (1961:1990)');
[~, ref_c] = intersect(years_c(:), (1961:1990)');
O_g = O_g - mean(O_g(:, ref_g), 2, 'omitnan');
O_c = O_c - mean(O_c(:, ref_c), 2, 'omitnan');

observations = s.observations;
locations = s.locations;
identity_ids = s.identity_ids;
temperature_jja = O_t(site_t, index_t)';
precipitation_gpcc_jja = O_g(site_g, index_g)';
precipitation_cru_jja = O_c(site_c, index_c)';
save(output_path, 'years', 'observations', 'locations', 'identity_ids', ...
    'temperature_jja', 'precipitation_gpcc_jja', 'precipitation_cru_jja', ...
    'site_t', 'site_g', 'site_c', '-v7.3');
disp('LOOP_V2_ROUND12_MOISTURE_JJA_EXPORT_COMPLETE');
end
