function export_climate_reconstruction_loop_v2_round08_temperature_seasons_v1(proxy_path, output_path, phyda_root)
% Export annual and documented seasonal Berkeley temperature at proxy9 sites.

old_dir = pwd;
cleanup_dir = onCleanup(@() cd(old_dir));
cd(phyda_root);
addpath(genpath(phyda_root));
s = load(proxy_path, 'years', 'locations');
years = round(s.years(:));
assert(isequal(years, (1940:2000)'), 'unexpected-year-axis');
assert(~any(years >= 1921 & years <= 1939), 'locked-period-requested');
plat = s.locations(:, 1)';
plon = s.locations(:, 2)';

[annual_source, lat, lon, source_years] = load_obs_S('bearth', 1, 12);
[jja_source, lat_jja, lon_jja, years_jja] = load_obs_S('bearth', 6, 8);
[amj_source, lat_amj, lon_amj, years_amj] = load_obs_S('bearth', 4, 6);
[april_source, lat_april, lon_april, years_april] = load_obs_S('bearth', 4, 4);
assert(isequal(lat, lat_jja) && isequal(lat, lat_amj) && isequal(lat, lat_april), 'latitude-grid-mismatch');
assert(isequal(lon, lon_jja) && isequal(lon, lon_amj) && isequal(lon, lon_april), 'longitude-grid-mismatch');
assert(isequal(source_years, years_jja) && isequal(source_years, years_amj) && isequal(source_years, years_april), 'season-year-mismatch');
sites = nearest_latlon(lat, lon, plat, plon);
[ok, indices] = ismember(years, source_years(:));
assert(all(ok), 'instrumental-year-mapping-failed');
annual = annual_source(sites, indices)';
jja = jja_source(sites, indices)';
amj = amj_source(sites, indices)';
april = april_source(sites, indices)';
save(output_path, 'years', 'annual', 'jja', 'amj', 'april', 'sites', '-v7.3');
disp('LOOP_V2_ROUND08_TEMPERATURE_SEASONS_COMPLETE');
end
