function export_round37_dod2k_annual_climate_v1(input_path, output_path, phyda_root)
% Export formal-season instrumental predictors for the Round 37 records.

old_dir = pwd;
cleanup_dir = onCleanup(@() cd(old_dir));
cd(phyda_root);
addpath(genpath(phyda_root));

s = load(input_path);
years = round(s.years(:));
assert(isequal(years, (1940:2000)'), 'unexpected-year-axis');
assert(~any(years >= 1921 & years <= 1939), 'locked-period-requested');
assert(isequal(size(s.observations), [61 3]), 'unexpected-observation-shape');

record_count = size(s.locations, 1);
temperature_formal_season = nan(61, record_count);
precipitation_gpcc_formal_season = nan(61, record_count);
precipitation_cru_formal_season = nan(61, record_count);
site_t = nan(record_count, 1);
site_g = nan(record_count, 1);
site_c = nan(record_count, 1);

for record = 1:record_count
    first_month = round(s.season_start_month(record));
    last_month = round(s.season_end_month(record));
    plat = s.locations(record, 1);
    plon = s.locations(record, 2);
    [O_t, lat_t, lon_t, years_t] = load_obs_S('bearth', first_month, last_month);
    [O_g, lat_g, lon_g, years_g] = load_obs_S('gpcp', first_month, last_month);
    close all force;
    [O_c, lat_c, lon_c, years_c] = load_obs_S('cru_ts4_pre', first_month, last_month);
    close all force;

    site_t(record) = nearest_latlon(lat_t, lon_t, plat, plon);
    site_g(record) = nearest_latlon(lat_g, lon_g, plat, plon);
    site_c(record) = nearest_latlon(lat_c, lon_c, plat, plon);
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
    temperature_formal_season(:, record) = O_t(site_t(record), index_t)';
    precipitation_gpcc_formal_season(:, record) = O_g(site_g(record), index_g)';
    precipitation_cru_formal_season(:, record) = O_c(site_c(record), index_c)';
end

observations = s.observations;
locations = s.locations;
identity_ids = s.identity_ids;
directions = s.directions;
season_start_month = s.season_start_month;
season_end_month = s.season_end_month;
save(output_path, 'years', 'observations', 'locations', 'identity_ids', 'directions', ...
    'season_start_month', 'season_end_month', 'temperature_formal_season', ...
    'precipitation_gpcc_formal_season', 'precipitation_cru_formal_season', ...
    'site_t', 'site_g', 'site_c', '-v7.3');
disp('LOOP_V2_ROUND37_DOD2K_ANNUAL_CLIMATE_EXPORT_COMPLETE');
end
