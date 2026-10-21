function export_round40_low_frequency_climate_v1(input_path, output_path, phyda_root)
% Export annual/formal-season climate for Round 40 low-frequency records.
old_dir = pwd; cleanup_dir = onCleanup(@() cd(old_dir)); cd(phyda_root); addpath(genpath(phyda_root));
s = load(input_path); years = round(s.years(:));
assert(isequal(years, (1940:2000)') && ~any(years >= 1921 & years <= 1939), 'year-contract-failed');
record_count = 5; temperature = nan(61, record_count); precipitation_gpcc = nan(61, record_count); precipitation_cru = nan(61, record_count);
for record = 1:record_count
    first_month = round(s.season_start_month(record)); last_month = round(s.season_end_month(record));
    [O_t, lat_t, lon_t, years_t] = load_obs_S('bearth', first_month, last_month);
    [O_g, lat_g, lon_g, years_g] = load_obs_S('gpcp', first_month, last_month); close all force;
    [O_c, lat_c, lon_c, years_c] = load_obs_S('cru_ts4_pre', first_month, last_month); close all force;
    site_t = nearest_latlon(lat_t, lon_t, s.locations(record, 1), s.locations(record, 2));
    site_g = nearest_latlon(lat_g, lon_g, s.locations(record, 1), s.locations(record, 2));
    site_c = nearest_latlon(lat_c, lon_c, s.locations(record, 1), s.locations(record, 2));
    [ok_t, index_t] = ismember(years, years_t(:)); [ok_g, index_g] = ismember(years, years_g(:)); [ok_c, index_c] = ismember(years, years_c(:));
    assert(all(ok_t) && all(ok_g) && all(ok_c), 'instrumental-year-mapping-failed');
    O_g = O_g / 30; O_c = O_c / 30;
    [~, ref_g] = intersect(years_g(:), (1961:1990)'); [~, ref_c] = intersect(years_c(:), (1961:1990)');
    O_g = O_g - mean(O_g(:, ref_g), 2, 'omitnan'); O_c = O_c - mean(O_c(:, ref_c), 2, 'omitnan');
    temperature(:, record) = O_t(site_t, index_t)'; precipitation_gpcc(:, record) = O_g(site_g, index_g)'; precipitation_cru(:, record) = O_c(site_c, index_c)';
end
observations = s.observations; sample_counts = s.sample_counts; locations = s.locations; identity_ids = s.identity_ids;
save(output_path, 'years', 'observations', 'sample_counts', 'locations', 'identity_ids', 'temperature', 'precipitation_gpcc', 'precipitation_cru', '-v7.3');
disp('LOOP_V2_ROUND40_LOW_FREQUENCY_CLIMATE_EXPORT_COMPLETE');
end
