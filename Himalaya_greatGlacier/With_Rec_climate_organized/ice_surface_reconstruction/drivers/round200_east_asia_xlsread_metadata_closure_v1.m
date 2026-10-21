function round200_east_asia_xlsread_metadata_closure_v1(input_xlsx, v3_csv, output_dir)
% Verify deposited xlsread metadata indices without opening annual data rows.

if exist(output_dir, 'dir')
    error('Refusing to overwrite independent result: %s', output_dir);
end
mkdir(output_dir);

[num, text, raw] = xlsread(input_xlsx, 1, 'A1:PH14'); %#ok<XLSRD>
if size(raw, 1) ~= 14
    error('Expected exactly 14 physical worksheet rows, found %d', size(raw, 1));
end
if size(raw, 2) ~= 424
    error('Expected 424 worksheet columns A:PH, found %d', size(raw, 2));
end
if size(num, 1) < 10 || size(num, 2) < 424
    error('Unexpected xlsread numeric-array dimensions: %dx%d', size(num,1), size(num,2));
end
if size(text, 1) < 12 || size(text, 2) < 424
    error('Unexpected xlsread text-array dimensions: %dx%d', size(text,1), size(text,2));
end

expected = readtable(v3_csv, 'VariableNamingRule', 'preserve');
if height(expected) ~= 423
    error('Expected 423 Round 199 metadata records, found %d', height(expected));
end

lat = num(5, 2:end)';
lon = num(6, 2:end)';
elev = num(7, 2:end)';
first_year = num(9, 2:end)';
last_year = num(10, 2:end)';
ids = string(text(1, 2:end))';
proxy_type = string(text(12, 2:end))';

if numel(lat) ~= height(expected)
    error('Deposited expression produces %d records, expected %d', numel(lat), height(expected));
end

numeric_error = [
    max(abs(lat - expected.latitude), [], 'omitnan'), ...
    max(abs(lon - expected.longitude), [], 'omitnan'), ...
    max(abs(elev - expected.elevation_m), [], 'omitnan'), ...
    max(abs(first_year - expected.first_year), [], 'omitnan'), ...
    max(abs(last_year - expected.last_year), [], 'omitnan')];
id_match = strcmp(ids, string(expected.generic_id));
type_match = strcmp(proxy_type, string(expected.archive_type));

closure = table(
    ["latitude"; "longitude"; "elevation_m"; "first_year"; "last_year"], ...
    numeric_error', ...
    'VariableNames', {'field', 'max_abs_error'});
writetable(closure, fullfile(output_dir, 'numeric_metadata_closure.csv'));

summary = struct();
summary.round = 200;
summary.matlab_release = version('-release');
summary.range_read = 'A1:PH14';
summary.raw_size = size(raw);
summary.num_size = size(num);
summary.text_size = size(text);
summary.proxy_count = numel(lat);
summary.numeric_max_abs_error = max(numeric_error);
summary.id_match_count = sum(id_match);
summary.proxy_type_match_count = sum(type_match);
summary.full_metadata_closure = ...
    summary.numeric_max_abs_error <= 1e-12 && ...
    summary.id_match_count == numel(ids) && ...
    summary.proxy_type_match_count == numel(proxy_type);
summary.annual_rows_15_onward_read = false;
summary.locked_period_values_accessed = false;
summary.jionglacuo_accessed = false;
if summary.full_metadata_closure
    summary.decision = 'deposited_xlsread_metadata_indices_verified';
else
    summary.decision = 'deposited_xlsread_metadata_indices_conflict';
end

fid = fopen(fullfile(output_dir, 'summary.json'), 'w');
if fid < 0
    error('Could not open summary.json for writing');
end
fprintf(fid, '%s\n', jsonencode(summary, 'PrettyPrint', true));
fclose(fid);

fid = fopen(fullfile(output_dir, 'FINAL_RESULT.md'), 'w');
if fid < 0
    error('Could not open FINAL_RESULT.md for writing');
end
fprintf(fid, '# Round 200 MATLAB xlsread metadata closure\n\n');
fprintf(fid, '- MATLAB release: `%s`.\n', summary.matlab_release);
fprintf(fid, '- Range read: `%s` only.\n', summary.range_read);
fprintf(fid, '- Proxy count: `%d`.\n', summary.proxy_count);
fprintf(fid, '- Numeric maximum absolute error: `%.16g`.\n', summary.numeric_max_abs_error);
fprintf(fid, '- ID / proxy-type matches: `%d/%d` / `%d/%d`.\n', ...
    summary.id_match_count, numel(ids), summary.proxy_type_match_count, numel(proxy_type));
fprintf(fid, '- Decision: `%s`.\n', summary.decision);
fprintf(fid, '- Worksheet rows 15 onward and all annual values were not read.\n');
fclose(fid);

save(fullfile(output_dir, 'metadata_closure.mat'), ...
    'lat', 'lon', 'elev', 'first_year', 'last_year', 'ids', 'proxy_type', 'summary');
disp(jsonencode(summary, 'PrettyPrint', true));
end
