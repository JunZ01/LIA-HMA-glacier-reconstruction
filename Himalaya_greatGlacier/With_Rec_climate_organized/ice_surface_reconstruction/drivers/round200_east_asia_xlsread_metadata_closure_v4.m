function round200_east_asia_xlsread_metadata_closure_v4(synthetic_path, v3_csv, output_dir)
% Apply deposited xlsread expressions to a value-free synthetic workbook.

if ~exist(output_dir, 'dir')
    error('Prepared independent output directory is missing: %s', output_dir);
end
if exist(fullfile(output_dir, 'summary.json'), 'file')
    error('Refusing to overwrite completed result: %s', output_dir);
end

[num, text, raw] = xlsread(synthetic_path, 1); %#ok<XLSRD>
expected = readtable(v3_csv, 'VariableNamingRule', 'preserve');
if height(expected) ~= 423
    error('Expected 423 Round 199 metadata records, found %d', height(expected));
end

lat = num(5, 2:end)';
lon = num(6, 2:end)';
elev = num(7, 2:end)';
first_year = num(9, 2:end)';
last_year = num(10, 2:end)';
proxy_years_expression = num(15:end, 1);
ids = string(text(1, 2:end))';
proxy_type = string(text(12, 2:end))';

record_count_match = numel(lat) == height(expected);
numeric_error = [Inf Inf Inf Inf Inf];
if record_count_match
    numeric_error = [
        max(abs(lat - expected.latitude), [], 'omitnan'), ...
        max(abs(lon - expected.longitude), [], 'omitnan'), ...
        max(abs(elev - expected.elevation_m), [], 'omitnan'), ...
        max(abs(first_year - expected.first_year), [], 'omitnan'), ...
        max(abs(last_year - expected.last_year), [], 'omitnan')];
end
id_match_count = 0;
if numel(ids) == height(expected)
    id_match_count = sum(strcmp(ids, string(expected.generic_id)));
end
type_match_count = 0;
if numel(proxy_type) == height(expected)
    type_match_count = sum(strcmp(proxy_type, string(expected.archive_type)));
end

closure = table( ...
    ["latitude"; "longitude"; "elevation_m"; "first_year"; "last_year"], ...
    numeric_error', ...
    'VariableNames', {'field', 'max_abs_error'});
writetable(closure, fullfile(output_dir, 'numeric_metadata_closure.csv'));

summary = struct();
summary.round = 200;
summary.version = 4;
summary.matlab_release = version('-release');
summary.source_rows_decoded = '1-14 only';
summary.synthetic_annual_content = 'column A year labels 1-2019; all proxy cells blank';
summary.raw_size = size(raw);
summary.num_size = size(num);
summary.text_size = size(text);
summary.deposited_metadata_record_count = numel(lat);
summary.expected_metadata_record_count = height(expected);
summary.record_count_match = record_count_match;
summary.numeric_max_abs_error = max(numeric_error);
summary.id_match_count = id_match_count;
summary.proxy_type_match_count = type_match_count;
summary.proxy_year_expression_count = numel(proxy_years_expression);
summary.proxy_year_expression_first = proxy_years_expression(1);
summary.proxy_year_expression_last = proxy_years_expression(end);
summary.proxy_year_expression_matches_1_2019 = isequal(proxy_years_expression, (1:2019)');
summary.full_metadata_closure = ...
    record_count_match && summary.numeric_max_abs_error <= 1e-12 && ...
    id_match_count == height(expected) && type_match_count == height(expected);
summary.original_annual_proxy_values_read = false;
summary.locked_period_values_accessed = false;
summary.jionglacuo_accessed = false;
if summary.full_metadata_closure && summary.proxy_year_expression_matches_1_2019
    summary.decision = 'deposited_xlsread_metadata_and_year_indices_verified';
elseif summary.full_metadata_closure
    summary.decision = 'metadata_verified_but_deposited_year_expression_shifted';
else
    summary.decision = 'deposited_xlsread_metadata_indices_conflict';
end

fid = fopen(fullfile(output_dir, 'summary.json'), 'w');
fprintf(fid, '%s\n', jsonencode(summary, 'PrettyPrint', true));
fclose(fid);

fid = fopen(fullfile(output_dir, 'FINAL_RESULT.md'), 'w');
fprintf(fid, '# Round 200 V4 synthetic xlsread closure\n\n');
fprintf(fid, '- MATLAB release: `%s`.\n', summary.matlab_release);
fprintf(fid, '- Numeric/text/raw sizes: `%s` / `%s` / `%s`.\n', ...
    mat2str(summary.num_size), mat2str(summary.text_size), mat2str(summary.raw_size));
fprintf(fid, '- Deposited/expected metadata records: `%d/%d`.\n', ...
    summary.deposited_metadata_record_count, summary.expected_metadata_record_count);
fprintf(fid, '- Numeric maximum absolute error: `%.16g`.\n', summary.numeric_max_abs_error);
fprintf(fid, '- ID/type matches: `%d/%d` / `%d/%d`.\n', ...
    summary.id_match_count, height(expected), summary.proxy_type_match_count, height(expected));
fprintf(fid, '- `num(15:end,1)` count/first/last: `%d/%g/%g`.\n', ...
    summary.proxy_year_expression_count, summary.proxy_year_expression_first, summary.proxy_year_expression_last);
fprintf(fid, '- Decision: `%s`.\n', summary.decision);
fprintf(fid, '- Original annual proxy values, locked values, and Jionglacuo were not read.\n');
fclose(fid);

save(fullfile(output_dir, 'synthetic_xlsread_closure.mat'), ...
    'lat', 'lon', 'elev', 'first_year', 'last_year', 'ids', 'proxy_type', ...
    'proxy_years_expression', 'summary');
disp(jsonencode(summary, 'PrettyPrint', true));
end
