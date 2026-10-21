driver_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(fileparts(fileparts(driver_dir))));
output_dir = fullfile(project_root, 'validation_data', 'results', ...
    'paleo_surface_validation_v0', ...
    'climate_reconstruction_precipitation_prior_recovery_preflight_v1');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

phyda_root = 'D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master';
archive_folder = char([25991 20214 25972 29702]);
paper_folder = ['1 ' char([35770 25991])];
compact_prior_path = fullfile('D:\Users\JUNZ\Desktop', archive_folder, ...
    paper_folder, 'work1', 'compared_result_all', 'python_data', 'prior_fields.mat');
output_path = fullfile(output_dir, 'precipitation_prior_336x848_mm_day.mat');

run_precipitation_prior_preflight(output_path, phyda_root, compact_prior_path);

