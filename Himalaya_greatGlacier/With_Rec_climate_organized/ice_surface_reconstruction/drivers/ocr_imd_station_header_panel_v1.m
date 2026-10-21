function ocr_imd_station_header_panel_v1()
root = "E:\3spyderProject\1_LIA_himalaya_oggm\validation_data\results\paleo_surface_validation_v0\climate_reconstruction_improvement_loop_v2\imd_station_header_panel_v1";
imageDir = fullfile(root, "header_images");
outputRoot = fullfile(root, "matlab_ocr");
if isfolder(outputRoot)
    error("Output directory already exists: %s", outputRoot);
end
mkdir(outputRoot);

files = dir(fullfile(imageDir, "*.png"));
completed = 0;
failed = strings(0, 2);
for index = 1:numel(files)
    inputPath = fullfile(files(index).folder, files(index).name);
    [~, stem, ~] = fileparts(files(index).name);
    outputPath = fullfile(outputRoot, stem);
    try
        ocr_imd_temperature_table_page_pilot_v2(inputPath, outputPath);
        completed = completed + 1;
    catch exception
        failed(end + 1, :) = [string(files(index).name), string(exception.message)]; %#ok<AGROW>
    end
end

if ~isempty(failed)
    writetable(array2table(failed, VariableNames=["image", "error"]), ...
        fullfile(root, "matlab_ocr_failures.csv"));
end
summary = struct();
summary.experiment = "imd_station_header_matlab_ocr_v1";
summary.requested_images = numel(files);
summary.completed_images = completed;
summary.failed_images = size(failed, 1);
summary.locked_period_accessed = false;
summary.source_files_modified = false;
fid = fopen(fullfile(root, "matlab_ocr_summary.json"), "w", "n", "UTF-8");
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, "%s\n", jsonencode(summary, PrettyPrint=true));
disp(jsonencode(summary, PrettyPrint=true));
end
