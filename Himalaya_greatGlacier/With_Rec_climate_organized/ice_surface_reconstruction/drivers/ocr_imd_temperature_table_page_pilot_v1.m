function ocr_imd_temperature_table_page_pilot_v1(imagePath, outputDir)
arguments
    imagePath (1, 1) string
    outputDir (1, 1) string
end

if isfolder(outputDir)
    error("Output directory already exists: %s", outputDir);
end
mkdir(outputDir);

sourceInfo = dir(imagePath);
if isempty(sourceInfo)
    error("Source image does not exist: %s", imagePath);
end

imageData = imread(imagePath);
result = ocr(imageData);

words = string(result.Words(:));
boxes = double(result.WordBoundingBoxes);
confidences = double(result.WordConfidences(:));
wordTable = table(
    words, boxes(:, 1), boxes(:, 2), boxes(:, 3), boxes(:, 4), confidences, ...
    'VariableNames', {'word', 'x', 'y', 'width', 'height', 'confidence'});
writetable(wordTable, fullfile(outputDir, "matlab_ocr_words.csv"));

textFile = fopen(fullfile(outputDir, "matlab_ocr_text.txt"), "w", "n", "UTF-8");
cleanupText = onCleanup(@() fclose(textFile));
fprintf(textFile, "%s", result.Text);

yearRows = wordTable(contains(lower(wordTable.word), "year"), :);
writetable(yearRows, fullfile(outputDir, "year_word_candidates.csv"));

summary = struct();
summary.experiment = "temperature_historical_imd_matlab_ocr_pilot_v1";
summary.source_image = imagePath;
summary.source_bytes = sourceInfo.bytes;
summary.image_height = size(imageData, 1);
summary.image_width = size(imageData, 2);
summary.word_count = height(wordTable);
summary.year_word_count = height(yearRows);
summary.mean_word_confidence = mean(confidences, "omitnan");
summary.locked_period_values_read = false;
summary.decision = "inspect_word_geometry_before_batch_ocr";
fid = fopen(fullfile(outputDir, "summary.json"), "w", "n", "UTF-8");
cleanupSummary = onCleanup(@() fclose(fid));
fprintf(fid, "%s\n", jsonencode(summary, PrettyPrint=true));

contract = [
    "# MATLAB OCR Pilot Contract" newline newline ...
    "- Read one 1886 IMD air-temperature table page only." newline ...
    "- Preserve word coordinates and confidences; do not promote values." newline ...
    "- Use MATLAB R2024b OCR as a second engine independent of IA OCR." newline ...
    "- Do not access 1921-1939, production data, EnKF, or OGGM." newline];
fidContract = fopen(fullfile(outputDir, "EXPERIMENT_CONTRACT.md"), "w", "n", "UTF-8");
cleanupContract = onCleanup(@() fclose(fidContract));
fprintf(fidContract, "%s", contract);

disp(jsonencode(summary, PrettyPrint=true));
end
