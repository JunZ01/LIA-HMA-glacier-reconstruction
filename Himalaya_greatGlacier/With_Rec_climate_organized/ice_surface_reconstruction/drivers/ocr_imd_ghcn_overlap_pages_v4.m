function ocr_imd_ghcn_overlap_pages_v4(imageRoot, outputDir)
arguments
    imageRoot (1, 1) string
    outputDir (1, 1) string
end

if isfolder(outputDir)
    error("Output directory already exists: %s", outputDir);
end
mkdir(outputDir);
pagesDir = fullfile(outputDir, "pages");
mkdir(pagesDir);

files = dir(fullfile(imageRoot, "**", "leaf_*.jpg"));
if isempty(files)
    error("No page images found under: %s", imageRoot);
end
statusRows = table();
warning("off", "all");

for index = 1:numel(files)
    imagePath = string(fullfile(files(index).folder, files(index).name));
    itemMatch = regexp( ...
        files(index).folder, 'india\.history\.resource\.\d+$', 'match', 'once');
    if isempty(itemMatch)
        error("Cannot parse item identifier from: %s", files(index).folder);
    end
    itemId = string(itemMatch);
    token = regexp(files(index).name, 'leaf_(\d+)\.jpg', 'tokens', 'once');
    leaf = str2double(token{1});
    itemSuffix = extractAfter(itemId, "resource.");
    pageKey = itemSuffix + "_" + compose("%04d", leaf);
    pageDir = fullfile(pagesDir, pageKey);
    mkdir(pageDir);
    started = tic;
    state = "completed";
    message = "";
    wordCount = 0;
    yearWordCount = 0;
    meanConfidence = NaN;
    try
        imageData = imread(imagePath);
        result = ocr(imageData);
        words = string(result.Words(:));
        boxes = double(result.WordBoundingBoxes);
        confidences = double(result.WordConfidences(:));
        wordTable = table( ...
            words, boxes(:, 1), boxes(:, 2), boxes(:, 3), boxes(:, 4), ...
            confidences, ...
            'VariableNames', {'word', 'x', 'y', 'width', 'height', 'confidence'});
        writetable(wordTable, fullfile(pageDir, "words.csv"));
        textFile = fopen(fullfile(pageDir, "text.txt"), "w", "n", "UTF-8");
        fprintf(textFile, "%s", result.Text);
        fclose(textFile);
        wordCount = height(wordTable);
        yearWordCount = sum(contains(lower(words), "year"));
        meanConfidence = mean(confidences, "omitnan");
    catch errorInfo
        state = "failed";
        message = string(errorInfo.message);
    end
    elapsedSeconds = toc(started);
    current = table( ...
        itemId, leaf, imagePath, pageKey, state, message, wordCount, ...
        yearWordCount, meanConfidence, elapsedSeconds, ...
        'VariableNames', { ...
            'item_id', 'leaf', 'image_path', 'page_key', 'status', 'message', ...
            'word_count', 'year_word_count', 'mean_word_confidence', ...
            'elapsed_seconds'});
    statusRows = [statusRows; current]; %#ok<AGROW>
    writetable(statusRows, fullfile(outputDir, "page_ocr_status.csv"));
end

pageDirectoryCount = numel(dir(fullfile(pagesDir, "*_*")));
summary = struct();
summary.experiment = "imd_ghcn_overlap_matlab_ocr_v4";
summary.supersedes = [ ...
    "v1_failed_before_page_ocr_due_to_csv_variable_name_detection", ...
    "v2_failed_before_page_ocr_due_to_csv_variable_name_preservation", ...
    "v3_ocr_completed_but_page_outputs_overwrote_due_to_dotted_folder_parse"];
summary.input_method = "direct_recursive_jpeg_enumeration_without_csv_or_java";
summary.item_identifier_method = "terminal_regular_expression";
summary.requested_pages = numel(files);
summary.completed_pages = sum(statusRows.status == "completed");
summary.failed_pages = sum(statusRows.status == "failed");
summary.unique_page_output_directories = pageDirectoryCount;
summary.total_words = sum(statusRows.word_count);
summary.locked_period_values_read = false;
summary.annual_values_promoted = false;
summary.decision = "advance_coordinate_value_extraction";
fid = fopen(fullfile(outputDir, "summary.json"), "w", "n", "UTF-8");
fprintf(fid, "%s\n", jsonencode(summary, PrettyPrint=true));
fclose(fid);

contract = [ ...
    "# IMD-GHCN Overlap MATLAB OCR Contract v4" newline newline ...
    "- Parse dotted item folders by terminal regular expression." newline ...
    "- Store every page in a distinct item-leaf output directory." newline ...
    "- Enumerate only JPEG pages already frozen in the v2 overlap panel." newline ...
    "- Preserve every word coordinate and confidence by page." newline ...
    "- Do not accept or compare a numeric value in this OCR-only stage." newline ...
    "- Do not access 1921-1939, production data, EnKF, or OGGM." newline];
fid = fopen(fullfile(outputDir, "EXPERIMENT_CONTRACT.md"), "w", "n", "UTF-8");
fprintf(fid, "%s", contract);
fclose(fid);

disp(jsonencode(summary, PrettyPrint=true));
end
