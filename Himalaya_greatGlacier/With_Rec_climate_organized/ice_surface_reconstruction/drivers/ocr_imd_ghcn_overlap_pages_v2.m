function ocr_imd_ghcn_overlap_pages_v2(manifestPath, outputDir)
arguments
    manifestPath (1, 1) string
    outputDir (1, 1) string
end

if isfolder(outputDir)
    error("Output directory already exists: %s", outputDir);
end
mkdir(outputDir);
pagesDir = fullfile(outputDir, "pages");
mkdir(pagesDir);

manifest = readtable( ...
    manifestPath, TextType="string", VariableNamingRule="preserve");
statusRows = table();
warning("off", "all");

for index = 1:height(manifest)
    itemId = manifest.("item_id")(index);
    leaf = manifest.("leaf")(index);
    imagePath = manifest.("image_path")(index);
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

summary = struct();
summary.experiment = "imd_ghcn_overlap_matlab_ocr_v2";
summary.supersedes = "imd_ghcn_overlap_matlab_ocr_v1_failed_before_page_ocr";
summary.variable_name_preservation_fix = true;
summary.requested_pages = height(manifest);
summary.completed_pages = sum(statusRows.status == "completed");
summary.failed_pages = sum(statusRows.status == "failed");
summary.total_words = sum(statusRows.word_count);
summary.locked_period_values_read = false;
summary.annual_values_promoted = false;
summary.decision = "advance_coordinate_value_extraction";
fid = fopen(fullfile(outputDir, "summary.json"), "w", "n", "UTF-8");
fprintf(fid, "%s\n", jsonencode(summary, PrettyPrint=true));
fclose(fid);

contract = [ ...
    "# IMD-GHCN Overlap MATLAB OCR Contract v2" newline newline ...
    "- Preserve manifest variable names; no scientific input changed from v1." newline ...
    "- OCR only the 1878-1890 full-resolution pages in the frozen v2 panel." newline ...
    "- Preserve every word coordinate and confidence by page." newline ...
    "- Do not accept or compare a numeric value in this OCR-only stage." newline ...
    "- Do not access 1921-1939, production data, EnKF, or OGGM." newline];
fid = fopen(fullfile(outputDir, "EXPERIMENT_CONTRACT.md"), "w", "n", "UTF-8");
fprintf(fid, "%s", contract);
fclose(fid);

disp(jsonencode(summary, PrettyPrint=true));
end
