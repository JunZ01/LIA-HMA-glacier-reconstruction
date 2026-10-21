function ocr_imd_darjeeling_header_preprocessed_v1()
root = "E:\3spyderProject\1_LIA_himalaya_oggm\validation_data\results\paleo_surface_validation_v0\climate_reconstruction_improvement_loop_v2";
inputDir = fullfile(root, "imd_station_header_panel_v1", "header_images");
outputDir = fullfile(root, "imd_darjeeling_header_preprocessed_ocr_v1");
if isfolder(outputDir)
    error("Output directory already exists: %s", outputDir);
end
mkdir(outputDir);

files = dir(fullfile(inputDir, "*darjeeling*clockwise.png"));
sensitivities = [0.35, 0.50, 0.65];
records = table();
for fileIndex = 1:numel(files)
    sourcePath = fullfile(files(fileIndex).folder, files(fileIndex).name);
    source = imread(sourcePath);
    if size(source, 3) == 3
        gray = im2gray(source);
    else
        gray = source;
    end
    [~, stem, ~] = fileparts(files(fileIndex).name);
    for sensitivity = sensitivities
        foreground = imbinarize(gray, "adaptive", ...
            ForegroundPolarity="dark", Sensitivity=sensitivity);
        longHorizontal = imopen(foreground, strel("line", 90, 0));
        longVertical = imopen(foreground, strel("line", 90, 90));
        cleaned = foreground & ~(longHorizontal | longVertical);
        cleaned = bwareaopen(cleaned, 8);
        prepared = uint8(~cleaned) .* uint8(255);

        tag = stem + "_s" + replace(compose("%.2f", sensitivity), ".", "p");
        imagePath = fullfile(outputDir, tag + ".png");
        imwrite(prepared, imagePath);
        result = ocr(prepared);
        textPath = fullfile(outputDir, tag + ".txt");
        fid = fopen(textPath, "w", "n", "UTF-8");
        fprintf(fid, "%s", result.Text);
        fclose(fid);

        words = string(result.Words(:));
        boxes = double(result.WordBoundingBoxes);
        confidence = double(result.WordConfidences(:));
        wordTable = table(words, boxes(:, 1), boxes(:, 2), boxes(:, 3), ...
            boxes(:, 4), confidence, VariableNames= ...
            ["word", "x", "y", "width", "height", "confidence"]);
        writetable(wordTable, fullfile(outputDir, tag + "_words.csv"));

        record = table(string(files(fileIndex).name), sensitivity, ...
            height(wordTable), mean(confidence, "omitnan"), ...
            VariableNames=["source_image", "sensitivity", "word_count", ...
            "mean_confidence"]);
        records = [records; record]; %#ok<AGROW>
    end
end
writetable(records, fullfile(outputDir, "ocr_manifest.csv"));
summary = struct();
summary.experiment = "imd_darjeeling_header_preprocessed_ocr_v1";
summary.source_pages = numel(files);
summary.preprocessing_variants = height(records);
summary.locked_period_accessed = false;
summary.source_files_modified = false;
fid = fopen(fullfile(outputDir, "summary.json"), "w", "n", "UTF-8");
fprintf(fid, "%s\n", jsonencode(summary, PrettyPrint=true));
fclose(fid);
disp(jsonencode(summary, PrettyPrint=true));
end
