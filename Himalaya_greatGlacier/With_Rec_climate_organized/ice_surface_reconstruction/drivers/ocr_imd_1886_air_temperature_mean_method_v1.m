function ocr_imd_1886_air_temperature_mean_method_v1()
root = "E:\3spyderProject\1_LIA_himalaya_oggm\validation_data\results\paleo_surface_validation_v0\climate_reconstruction_improvement_loop_v2\imd_1886_air_temperature_mean_method_v1";
for leaf = [31, 32, 33]
    inputPath = fullfile(root, compose("official_leaf_%04d.jpg", leaf));
    outputPath = fullfile(root, compose("matlab_ocr_leaf_%04d", leaf));
    ocr_imd_temperature_table_page_pilot_v2(inputPath, outputPath);
end
end
