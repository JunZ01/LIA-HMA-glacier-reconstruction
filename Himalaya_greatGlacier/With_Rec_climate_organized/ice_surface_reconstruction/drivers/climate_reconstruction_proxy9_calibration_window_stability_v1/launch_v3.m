try
    driver = 'E:/3spyderProject/1_LIA_himalaya_oggm/Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/climate_reconstruction_proxy9_calibration_window_stability_v1';
    addpath(driver);
    run_calibration_windows_v3( ...
        'E:/3spyderProject/1_LIA_himalaya_oggm/validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_calibration_window_stability_v1/calibration_windows_v3.mat', ...
        'D:/Users/JUNZ/MATLAB/Projects/untitled/PHYDA-v1-master', ...
        'D:/Users/JUNZ/Desktop/文件整理/1 论文/work1/compared_result_all/python_data/proxy_data_9.mat');
catch ME
    disp(getReport(ME, 'extended', 'hyperlinks', 'off'));
    exit(1);
end
