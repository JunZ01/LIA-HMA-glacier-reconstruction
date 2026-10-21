try
    driver = 'E:/3spyderProject/1_LIA_himalaya_oggm/Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/climate_reconstruction_proxy9_nested_error_weight_v2';
    output_dir = 'E:/3spyderProject/1_LIA_himalaya_oggm/validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_nested_error_weight_v2';
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    addpath(driver);
    run_design_audit( ...
        fullfile(output_dir, 'design_rank_audit.mat'), ...
        'D:/Users/JUNZ/MATLAB/Projects/untitled/PHYDA-v1-master', ...
        'D:/Users/JUNZ/Desktop/文件整理/1 论文/work1/compared_result_all/python_data/proxy_data_9.mat');
catch ME
    disp(getReport(ME, 'extended', 'hyperlinks', 'off'));
    exit(1);
end
