function run_regem_completion(input_path, output_path, regem_source)
% Complete only the newly restored missing residual entries with archived RegEM.
addpath(regem_source);
S = load(input_path, 'R0v');
R0v_filled = regem(S.R0v);
save(output_path, 'R0v_filled', '-v7');
disp('RESIDUAL_MISSINGNESS_REGEM_COMPLETE');
end
