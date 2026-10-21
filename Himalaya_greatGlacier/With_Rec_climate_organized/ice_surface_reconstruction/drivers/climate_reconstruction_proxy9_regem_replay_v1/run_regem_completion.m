function run_regem_completion(input_path, output_path, regem_source)
% Run the archived RegEM implementation on an independently extracted matrix.
addpath(regem_source);
S = load(input_path, 'R0v');
R0v_filled = regem(S.R0v);
save(output_path, 'R0v_filled', '-v7');
disp('REGEM_COMPLETE');
end
