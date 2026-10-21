from polim_cpu.cn_cuda_v1.policy import SOLVER as BASE_SOLVER
VERSION='cn-sparse-v49-1'
SOLVER=dict(BASE_SOLVER)
ASSEMBLY=dict(matrix_relative=1e-13,matrix_absolute=1e-13,action_relative=1e-12,action_absolute=1e-12,
    solve_relative=1e-10,solve_absolute=1e-10,LU_residual=1e-11,near_zero_norm=1.,operator_relative=1e-9)
SELECTION=dict(candidate_min_saving=.02,case_regression_limit=.10,required_combined_saving=0.)

