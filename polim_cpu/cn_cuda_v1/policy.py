"""Numerical policy fixed before operator evaluation; no result-dependent tolerance."""
VERSION='cn-cuda-v48-1'
CONSISTENCY=dict(residual_force_scaled=1e-10,force_relative=1e-11,Fu_relative=1e-9,FH_relative=1e-8,
    flux_relative=1e-11,Ru_relative=1e-10,dry_absolute=0.,near_zero_absolute=1e-10,finite_difference_relative=2e-5)
SOLVER=dict(BP_tolerance=1e-6,BP_Newton=80,BP_CG=500,BP_line=35,full_BP_per_step=8,
    tangent_per_step=64,tangent_CG=200,tangent_scaled_tolerance=1e-10,tangent_true_tolerance=1e-8,
    outer=8,GMRES_restart=8,GMRES_cycles=1,GMRES_rtol=.02,GMRES_true=.2,outer_line=8,
    fixed_u_Newton=20,fixed_u_line=8,limiter_rounds=6,step_seconds=300,batch_seconds=3600,
    rebuild_wet_fraction=.005,rebuild_iteration_trigger=96,graph_batch=4)
