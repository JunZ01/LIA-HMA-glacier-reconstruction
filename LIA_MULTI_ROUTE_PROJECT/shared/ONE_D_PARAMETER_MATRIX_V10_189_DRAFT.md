# One-dimensional parameter matrix v10.189 draft

Status: proposed, not launch-authorized

Model family: `oggm-flowline-1d`

Primary objective: extend the comparable one-dimensional curve population from
the canonical 58 accepted systems toward approximately 100 accepted systems.
This is the production route, not a planform-2D experiment. Any earlier
`GPU_SPINUP_BENCHMARK_V10_189_DRAFT.md` is withdrawn and has no authority.

## Matrix axes

1. `delta_t_eq`: `[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]` degC.
2. `glen_a`: inherited baseline times `[0.5, 1.0, 2.0, 3.0]`.
3. `prcp_fac` and `melt_f`: inherited baselines times
   `[0.8, 1.0, 1.2] x [0.8, 1.0, 1.2]`.
4. `temp_melt`: `[-1.5, -1.0, -0.5]` degC.

`base_temp_bias` remains inherited provenance. Effective equilibrium temperature
bias is `base_temp_bias + delta_t_eq`. Length is an output, not a spin-up target.

## Adaptive spin-up and expansion

- Compute and cache each unique glacier-parameter fingerprint once.
- Continue in 300-year blocks to a hard ceiling of 3000 years.
- At each boundary, evaluate the final 90 years by same-phase comparisons across
  the repeated 30-year climate cycle.
- Begin population-wide with the seven `delta_t_eq` candidates at inherited
  baseline values for the other blocks.
- Freeze converged states. Open `glen_a` only for unresolved systems, then the
  joint mass-balance block, then `temp_melt`.
- Do not run an unconditional 756-cell Cartesian search per glacier.

## Forward reuse

Store the final state, equilibrium length, area, volume, parameters, input hashes,
and convergence evidence. Compatible forward candidates reuse that exact state.
Forward outputs retain annual length, area, volume, and mass-balance series.
Historical length and hard gates apply in forward/reproduction, not in spin-up.

## Parallel execution

Parallelize independent glacier-parameter chains with CPU worker processes. The
existing OGGM flowline implementation remains the production solver. This draft
introduces no GPU port and no planform-2D adapter. Select the worker count using
a short exact-contract runtime and memory preflight.

## Authorization boundary

This draft grants no launch authority. Exact convergence thresholds, confirmation
count, member ledger, worker count, output root, and a one-chain preflight must be
registered in a canonical transaction and explicitly approved before launch.
