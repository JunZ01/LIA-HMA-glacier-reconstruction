# LIA Multi-Route Project

This directory is the coordination layer for the next LIA phase. It does not
replace the canonical project contracts, authorize execution, or duplicate
simulation data.

Authority remains, in order:

1. `../LIA_PROJECT_FACTS.md`
2. `../LIA_EXECUTION_CONTRACT.md`
3. `../LIA_CURRENT_STATE.json`

## Objective

Build a reusable spin-up state library and separate forward-forcing matrix
while allowing multiple model routes to be investigated without mixing their
evidence or scientific claims.

## Shared Contracts

- `shared/SPINUP_MATRIX_DRAFT.md`: prospective state-generation contract.
- `shared/FORWARD_MATRIX_DRAFT.md`: prospective state-by-forcing evaluation
  contract.
- `shared/DECISIONS_AND_STOP_RULES.md`: explicit status of each decision and
  the stop rules that apply before, during, and after any future transaction.
- `shared/GPU_SPINUP_BENCHMARK_V1.md` and
  `shared/gpu_spinup_benchmark_v1.json`: fixed-member, fixed-physics proposal
  for the next minimal GPU benchmark transaction.

All three are organizational drafts. They become executable only when an
explicit user-approved, versioned canonical transaction records exact values,
fingerprints, evidence roots, and authorization.

## Route Separation

- `control`: owns shared definitions, route registration, and cross-route
  decisions. It launches no simulations.
- `one_d_production`: extends the curve population with the OGGM flowline
  family used by the existing accepted curves.
- `two_d_gpu_validation`: evaluates the modified planform-2D GPU family on a
  representative subset. Its results are not pooled silently with 1-D curves.
- `curve_analysis`: compares accepted, excluded-shape, failed, and new curves.

Each route has its own task, evidence namespace, parameter contract, state, and
stop rule. A result from one route cannot change another route's accepted count.
State identifiers always include `model_family`; a 1-D state is never a 2-D
initial condition, reproduction, or acceptance result.

## Current Launch Boundary

The canonical state is `v10.188`, validator status is `pass`, and launch
authorization is false. No 50-year, adaptive spin-up, historical, or parameter
matrix simulation is authorized by this scaffold.

## Proposed Spin-Up Library

For a glacier and parameter vector, spin-up produces a state rather than
matching a prescribed length:

`S(theta) = {thickness field, equilibrium length, area, volume, provenance}`

The current design proposal is:

- resume in 300-year blocks;
- inspect the last 90 years at identical phases of the repeated 30-year
  climate window;
- allow early stopping after registered convergence confirmation;
- use 3000 years as the maximum, not the mandatory duration;
- retain every technically valid converged length as a state coordinate;
- apply observation-length and curve-shape evaluation only after forward runs.

The exact thresholds and confirmation rule remain proposals until frozen in a
versioned canonical route transaction.

## Transaction Boundary

The organizational layer may clarify vocabulary, proposals, evidence
separation, and pending decisions. It must not edit `LIA_CURRENT_STATE.json`,
append canonical facts, change counts, mark a candidate accepted, or set
`launch_authorized=true`. Those actions require the complete transaction in
`../LIA_EXECUTION_CONTRACT.md` and explicit user approval.
