# Stage 4 LIA initial-state forward validation

**Status:** v0.1 execution specification; no climate-scenario production run yet  
**Date:** 2026-07-24  
**Scope:** use the v0.1 reconstructed LIA ice-thickness initial states in the OGGM transient validation workflow

## 1. Purpose

This document defines the next workflow step after the v0.1 LIA ice-thickness
reconstruction and OGGM initial-state adapter.

The v0.1 reconstruction has already produced OGGM-initializable LIA flowline
states and passed a short zero-mass-balance dynamic smoke test. That result
closes the **initial geometry** problem only. It does not replace the OGGM
modern calibration workflow and it does not authorize direct multi-scenario
climate production.

Stage 4 tests whether the reconstructed LIA initial geometry, when combined
with the frozen Stage 2 mass-balance and rheology parameters and driven by the
registered reconstructed climate, evolves toward the modern RGI state without
rewriting any calibrated parameter.

In short:

```text
Stage 1/2: modern calibration and parameter freeze
Stage 3: LIA initial state
Stage 4: forward transient validation from LIA to modern
```

The new v0.1 LIA ice-thickness package is a Stage 3 initial-state artifact. It
enters Stage 4 as a read-only restart state.

## 2. Non-negotiable boundaries

1. Stage 4 may read Stage 2 parameters, reconstructed climate, modern RGI
   geometry, LIA timing metadata, and v0.1 initial states.
2. Stage 4 may not write or modify `glen_a`, `fs`, `melt_f`, `prcp_fac`,
   `temp_bias`, or any Stage 1/2 posterior artifact.
3. Stage 4 terminal residuals may diagnose failures but may not feed back into
   Stage 1/2 calibration.
4. Reconstructed LIA ice thickness is not regenerated per climate member. The
   v0.1 3D thickness field is the initial condition.
5. Dynamic relaxation and forward transient residuals must remain separate
   diagnostics. A successful zero-MB smoke test is not a successful transient
   climate validation.
6. Review categories from the v0.1 initial-state release must be preserved.
   `dynamic-relaxation-review`, `width-remap-review`, and `input-missing`
   members cannot silently enter the primary production set.

## 3. Required inputs

### 3.1 Frozen Stage 2 parameter artifacts

Primary directory:

```text
outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/stage2_freeze_stage3_routing_v0/
```

Required files:

- `stage3_routing_245.csv`
- `stage2_system_ensemble.csv`
- `stage2_member_ensemble.csv`
- `routing_summary.json`
- `run_manifest.json`

The Stage 2 gate has already declared:

- 245 routed LIA systems;
- 191 core systems;
- 573 system-ensemble rows;
- 2,850 member-ensemble rows;
- velocity selection weight equal to zero.

Stage 4 consumes these rows but never edits them.

### 3.2 v0.1 LIA initial-state release

Primary directory:

```text
../../validation_data/results/paleo_surface_validation_v0/initial_state_release_v01_core191/
```

Required files:

- `initial_state_v01_qc_routes.csv`
- `initial_state_v01_system_coverage.csv`
- `production_ready_initial_states_v01.csv`
- `production_ready_default_one_member_per_system_v01.csv`
- `initial_state_v01_qc_summary.json`

Default Stage 4 pilot input:

```text
production_ready_default_one_member_per_system_v01.csv
```

This manifest currently contains 187 strict production-ready systems. Four core
systems do not have a strict default member:

- `2342` / `RGI60-15.04541`: dynamic relaxation review only;
- `2790` / `RGI60-15.07501`: dynamic relaxation review only;
- `2906` / `RGI60-13.26439`: width remap review only;
- `3053` / `RGI60-15.09408`: dynamic relaxation review only.

These four systems are retained in the accounting ledger but excluded from the
primary Stage 4 pilot unless a review-specific experiment is explicitly
requested.

### 3.3 Stage 3 geometry and climate registries

Primary directory:

```text
outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/stage3_initial_state_pilot_v0/
```

Relevant files:

- `climate_registry_6.csv`
- `climate_pixel_registry_330.csv`
- `lia_geometry_targets_12.csv`
- `pilot_selection_12.csv`
- `pilot_stage2_system_ensemble.csv`
- `pilot_stage2_member_ensemble.csv`
- `run_manifest.json`

For the first v0.1 Stage 4 pilot, use only the reference reconstructed climate
member already used by the Stage 3 gate, unless a later run explicitly expands
to all six climate individuals.

### 3.4 Modern terminal validation data

Modern RGI geometry remains the terminal validation target. The terminal
metrics must be derived from registered Stage 0/Stage 2 routing artifacts, not
from the v0.1 LIA initial-state package.

Minimum terminal targets:

- modern RGI area;
- modern main-flowline length or system length proxy where registered;
- optionally modern hypsometry or elevation quantiles where the existing Stage
  3 target machinery already provides a consistent operator;
- model volume and geometry diagnostics for interpretation, not as primary
  observations unless registered.

## 4. Stage 4 forward operator

For each selected LIA system:

1. Load the v0.1 production-ready OGGM initial-state pickle from the selected
   manifest row.
2. Resolve the corresponding modern/LIA OGGM glacier directory or cloned
   working directory required by OGGM.
3. Load the frozen Stage 2 member parameters for the same LIA system and
   member RGI glacier(s).
4. Construct a `MonthlyTIModel` or existing project-specific MB model using
   the frozen `melt_f`, `prcp_fac`, `temp_bias`, precipitation policy, and
   climate file contract.
5. Drive the model with reconstructed climate from the LIA start/restart year
   to the modern validation year.
6. Write model geometry and diagnostics to a Stage 4 output directory with an
   immutable run tag.
7. Compare terminal geometry to modern RGI targets.

The forward run must not run a new constant-climate spinup to create the LIA
state. The v0.1 initial-state pickle is the starting state.

## 5. Start year, end year, and climate member policy

The first pilot uses a deliberately narrow policy:

- start year: use the representative LIA restart year already defined by the
  Stage 3 timing/config contract, or if absent, use the centre of the relevant
  regional LIA core window;
- end year: modern validation year used by the existing notebook/OGGM terminal
  comparison, expected to be around 2000 CE;
- climate members: reference reconstructed climate only;
- turnover roles: mid-turnover only for the first operator pilot;
- bed scenario: the default v0.1 manifest selection, currently preferring
  Farinotti2019 when production-ready.

Expansion order:

1. operator pilot: 5-10 systems, mid-turnover, reference climate;
2. 12-system pilot: the existing Stage 3 pilot set, mid-turnover, reference
   climate, limited to systems with v0.1 production-ready defaults;
3. turnover pilot: 12-system set, three Stage 2 turnover roles;
4. climate replication: 12-system set, all six climate individuals;
5. scale-up: 187 production-ready default systems;
6. optional structural ensemble: all 368 production-ready bed members.

No step advances until the previous step has produced classified outputs and
source hashes remain unchanged.

## 6. Pilot system selection

The first operator pilot should be small and deliberately heterogeneous. Use
only systems with `has-production-ready-default` in
`initial_state_v01_system_coverage.csv`.

Recommended selection criteria:

- one small/simple system;
- one large system;
- one multi-flowline system;
- at least one RGI13/RGI14/RGI15 representative where available;
- one system previously used in the Stage 3 pilot, so old and new initial-state
  routes can be compared.

Do not include the four no-default systems in the operator pilot:

- `2342`;
- `2790`;
- `2906`;
- `3053`.

They belong to a separate review experiment after the primary operator is
stable.

## 7. Output layout

Write the dry-run planner artifacts under:

```text
../../validation_data/results/paleo_surface_validation_v0/stage4_lia_v01_forward_validation_dryrun/
```

The later transient forward runner may use a dedicated Stage 4 work directory:

```text
outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/stage4_lia_v01_forward_validation/
```

Expected forward-run files:

- `run_manifest.json`
- `source_hashes_before.json`
- `source_hashes_after.json`
- `stage4_task_plan.csv`
- `stage4_forward_results.csv`
- `stage4_terminal_residuals.csv`
- `stage4_failures.csv`
- `stage4_summary.json`
- `logs/stage4_forward_validation.log`

Per-task model outputs should use immutable suffixes containing:

```text
system id + bed scenario + turnover role + climate id + run tag
```

No Stage 4 run may overwrite Stage 3 candidate NetCDF files or v0.1
initial-state pickles.

## 8. Minimum metrics

Each completed task writes one row with:

- `lia_unique_id`;
- `main_rgi_id`;
- `bed_scenario`;
- `turnover_role`;
- `climate_id`;
- `initial_state_pickle`;
- `start_year`;
- `end_year`;
- `initial_volume_km3`;
- `terminal_volume_km3`;
- `initial_area_km2`;
- `terminal_area_km2`;
- `modern_target_area_km2`;
- `terminal_area_ratio`;
- `terminal_area_log_error`;
- `initial_length_m`;
- `terminal_length_m`;
- `modern_target_length_m` if available;
- `terminal_length_ratio` if available;
- `finite_nonnegative`;
- `domain_censored`;
- `dynamic_model_status`;
- `terminal_validation_status`;
- `failure_class`;
- `output_geometry_path`;
- `diagnostics_path`.

Optional diagnostic fields:

- velocity residual class from Stage 2 routing;
- debris class/fraction;
- modern geodetic MB residual class;
- candidate dynamic relaxation class from the v0.1 initial-state release.

These optional fields are for interpretation only.

## 9. Acceptance rules

### 9.1 Operator pilot pass

The operator pilot passes if:

1. every selected task starts from the v0.1 initial-state pickle without
   rebuilding LIA geometry;
2. every task loads frozen Stage 2 MB/rheology parameters without rewriting
   Stage 2 artifacts;
3. every task writes classified success/failure rows;
4. no task fails due to missing path, missing climate file, missing MB
   parameter, or file overwrite conflict;
5. all source hashes are unchanged.

The operator pilot is allowed to produce bad terminal residuals. Bad terminal
residuals are scientific diagnostics, not operator failures.

### 9.2 Scientific pass

Scientific validation is evaluated separately from operator correctness.
Candidate indicators:

- terminal area within a registered tolerance of modern RGI area;
- terminal length within tolerance where length target is available;
- no negative or non-finite geometry;
- no domain-censored terminus unless classified;
- residual patterns are interpretable by frozen Stage 2 velocity/debris/MB
  diagnostics.

The first Stage 4 pilot should report these metrics without using them to tune
parameters.

## 10. Failure taxonomy

Minimum failure classes:

- `missing-initial-state`;
- `missing-stage2-parameter`;
- `missing-climate`;
- `mb-model-construction-failed`;
- `flux-model-initialization-failed`;
- `forward-run-failed`;
- `nonfinite-or-negative-geometry`;
- `domain-censored`;
- `terminal-area-miss`;
- `terminal-length-miss`;
- `review-category-excluded`;
- `source-hash-changed`;
- `unexpected-error`.

Terminal misses should remain valid Stage 4 outputs, not silent exclusions.

## 11. Relationship to the old Stage 3 spinup search

The existing Stage 3 search created LIA-like states by tuning an initial-state
temperature offset, then checking LIA geometry and timing. The v0.1
reconstruction changes the source of the initial geometry:

```text
old Stage 3 route:
modern/zero state -> climate-offset spinup -> selected LIA-like state

new v0.1 route:
reconstructed LIA 3D thickness -> OGGM initial-state pickle -> Stage 4 forward
```

The new route should first be evaluated as a separate Stage 4 branch. It does
not erase the old Stage 3 search; it provides an independent initial-state
candidate with clearer geometric meaning.

## 12. Immediate implementation plan

1. Build `stage4_task_plan.csv` from:
   - `production_ready_default_one_member_per_system_v01.csv`;
   - Stage 2 system/member ensemble rows;
   - reference reconstructed climate registry.
2. Implement a dry-run resolver that checks all paths and parameter joins
   without launching OGGM.
3. Run an operator pilot on 5-10 systems.
4. Inspect terminal residuals and failure taxonomy.
5. Only if the operator pilot is clean, run the 12-system Stage 3 pilot set
   using the v0.1 initial state route.

The next code task should therefore be a **Stage 4 dry-run planner**, not a
full transient production run.

## 13. Dry-run planner contract

First implementation target:

```text
src/stage4_forward_validation/build_v01_forward_task_plan.py
```

The planner performs joins and path checks only. It must not launch OGGM and
must not write to any glacier directory.

### 13.1 Default pilot join

Default filter:

```text
initial_state_qc_status = production-ready-initial-state
system_initial_state_status = has-production-ready-default
ensemble_role = mid-turnover
climate_id = ensemble_rec_ref
bed policy = default one-member-per-system manifest
```

Join keys:

| Left artifact | Right artifact | Join key | Notes |
|---|---|---|---|
| `production_ready_default_one_member_per_system_v01.csv` | `stage2_system_ensemble.csv` | `lia_unique_id` | select `ensemble_role = mid-turnover` for the first pilot |
| default manifest | `stage2_member_ensemble.csv` | `lia_unique_id` | member rows provide per-RGI MB parameters |
| task rows | `climate_registry_6.csv` | fixed `climate_id` | first pilot uses `ensemble_rec_ref` only |
| task rows | `stage3_routing_245.csv` | `lia_unique_id` | attach velocity/debris/routing diagnostics |
| task rows | `initial_state_v01_system_coverage.csv` | `lia_unique_id` | preserve release QC categories |

For the default v0.1 pilot, do not require a structural match between
`bed_scenario` and `thickness_target_product`; the manifest already selects the
default initial state. Structural matching becomes required only for a later
all-bed-member ensemble.

### 13.2 Planner outputs

Write under:

```text
../../validation_data/results/paleo_surface_validation_v0/stage4_lia_v01_forward_validation_dryrun/
```

Required dry-run outputs:

- `stage4_v01_task_plan_dryrun.csv`;
- `stage4_v01_task_plan_summary.json`;
- `stage4_v01_planner_failures.csv`;
- `source_hashes_dryrun.json`.

Minimum summary fields:

- total manifest systems;
- systems after QC filter;
- task count;
- missing initial-state pickle count;
- missing Stage 2 system parameter count;
- missing Stage 2 member parameter count;
- missing climate file count;
- missing routing row count;
- selected climate id;
- selected ensemble role;
- input artifact hashes.

The dry-run planner passes only if all required files exist, every selected
system has one Stage 2 system row for the selected role, at least one Stage 2
member row, a valid initial-state pickle, and a valid reconstructed-climate
file.
