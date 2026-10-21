# Layer 0/1 calibration implementation specification

**Workflow mapping:** physical Layers 0/1, implemented by workflow Stages 1/2  
**Document status:** v0.1 design specification (2026-07-18)  
**Execution status:** closed until the data-registration gates in Section 14 pass

## 1. Purpose

This document defines the implementation boundary between the observation
inventory and calibration code. It specifies interfaces, artifacts, ownership,
validation hooks, failure states, and convergence rules for:

- Layer 0: product-conditional effective rheology;
- Layer 1: annual mass-balance parameters and turnover-uncertainty freeze;
- the Stage 1 apparent-MB consistency loop;
- the Stage 2 equivalence-ensemble freeze and velocity soft-evidence ledger.

It does not define the LIA initial-state search, transient climate test, or
outer-loop scientific inference. Those consume the frozen Stage 2 posterior and
must not rewrite it.

## 2. Layer-to-stage map

| Workflow stage | Physical responsibility | Free quantities | Output ownership |
|---|---|---|---|
| Stage 0 | Geometry, observation registration, and structural-scenario declaration | Discrete structure only | Input registry and structure manifest |
| Stage 1 | Static calibration and apparent-MB consistency | Product-conditional `glen_a`; provisional Layer 1 posterior ridge | Layer 0 posterior, Layer 1 candidate posterior, alpha-loop diagnostics |
| Stage 2 | Turnover-uncertainty freeze | Unweighted low/middle/high representatives along the Stage 1 Layer 1 equivalence surface | Frozen Layer 1 ensemble, velocity routing evidence, and identifiability diagnostics |
| Stage 3 | Initial-state search | Initial model state only | LIA initial-state ensemble |
| Stage 4 | Transient forward test | None | Trajectory and modern terminal residual |
| Stage 5 | Diagnosis and beta/gamma decisions | Structural weights or climate correction | Scientific diagnostics; never Stage 1/2 parameter rewrites |

## 3. Non-negotiable implementation invariants

1. Original notebooks, original OGGM working directories, and original data are
   read-only inputs. Every new run writes under the organized workspace.
2. No LIA geometry, `SP/LIA`, `FW/RGI`, forward endpoint residual, or reconstructed
   climate metric may enter a Stage 1/2 objective function.
3. OGGM `distributed_thickness`, `volume_m3`, `area_m2`, `length_m`, and the
   generated bare DEM are model products, never observation records.
4. Surface velocity contributes no Stage 1/2 parameter likelihood under the
   audited policy. Stage 2 owns its single registered use as soft structural
   evidence and routing metadata; Stage 4 may read it for residual diagnosis.
5. Farinotti and Millan thickness runs are separate structural scenarios. Their
   difference is not collapsed into a single supposedly independent truth.
6. Stage 2 creates a new immutable artifact derived from Stage 1. It never edits
   a Stage 1 artifact in place.
7. Stage 3 and later may read Stage 1/2 parameters but cannot write any Layer
   0/1 parameter field.
8. A weakly identifiable glacier exports a posterior interval or ensemble, not
   a forced point value and not a silent regional-mean replacement.

## 4. Isolated execution layout

The implementation must use this logical layout. Directories are created only
when execution code is authorized and the data gates pass.

```text
With_Rec_climate_organized/
  00_constraint_inventory.md
  01_layer0_1_calibration_spec.md
  registry/
    observations.jsonl
    structures.jsonl
    registry_manifest.json
  runs/
    <run_id>/
      run_manifest.json
      oggm_workdir/
      stage_01_static/
        layer0_parameter_rows.jsonl
        layer0_posterior.nc
        layer1_candidate_rows.jsonl
        layer1_candidate_posterior.nc
        alpha_diagnostics.jsonl
        failures.jsonl
      stage_02_turnover/
        layer1_parameter_rows.jsonl
        layer1_posterior.nc
        jacobian_diagnostics.jsonl
        velocity_residuals.nc
        failures.jsonl
      audit/
        observation_use_ledger.jsonl
        write_validation.jsonl
        artifact_hashes.json
```

`run_id` is immutable and unique. Re-entry creates a child run with
`parent_run_id`; it does not overwrite the parent. All paths in a run manifest
must resolve inside that run directory except declared read-only inputs.

## 5. Observation registry contract

Every observation or prior used by a likelihood must have one registry row.

| Field | Required meaning |
|---|---|
| `observation_id` | Stable product-and-variable identifier |
| `product_name`, `product_version` | Published product identity |
| `local_path` | Registered read-only file or directory |
| `data_hash` | File hash or deterministic directory manifest hash |
| `variable`, `unit` | Quantity supplied to the observation operator |
| `time_start`, `time_end` | Observation epoch or averaging period |
| `coverage_geometry` | Spatial coverage and mask reference |
| `uncertainty_model_id` | Error distribution and spatial covariance definition |
| `correlation_group` | Shared-error group across products or variables |
| `operator_id`, `operator_version` | Model-to-observation sampling operator |
| `allowed_layer`, `allowed_stage` | Causal ownership boundary |
| `likelihood_role` | `primary`, `alternative-scenario`, `cross-check`, or `prior` |
| `quality_flags` | Product flags retained without silent filtering |
| `citation` | Dataset citation and license record |

An observation with an absent path, hash, unit, uncertainty model, or operator
cannot enter calibration. It may remain registered with `status = unavailable`.

### 5.1 Initial product policy

| Quantity | Primary use | Alternative/cross-check | Likelihood rule |
|---|---|---|---|
| Thickness | Farinotti 2019 structure scenario | Millan 2022 structure scenario; GlaThiDa local checks | Farinotti and Millan are separate runs, not simultaneous independent terms |
| Velocity | ITS_LIVE composite | Millan 2022 velocity | ITS_LIVE is Stage 2 soft structural evidence and a Stage 4 diagnostic; Millan remains a correlated structural cross-check, never an added likelihood term |
| Geodetic MB | Hugonnet 2000-2020 | Product flags/uncertainty variants | Stage 1 likelihood |
| Precipitation prior | W5E5-based winter precipitation mechanism | HAR/ERA5/gauge evidence for regional width | Stage 1 soft prior; Karakoram width is explicitly larger |
| Debris | Scherler 2018 | Herreid and Pellicciotti 2020 | Stage 0 structure and later stratification, not a continuous MB tuning target |

Millan thickness is velocity-driven. Its registry row must share an appropriate
`correlation_group` with related velocity information or use conservative error
inflation. A different filename does not imply independent information.

## 6. Structural-scenario contract

Every Stage 1 run operates under one immutable `structure_id`.

| Field | Meaning |
|---|---|
| `structure_id` | Hash-stable scenario identifier |
| `thickness_product` | `farinotti2019` or `millan2022` in the first ensemble |
| `motion_partition` | `fs_zero` or a versioned sliding-prior scenario |
| `debris_model` | `none` or a versioned debris-aware option |
| `bed_flowline_version` | Exact bed and flowline configuration |
| `dem_version` | Surface DEM identity and hash |
| `climate_baseline_version` | CRU/W5E5 processing identity used in modern calibration |
| `dependency_notes` | Known model or data dependence, including velocity-driven thickness |

Continuous parameters cannot change a `structure_id`. Changing any structural
field starts a beta-loop child run from Stage 0.

## 7. Calibration interfaces

The names below define semantic interfaces. Exact Python module names may
follow the repository style, but input/output ownership cannot change.

### 7.1 Registry and context

```python
load_calibration_context(run_manifest, observation_registry, structure_manifest)
    -> CalibrationContext

validate_calibration_context(context)
    -> ValidationReport
```

The context is read-only after validation and carries data hashes, priors,
bounds, structural identity, glacier selection, random seed, and method version.

### 7.2 Layer 1 static candidate posterior

```python
calibrate_static_mb_candidates(context, glacier_id, geodetic_mb, prcp_prior)
    -> MbCandidatePosterior
```

This function evaluates `prcp_fac`, `melt_f`, and `temp_bias` using modern
geodetic MB and the precipitation prior. Before velocity is applied, the result
may be a ridge or broad posterior. It must not be reduced to a unique point just
to satisfy a file format.

### 7.3 Layer 0 effective rheology

```python
calibrate_effective_rheology(
    context, glacier_id, structure_id, mb_candidate_posterior, thickness_target
) -> RheologyPosterior
```

The result is explicitly conditional on the thickness product, motion scenario,
bed/flowline version, and MB candidate distribution. No surface-velocity term is
allowed in this v1 interface.

### 7.4 Apparent-MB consistency loop

```python
iterate_apparent_mb_consistency(
    context, glacier_id, mb_candidate_posterior, rheology_posterior
) -> Stage1CalibrationArtifact
```

Each iteration recomputes apparent MB and inversion-dependent model state using
modern evidence only, then updates the conditional Layer 0 and Layer 1
posteriors. It records every iteration rather than replacing earlier values.

### 7.5 Stage 2 turnover-uncertainty freeze

```python
freeze_mb_equivalence_ensemble(
    context, stage1_artifact, registered_velocity_diagnostics
) -> Stage2CalibrationArtifact
```

Layer 0 and the structural scenario are frozen. The function selects
low/middle/high representatives along the Stage 1 Layer 1 equivalence surface
without velocity weighting. Velocity diagnostics are attached as soft evidence
and routing fields; they cannot change candidate weights or parameter values.

### 7.6 Validation and publication

```python
assess_identifiability(context, stage2_artifact) -> IdentifiabilityReport
validate_stage_write(context, proposed_artifact) -> ValidationReport
publish_immutable_artifact(context, proposed_artifact) -> ArtifactReference
```

Publication is atomic: validation passes first, then the artifact and its hash
are written, and only then is the manifest updated.

## 8. Observation operators and objectives

### 8.1 Thickness objective

The thickness operator must declare whether it compares volume, gridded
thickness, centerline thickness, or a combination. Product masks, resolution,
and spatial covariance are part of the operator, not preprocessing details.

The Layer 0 target is written as a product-conditional posterior:

```text
p(glen_a | thickness_product, MB candidates, structure_id)
```

It must never be described as calibration to direct thickness measurements
unless the contributing records are actual GlaThiDa-type measurements.

### 8.2 Static MB objective

The Stage 1 Layer 1 objective contains only:

- modern geodetic MB likelihood;
- precipitation-factor prior;
- declared priors/bounds for `melt_f` and `temp_bias`;
- registered modern-climate uncertainty.

The current notebook behavior of fixing `prcp_fac = 8` and using one scalar MB
to select both `melt_f` and `temp_bias` is not an identifiable v1 objective.

### 8.3 Velocity/turnover soft-evidence contract

The Stage 2 evidence operator must specify:

- surface-to-depth-mean velocity treatment under each motion scenario;
- valid velocity epoch and match to the modern geometry/MB period;
- along-flowline or cross-section sampling;
- uncertainty inflation for low-speed, debris-covered, shadowed, or masked ice;
- covariance with velocity-driven thickness products;
- the exact `observation_use_id` entered in the soft-evidence ledger.

Velocity diagnoses annual flux/turnover structure but does not contract the
static MB candidate posterior. It is retained for routing and Stage 4 residual
interpretation and is not used to retune Layer 0 or Layer 1.

## 9. Stage 1 alpha-loop contract

For each glacier and `structure_id`:

1. Construct the initial Layer 1 static candidate posterior.
2. Compute apparent MB for posterior representatives or draws.
3. Run the declared inversion/flowline operator.
4. Fit the product-conditional Layer 0 effective-rheology posterior.
5. Re-evaluate any Layer 1 quantities affected by the updated inversion state.
6. Repeat until the registered convergence rule passes or `max_iterations` is reached.

The convergence configuration must contain:

| Field | Meaning |
|---|---|
| `max_iterations` | Hard bound on the alpha loop |
| `parameter_shift_tol` | Maximum standardized posterior-location change |
| `posterior_width_shift_tol` | Maximum standardized posterior-width change |
| `thickness_residual_tol` | Product-specific, uncertainty-scaled residual threshold |
| `mb_residual_tol` | Uncertainty-scaled modern MB residual threshold |
| `required_stable_iterations` | Consecutive iterations required for convergence |

No threshold is hard-coded in a notebook. Values and justification live in the
run manifest. A non-converged loop exports diagnostics and a failure code; it
does not pass its latest point estimate downstream as if converged.

## 10. Identifiability calculation and routing

The Jacobian is computed in standardized parameter coordinates and whitened by
the observation covariance:

```text
J_w = L_y^-1 * d h(z) / d z
```

where `z` is the dimensionless Layer 1 parameter vector and
`L_y * L_y^T` is the registered observation covariance. The report contains:

- singular values and numerical rank;
- condition number `kappa(J_w)`;
- singular-value tolerance and `kappa_max` used;
- dominant weak parameter combination;
- posterior/prior width ratio for each parameter;
- observation coverage and quality flags.

Routing is deterministic:

| Condition | Status | Downstream representation |
|---|---|---|
| Full numerical rank and `kappa <= kappa_max` | `ok` | Posterior draws plus summaries |
| Rank deficient or `kappa > kappa_max` | `weak-identifiability` | Widened or prior-dominated posterior draws; no forced point row |
| Parameter posterior mass accumulates at a bound | `parameter-at-bound` | Classified failure; no normal publication |
| Missing/invalid required observation | `missing-observation` | No calibration artifact |
| Alpha loop exceeds iteration bound | `alpha-nonconverged` | Diagnostics only |
| No candidate can match uncertainty-scaled modern evidence | `structural-incompatibility` | Beta-loop review list |
| Observation operator fails or has zero valid samples | `operator-failure` | Classified data/operator review |

`weak-identifiability` means reduced precision, not glacier exclusion. Its draws
must propagate into the C-line climate inference.

## 11. Parameter artifact schema

Each summary row contains the provenance fields frozen in the inventory plus:

| Field | Meaning |
|---|---|
| `structure_id` | Structural scenario conditioning this posterior |
| `data_scenario_id` | Primary/alternative observation scenario |
| `candidate_set_id` | Stage 1 posterior ridge/ensemble identity |
| `parent_artifact_hash` | Exact Stage 1 parent used by Stage 2 |
| `posterior_file`, `posterior_variable` | Location of full posterior draws |
| `credible_interval` | Registered interval level and limits |
| `identifiability_status` | `ok`, `weak-identifiability`, or failure state |
| `jacobian_rank`, `jacobian_condition` | Whitened sensitivity diagnostics |
| `boundary_mass` | Posterior mass near each configured bound |
| `observation_use_ids` | Likelihood-ledger entries consumed by this result |

`posterior_mean` and `posterior_std` are summaries only. The full posterior file
is the downstream scientific object.

## 12. Validation hooks

Every run must execute these hooks before artifact publication:

1. `path_isolation`: reject writes outside `runs/<run_id>`.
2. `input_hash_match`: reject changed inputs relative to the run manifest.
3. `stage_write_ownership`: reject writes to fields not owned by the stage.
4. `observation_use_unique`: reject duplicate primary likelihood use of the
   same `observation_use_id` within a data scenario.
5. `forbidden_evidence_scan`: reject Stage 1/2 objective declarations containing
   LIA, spinup, forward, terminal-residual, or reconstructed-climate inputs.
6. `model_product_blacklist`: reject OGGM diagnostic products registered as observations.
7. `posterior_required`: reject point-only output for uncertain parameters.
8. `rank_route_match`: verify that rank/condition results produced the required status.
9. `bounds_and_units`: validate parameter units, priors, bounds, and boundary mass.
10. `lineage_complete`: verify parent hashes, method version, code hash, and timestamps.
11. `resume_idempotency`: verify that re-running a completed stage with identical
    inputs produces the same artifact hash or exits without rewriting it.

## 13. Disposition of current notebook cells

The master notebook remains a historical and exploratory record. Its existing
cells map to the new specification as follows:

| Current code | Disposition | Reason |
|---|---|---|
| Code cell 8: fixed `prcp_fac=8`, scalar-MB calibration of `melt_f/temp_bias` | Historical baseline only | One scalar MB plus a fixed precipitation factor does not close the three-parameter Layer 1 posterior |
| Code cell 11: copy RGI parameters to LIA; fill missing glaciers with regional means | Forbidden for calibrated artifacts | Silent mean replacement destroys glacier-level provenance and uncertainty |
| Code cell 24: apparent MB, inversion, filtering, present-time flowline initialization | Candidate Stage 1 computational operator | The chain may generate model state, but its thickness/volume outputs are not observations |
| Code cells 27-36: LIA-driven `melt_f`/`temp_bias` scans | Historical cross-layer baseline | Retained for the disciplined-versus-cross-layer comparison only |

No parameter selected by cells 27-36 may be imported into a Stage 1/2 artifact.

## 14. Entry gates before calibration code

Implementation may begin only after the interface-level fields below are
decided. Execution may begin only after their concrete data records validate.

| Gate | Required decision |
|---|---|
| G1 thickness | Exact Farinotti/Millan/GlaThiDa versions, paths, hashes, uncertainty, and comparison operator |
| G2 velocity | ITS_LIVE version/epoch/path, Millan cross-check role, masks, uncertainty, and sampling operator |
| G3 precipitation | W5E5 version/path and quantitative regional prior-width rule |
| G4 modern climate | Whether Stage 1 remains CRU-based or moves to W5E5; no unregistered numeric-factor transfer |
| G5 MB | Hugonnet uncertainty fields, flags, units, correlation assumptions, and glacier matching |
| G6 structures | First sliding, debris, bed/flowline, and thickness-product scenario manifest |
| G7 identifiability | Standardization, singular-value tolerance, `kappa_max`, and posterior-widening rule |
| G8 pilot set | Versioned glaciers representing clean, Karakoram/weak-prior, and debris/low-speed cases |

Until G1-G8 are registered, no new centennial scan or calibrated parameter file
may be produced.

## 15. Planned implementation order after gates pass

1. Implement registry and manifest validation without running OGGM.
2. Implement immutable artifact publication and write-ownership tests.
3. Implement observation operators and synthetic fixtures.
4. Implement Stage 1 static candidate posterior and Layer 0 product scenarios.
5. Implement the alpha loop and its convergence/failure artifacts.
6. Implement Stage 2 unweighted equivalence-ensemble freeze and velocity
   soft-evidence ledger.
7. Run the representative pilot set and freeze thresholds only after reporting
   sensitivity to the pre-registered alternatives.
8. Scale to the 245-glacier subset only after pilot audit passes.

The next coding step is therefore registry/validation infrastructure, not an
OGGM parameter scan.
