# LIA-to-modern system manifest specification

**Artifact version:** v0 pre-Stage-0 audit (2026-07-19)

**Execution status:** specification input only; not calibrated parameters

**Machine-readable artifacts:**

- `outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/system_manifest_work/system_manifest_245_v0.csv`
- `outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/system_manifest_work/system_manifest_245_v0_summary.json`

## 1. Purpose and ownership

The manifest is the system-level join between the 245 LIA glacier systems and
their modern observation members. One row is one future dynamic trajectory.
The 1,205 RGI60 glaciers are modern geometry/observation members aggregated to
those trajectories; they are not 1,205 independent LIA initial models.

The artifact is written by `pre-stage0-system-manifest` and has the frozen
allowed-use stamp:

```text
stage0-1-specification-input;not-calibrated-parameters
```

It contains no calibrated `glen_a`, `prcp_fac`, `melt_f`, `temp_bias`, initial
state, OGGM transient result, or reconstructed-climate test result.

## 2. System and membership rules

1. `lia_unique_id` is the stable system key and must be unique.
2. `main_rgi_id` identifies the modern main trunk used for the preliminary
   velocity gate. It does not replace the complete modern member set.
3. `download_member_rgi_ids` is the conservative modern-data acquisition
   scope. System-level modern geometry, thickness, debris, and MB audit fields
   in v0 use this downloaded scope.
4. `confirmed_member_count` and `candidate_member_count` preserve membership
   certainty. Candidate members must never be silently discarded or promoted.
5. Member ID lists are semicolon-delimited, stable RGI60 identifiers.

The v0 crosswalk contains 245 unique LIA systems and 1,205 unique downloaded
modern members. System 2692 remains unresolved: 25 members were declared, 26
were downloaded, 17 are confirmed, and 9 are spatial-superset candidates. Its
aggregates are retained for acquisition and audit but are not authorized for
scientific likelihoods until the membership review is closed.

## 3. Geometry construction and repair

All area operations use the audited projected geometry view. Source
shapefiles and RGI records remain read-only.

- Invalid source geometries are repaired only in the derived view with a
  make-valid operation before intersection or union.
- The repair stamp is `lia-make-valid`, `modern-make-valid`, or `none`.
- No repaired geometry is written back to the source dataset.
- Thirty-two LIA geometries and one modern member geometry required a derived
  repair in v0.
- Modern member polygons are unioned by `lia_unique_id`; overlap is therefore
  counted once in `modern_union_area_km2`.

The system geometry diagnostics are:

```text
modern_inside_lia_area = area(modern_union intersect LIA)
modern_outside_lia_area = area(modern_union minus LIA)
modern_outside_lia_fraction = modern_outside_lia_area / modern_union_area
deglaciated_lia_area = area(LIA minus modern_union)
modern_area_fraction_of_lia = modern_union_area / LIA_area
```

`modern_rgi_area_sum_km2` remains the sum of member registry areas and is kept
separate from the spatial union area. `modern_connected_components` records
the number of components in the modern union.

Geometry routing is frozen as follows:

| Condition | Geometry status | Workflow effect |
|---|---|---|
| Derived repair only, no other warning | `ready-after-derived-repair` | Retain repair stamp and continue |
| Modern-outside-LIA fraction from 0.01 through 0.05 | `watch-modern-outside-lia` | Continue with a non-blocking geometry warning |
| Modern-outside-LIA fraction greater than 0.05 | `review-modern-outside-lia` | `review-required` before Stage 0 construction |
| Unresolved member superset | `review-crosswalk` | `review-required` before scientific aggregation |

The three review systems are 2692, 2907, and 4637. Systems 2907 and 4637 have
6.23% and 5.32% of the modern union outside the LIA polygon, respectively.

## 4. Observation aggregation rules

### 4.1 Thickness and bed scenarios

Farinotti 2019 and Millan 2022 v1.1 member volumes are summed within each
downloaded member set. Product-specific equivalent mean thickness is the
summed volume divided by that product's positive-thickness raster support.
The reported relative difference is:

```text
100 * (Millan system volume - Farinotti system volume) / Farinotti system volume
```

Farinotti is the primary structural scenario and Millan is the alternative
structural scenario. They are not two independent likelihood terms. All 1,205
downloaded members have paired values. The v0 totals are 642.50 km3 and 748.77
km3; Millan is 16.54% higher.

The new bare DEM is still pending inversion. Its only allowed construction is:

- modern ice: surface DEM minus model-consistent inversion thickness;
- deglaciated LIA area: modern exposed surface DEM.

The historical bare DEM is forbidden in the new workflow and is not an
independent thickness observation.

### 4.2 Hugonnet mass balance

Only members with valid `dmdtda` for 2000-2020 contribute to the system MB
likelihood. Let `A_i` be the Hugonnet product support area, `b_i` the member MB,
and `s_i` its reported uncertainty. Then:

```text
b_system = sum(A_i * b_i) / sum(A_i)
s_independent = sqrt(sum((A_i * s_i)^2)) / sum(A_i)
s_fully_correlated = sum(A_i * s_i) / sum(A_i)
```

`s_independent` is a lower bound and `s_fully_correlated` is an upper bound.
The pair brackets uncertainty without pretending that a spatial covariance
model has been identified; every row is stamped
`bracket-no-spatial-covariance`.

Members with missing `dmdtda` receive zero MB-likelihood weight but remain in
geometry, thickness, debris, and membership accounting. Five modern members
are missing MB values in v0. Their IDs remain explicit in the row and summary
provenance. Missing-member count, product-area coverage, RGI-area coverage,
measurement coverage, and valid observation years must propagate unchanged to
the Stage 1 observation operator.

### 4.3 Precipitation prior and baseline climate

The manifest registers the W5E5 prior class and its multiplier/log-space
interval for each system. `standard-w5e5` applies to 159 systems and
`upper-indus-wide` to 86 systems. These fields are prior metadata, not fitted
`prcp_fac` values. The baseline is GSWP3_W5E5, OGGM flattened 2023.2 monthly.
CRU-to-W5E5 parameter transfer remains forbidden.

### 4.4 Velocity

The v0 velocity fields are copied from the ITS_LIVE v2.1 gate on
`velocity_gate_rgi_id`, currently the system's main RGI trunk. The interval is
the registered robust gate interval with a 0.5 m yr-1 formal-error floor.

This is a gate-availability and identifiability audit, not a system-level
parameter likelihood. The audited operator is registered once before Stage 2
as soft structural evidence and a routing signal. It cannot calibrate effective
`glen_a` or contract the MB equivalence ensemble; Stage 4 may reuse the frozen
diagnostic only for residual interpretation, not parameter fitting.

### 4.5 Debris and LIA timing

Debris fractions are system-area fractions formed from the modern member
registry: member debris-covered areas are summed and divided by summed modern
member area. Scherler LS8_RATIO is primary; FDC class and Herreid disagreement
remain scenario/diagnostic fields, not continuous MB tuning targets.

LIA timing is joined directly by `lia_unique_id`. The four regional
core/outer trapezoid windows are Stage 3 initial-state likelihood metadata.
They are not hard exact dates and cannot enter Stage 1/2 calibration.

## 5. Overall status precedence

Status is a routing result, not a scientific quality score.

1. `review-required` takes precedence for an unresolved crosswalk or a
   modern-outside-LIA fraction above 0.05.
2. Otherwise, a system with `velocity_qc_status = unusable` receives
   `weak-identifiability`. It remains in the inventory with wider downstream
   uncertainty and is not replaced by a point estimate.
3. All remaining systems receive `ready-for-stage0-spec`, including systems
   with non-blocking watch flags or repaired derived geometry.

The frozen v0 counts are 230 `ready-for-stage0-spec`, 12
`weak-identifiability`, and 3 `review-required`. The weak-identifiability rows
have no blocking reason because reduced precision is not exclusion.

## 6. Provenance and reproducibility

The summary JSON records the generation timestamp, source paths, SHA-256
hashes, shapefile component hashes, product roles, aggregation statements, and
all audit counts. The XLSX is a review rendering of the CSV/JSON and contains
`QC Summary`, `System Manifest`, `Field Dictionary`, and `Provenance` sheets.
The CSV and summary JSON are the machine-readable source of truth.

Any regenerated manifest must fail publication unless it reproduces these
v0 invariants or explicitly increments the manifest version:

- 245 rows and 245 unique `lia_unique_id` values;
- 1,205 downloaded modern members;
- 87 ordered fields;
- 5 Hugonnet missing members;
- overall status counts of 230/12/3;
- system 2692 candidate count and parsed candidate list both equal to 9;
- no calibrated parameter or simulation-result field.

## 7. Allowed and forbidden uses

Allowed uses are system selection, observation-registration audit, Stage 0/1
implementation specification, review-queue routing, and construction of a
versioned Stage 0 input registry after the review gates pass.

The manifest must not be used as:

- a calibrated Layer 0/1 parameter artifact;
- authorization to start LIA-to-modern transient simulations;
- a velocity-weighted Stage 1/2 parameter likelihood;
- evidence that Farinotti/Millan thickness or a generated bare DEM is a direct
  independent observation;
- permission to include unresolved system 2692 in scientific aggregation;
- a route for LIA geometry or modern endpoint residuals to rewrite Stage 1/2
  parameters.

## 8. Next gate

The immediate next task is the Stage 0 geometry/bed implementation
specification. Before any OGGM run it must close the three review systems,
freeze the surface DEM identity and uncertainty, define model-consistent bare
DEM construction for both thickness scenarios, and register validation hooks
that enforce the allowed-use and write-ownership stamps above.
