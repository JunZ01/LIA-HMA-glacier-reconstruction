# Boundary-constrained LIA surface reconstruction and validation

**Artifact version:** v0, 2026-07-22

**Status:** executable method and validation specification; Stage 0-2 inputs
remain read-only.

## 1. Scientific object

For a fixed LIA footprint and a fixed continuous bed, reconstruct an ensemble
of non-negative ice-thickness fields and surfaces:

```text
H_LIA(x, y) >= 0
S_LIA(x, y) = B(x, y) + H_LIA(x, y)
H_LIA = 0 at the mapped glacier margin
```

The LIA outline identifies the initial state only. It cannot modify frozen
`glen_a`, `fs`, `melt_f`, `prcp_fac`, or the transient climate series.

## 2. Primary reconstruction operator

The primary branch is geometry-first and climate-independent. LIA system
centerlines, tributary topology, widths and catchments are derived within the
LIA polygon. Along each flowline, the finite-difference Benn-Hulton balance is
solved from the terminus or tributary junction upstream:

```text
tau_b = rho_i * g * H * F * sin(alpha)
S = B + H
(S_i-S_prev) * (S_i+S_prev-B_i-B_prev)
    = 2 * dx * tau_b / (rho_i * g * F)
```

Negative discriminants, negative thickness, excessive thickness and invalid
profiles are failures. They are never replaced by the previous value or a
hard-coded cap. A tributary inherits the parent surface at its junction.
Surface continuity is mandatory; flux-capacity mismatch is retained as a
diagnostic because no independent LIA flux observation exists.

Flowline thickness is extended over the fixed polygon with a boundary-zero
harmonic operator. The operator preserves solved centerline values and the LIA
margin. It is a transparent two-dimensional interpolation, not a claim of a
complete two-dimensional stress balance.

## 3. Structural ensemble

The minimum ensemble has two independent axes:

1. bed: Farinotti-conditioned primary and Millan-conditioned alternative;
2. stress/shape: thin, central and thick scenarios.

The stress/shape scenarios are priors, not calibrated points. The first pilot
uses 60, 100 and 140 kPa with paired shape-factor multipliers. Scenario ranges
must be revised from validation evidence without using LIA-to-modern endpoint
residuals. A failed scenario remains in the ledger.

## 4. OGGM projection and transient run

Accepted raster and flowline states are converted to OGGM flowline sections.
A short adaptive projection removes conversion discontinuities. It is stopped
or rejected when area changes by more than 2%, the terminus moves by more than
one grid cell, or LIA outline overlap degrades beyond its registered tolerance.
The projection cannot run to a new equilibrium outline.

The accepted pre-projection and post-projection states are both saved. The
post-projection state is the restart for LIA-to-modern transient runs.

## 5. Validation hierarchy

Validation evidence is kept separate by independence and purpose.

### Level A: analytic and synthetic truth

- flat and sloping beds with analytic or high-resolution reference profiles;
- tributary-junction continuity;
- known two-dimensional masks and centerline constraints;
- synthetic OGGM/PISM states with known bed and surface.

This level tests numerical correctness and failure handling, not realism.

### Level B: modern pseudo-paleo reconstruction

For selected modern glaciers, hide the observed surface and reconstruct it
from the modern outline and registered bed. Compare reconstructed surface and
thickness with the withheld DEM and thickness products. Farinotti and Millan
are separate structural scenarios, not independent observations.

This level tests operator fidelity and efficiency under the same data model as
the LIA application. It cannot establish absolute truth when the bed is model
derived.

### Level C: direct thickness observations

Use GlaThiDa 3.1.0 direct profiles where they intersect the target glaciers.
The downloaded ITMIX archive is an inverse benchmark (published surface plus
measured bed profiles, with participant bed grids), so it is audited as an
external bed/thickness structure comparison rather than treated as direct
truth for the forward LIA surface operator. Direct points and profiles are
held out from reconstruction and used for RMSE, median bias, normalized
absolute error, elevation-stratified error and distance-to-observation error
where coverage exists. The audit records the zero-coverage case explicitly.
The original investigators and data flags remain attached to every datum.

### Level D: independent historical reconstruction

Use Reinthaler and Paul (2025), DOI `10.5281/zenodo.14336827`, as an
independent LIA surface-method comparison. Compare hypsometry, mean thickness,
volume and surface differences. This product is not direct ancient-surface
truth and therefore cannot calibrate the HMA result.

## 6. Metrics and acceptance

Every candidate reports:

- boundary maximum thickness;
- non-finite and negative-thickness counts;
- centerline reprojection RMSE;
- junction surface mismatch and flux-capacity mismatch;
- area, volume, mean/max thickness and hypsometry;
- observed-point RMSE, median bias and normalized absolute error;
- surface RMSE and quantile errors for pseudo-paleo and independent products;
- runtime, peak memory and failed-system class.

No single aggregate score hides failed physical checks. Primary acceptance
requires finite non-negative thickness, boundary preservation, centerline and
junction continuity, and OGGM projection tolerances. Observation metrics rank
accepted scenarios but never rescue a physically failed candidate.

## 7. Method comparison

The comparison includes:

1. the archived `paleoice_v1.zip` implementation as a historical baseline;
2. the new boundary-perfect-plasticity/harmonic operator;
3. OGGM dynamic spinup as a climate-conditioned structural branch;
4. published GIS reconstruction products where available.

The ITMIX method ranking is descriptive only. It is not a likelihood ranking
for the HMA LIA experiment, and no ITMIX participant result is copied into the
production bed or parameter files.

The archived prototype and PalaeoIce literature are references only. Known
prototype issues include inconsistent dynamic-shape-factor field names,
empirical stress assignment, non-conservative junction handling, unconstrained
`griddata` smoothing, and elevation-only filling that can erase a zero-margin
condition.

## 8. Scale-up gate

The method is tested first on analytic cases, ITMIX synthetic/real cases and
the existing 12-system regional pilot. Scale-up to the 191 core LIA systems is
authorized only after:

- all numerical unit tests pass without warnings;
- public-data provenance and checksums are complete;
- at least one direct-observation benchmark is available;
- failure classes and runtime are stable under parallel execution;
- Stage 0-2 source hashes remain unchanged.
