# Ice-surface reconstruction module

Self-contained, path-explicit implementation of the boundary-constrained LIA
surface reconstruction and its validation hierarchy. This module is the single
source of truth; `src/paleo_surface/` is now a thin compatibility layer that
re-exports from here.

## Layout

```
ice_surface_reconstruction/
├── paths.py          One path contract for the whole module (see below)
├── core/             Pure numerics, zero path/IO dependency
│   ├── solver.py       solve_flowline, harmonic_extend_thickness, surface_quality_metrics
│   └── network.py      solve_flowline_network (topology, junction diagnostics)
├── drivers/          Validation entry points (read config, write results)
│   ├── run_synthetic_validation.py    Level A
│   ├── validate_modern_profiles.py    Level B (+ legacy_profile baseline)
│   ├── audit_public_data.py           Level C data audit (needs geopandas/rasterio)
│   └── build_validation_summary.py
├── config/
│   └── paleo_surface_validation_v0.json   Path specs use anchor+sub form
└── tests/            test_paths, test_paleo_surface_solver, test_paleo_surface_network
```

## Path contract (`paths.py`)

All paths resolve against one of three anchors. Each is auto-detected and can be
overridden by an environment variable. No code recomputes a root via
`Path(__file__).parents[N]`, `../..`, or run-UUID guessing.

| Anchor | Default | Env override |
|---|---|---|
| `repo` | `Himalaya_greatGlacier/With_Rec_climate_organized/` | `LIA_REPO_ROOT` |
| `project` | `1_LIA_himalaya_oggm/` (repo's grandparent) | `LIA_PROJECT_ROOT` |
| `validation` | `<project>/validation_data/` | `LIA_VALIDATION_ROOT` |

`resolve(spec)` accepts:
- New form (recommended): `{"anchor": "validation", "sub": "results/foo"}` → `anchor_root / sub`.
- Legacy string: absolute → as-is; relative → `repo_root / value` (matches the retired `REPO_ROOT/…` behaviour, so the old flat config and old entry points still work).

`validation_data/` and `outputs/<UUID>/` are **not** part of this module; the
module only reads/writes them through resolved anchors.

## Running

Requires a scientific env with geopandas + rasterio for the audit driver
(e.g. `oggm_env1`). The pure solver/network and their tests need only numpy/scipy.

New entry points:
```
python -m ice_surface_reconstruction.drivers.audit_public_data          --config ice_surface_reconstruction/config/paleo_surface_validation_v0.json
python -m ice_surface_reconstruction.drivers.run_synthetic_validation   --config ice_surface_reconstruction/config/paleo_surface_validation_v0.json
python -m ice_surface_reconstruction.drivers.validate_modern_profiles   --config ice_surface_reconstruction/config/paleo_surface_validation_v0.json
python -m ice_surface_reconstruction.drivers.build_validation_summary   --output-dir <resolved output_dir>
```
Old entry points (`python src/paleo_surface/<driver>.py --config config/paleo_surface_validation_v0.json`)
still work via the compatibility shims and delegate to the same implementation.

Tests:
```
python -m pytest ice_surface_reconstruction/tests/ -W error -q
```

## LIA pilot preflight (diagnostic only)

These entry points read the 12-system pilot and write versioned diagnostics to
the configured validation output. They do not edit a modern gdir or produce a
LIA initial surface.

```
python -m ice_surface_reconstruction.drivers.lia_geometry_preflight --config ice_surface_reconstruction/config/lia_surface_production_v0.json
python -m ice_surface_reconstruction.drivers.bed_margin_seam_audit --config ice_surface_reconstruction/config/lia_surface_production_v0.json
python -m ice_surface_reconstruction.drivers.lia_centerline_pilot --config ice_surface_reconstruction/config/lia_surface_production_v0.json
python -m ice_surface_reconstruction.drivers.lia_boundary_preflight --config ice_surface_reconstruction/config/lia_surface_production_v0.json
python -m ice_surface_reconstruction.drivers.lia_boundary_ridge_evidence --config ice_surface_reconstruction/config/lia_surface_production_v0.json
python -m ice_surface_reconstruction.drivers.modern_tau_calibration --config ice_surface_reconstruction/config/lia_surface_production_v0.json
python -m ice_surface_reconstruction.drivers.lia_3d_zero_boundary_pilot --config ice_surface_reconstruction/config/lia_surface_production_v0.json
python -m ice_surface_reconstruction.drivers.build_lia_readiness_summary --config ice_surface_reconstruction/config/lia_surface_production_v0.json
```

The bed taper and centerline elevation fields are registered structural probes,
not silent repairs. Adjacency marks an ice-divide review candidate; it does not
prove an ice divide or assign a flux boundary condition. Ridge-evidence PNGs
and CSVs are likewise review aids: current DEM relief is not an LIA surface
observation and cannot by itself freeze a Neumann boundary.
The calibrated 3-D pilot uses one modern, bed-product-specific effective stress
per system (fixed shape factor 0.8), an explicit connector with alpha=beta=0,
and a 100 m vector-boundary exclusion for raster constraints. This paragraph
describes the retained v0 baseline. Candidate ice divides still use zero
thickness in v0 and carry an explicit bias flag. Typed failures remain failures
and are not converted to posterior values.

## Refined-flowline scale-out v1

The v1 method freezes three numerical rules from explicit pilot and scale-out
convergence scans:

- one-dimensional perfect-plasticity integration uses a 16x piecewise-linear
  subdivision of every original bed segment; no bed smoothing is applied;
- raster constraints use `max(100 m, 1.25 * map_dx_m)` as the head/boundary
  exclusion distance;
- constraints that fall on the harmonic solver's raster zero-boundary cells
  are excluded explicitly, so vector-distance filtering and the raster
  Dirichlet boundary use the same contract.

Connector endpoint beds use the shared parent/tributary flowline-node samples,
and the first/last connector distances use the original subpixel endpoint
coordinates. Interior connector pixels and the least-cost route are unchanged.
This removes a rounded-pixel endpoint mismatch; it is not a thickness clip.

Evidence and production-candidate entry points:

```
python -m ice_surface_reconstruction.drivers.flowline_step_convergence --config ice_surface_reconstruction/config/lia_surface_production_v0.json
python -m ice_surface_reconstruction.drivers.grid_aware_boundary_scan_v1 --config ice_surface_reconstruction/config/lia_surface_production_v0.json
python -m ice_surface_reconstruction.drivers.connector_start_audit_v1 --config ice_surface_reconstruction/config/lia_surface_production_v0.json
python -m ice_surface_reconstruction.drivers.lia_3d_refined_flowline_pilot_v1 --config ice_surface_reconstruction/config/lia_surface_production_v0.json
python -m ice_surface_reconstruction.drivers.summarize_scaleout_batch1_v1 --config <scaleout-batch-config>
```

On the 12-system pilot, all 21 members with available inputs and stable
centerlines pass the v1 one-dimensional and two-dimensional gates. The separate
remaining review records are one unstable centerline system and one missing
Millan input.

On the independent 16-system scale-out batch, 13 systems had stable centerline
geometry and all 26 corresponding bed members passed at 16x. The remaining
three systems are geometry-preprocessing reviews (multipart or grid-clipped
LIA polygons), not one-dimensional or harmonic-solver failures. Against a 32x
reference, the 16x volume change has a median absolute value of 0.00019% and a
maximum of 0.0142%. Re-running the original pilot changes volume by 0.0081%
median absolute and 0.194% maximum relative to the former 4x result.

## LIA system-grid correction

The main modern RGI gdir is not the computational domain of a multi-member LIA
system. `lia_3d_system_grid_pilot_v1` builds a grid covering the complete LIA
geometry, reprojects every declared modern-member gdir, and solves disconnected
polygon components independently. There are no connectors or harmonic
propagation across disconnected components. Source overlaps use one explicit
winner per cell: the source furthest inside its own valid grid or glacier mask;
values are never averaged, and overlap differences are written to the mosaic
audit.

```
python -m ice_surface_reconstruction.drivers.lia_3d_system_grid_pilot_v1 --config <scaleout-batch-config>
```

On scale-out Batch 1, all 16 systems reconstruct their eligible components.
One system has a successful 65.474 km2 component and one documented 0.179 km2
small-component exclusion. Exclusion requires both area <= 0.25 km2 and area
fraction <= 0.5%, and only applies to a typed `centerline is too short`
resolution failure. The excluded area has no thickness fill and is stored in a
separate NetCDF mask. The system-grid correction increases
volume by 8.78% median and 20.33% maximum relative to the old main-gdir-only
calculation because modern ice in non-main member gdirs is no longer treated as
bare terrain.

## LIA dynamic-initialisation pilot

The frozen v0.1 gridded reconstruction is not modified by the OGGM dynamics
work. `lia_dynamic_matrix_pilot` builds separate LIA-component GlacierDirectory
proxies and constructs `MixedBedFlowline` objects directly. This is required
because OGGM's native `init_present_time_glacier` replaces node-wise trapezoid
lambda values with one global value.

The pilot uses source-grid transverse sections, preserves each section and its
surface width, and records any representative-thickness projection required by
OGGM's positive-bottom-width trapezoid domain. This projection is a model
representation diagnostic; it is not written back to the LIA reconstruction.
The runner also closes a stepwise zero-mass-balance budget, including unmatched
negative tributary flux and negative-section clipping. Rasterised LIA geometry
must remain one connected component; disconnected masks fail before OGGM's
catchment routing instead of being silently joined.

```text
python -m ice_surface_reconstruction.drivers.lia_dynamic_matrix_pilot \
  --config ice_surface_reconstruction/config/lia_surface_production_v01.json
```

Initial evidence does not yet release the flowline dynamics workflow. System 7
runs at 17/35/50 m, but all six five-year zero-MB cases exceed the 0.1% mass
gate because OGGM clips negative tributary transfer. System 553 requires large
equivalent-bed adjustments at many nodes. System 2692 becomes disconnected
after rasterisation at 35 and 50 m. These are explicit representation and
topology reviews to resolve before dynamic spin-up or HMA production.

## Migration baseline freeze (2026-07-22)

State recorded before the reorganisation, to prove no data/logic was disturbed:

- Ice-surface tests: **10 passing** (`test_paleo_surface_solver.py` + `test_paleo_surface_network.py`, `-W error`).
- Repo-wide collection: 14 tests collected + 2 pre-existing ImportErrors in
  `tests/test_stage3_preflight.py`, `tests/test_stage3_search.py` (unrelated to
  ice surface; not touched by this migration).
- Key input SHA-256 (must be unchanged after migration):

| file | sha256 |
|---|---|
| `public/glathida_3_1_0/glathida-3.1.0.zip` | `4929100a…b26f29f` |
| `public/alps_lia_zenodo_14336827/LIA_glacier_outlines_Alps.zip` | `1b301ab0…045c67999` |
| `public/itmix/ITMIX_dataset.zip` | `d3c8f49c…6717a8d64` |
| `stage3_initial_state_pilot_v0/pilot_selection_12.csv` | `cefab39d…40074cb` |
| `thickness_work/farinotti_glacier_stats.csv` | `dbdb1a07…50195e6f` |
| `thickness_work/millan_glacier_stats.csv` | `4ef6b644…d1eb86a8` |

## Out of scope for this reorganisation

This change only reorganises code and path handling. It does **not** wire the
drivers to `solve_flowline_network`, implement ice-divide/boundary
classification, change the harmonic solver, add full-field/stratified/dx
validation, rebuild data, alter initial state, run climate forward integration,
or commit to git. Those are tracked separately.
