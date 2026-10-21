# Full-chain modern replay inputs: three frozen controls

Data export only: no tau fitting, perfect-plasticity/network solve, LIA reconstruction or dynamic simulation was run. All NPZ arrays load with `np.load(path, allow_pickle=False)`; no OGGM objects or pickle are stored. Float64 profile and raster inputs preserve the source/warp precision. The numerical solver module itself currently imports SciPy as well as NumPy.

## Two grid frames

Unprefixed 2-D fields (`reference_H_m`, `topo_m`, `bed_m`, `modern_mask`) are on the same saved LIA grid used in the earlier operator controls. `bed_m = topo_m - reference_H_m`; reference thickness is zero outside the modern mask, and terrain outside the source DEM coverage may be NaN. Ignore invalid/outside cells. `dx_m`, `grid_x_m`, `grid_y_m`, `crs_wkt` and `grid_transform_gdal` specify this frame.

`native_*` rasters are the original modern gdir grid. `native_bed_m` follows gridded_bed.py: topo minus finite distributed thickness inside native mask, topo elsewhere. `native_grid_json` and `native_transform_gdal` describe the source frame. Native spacings are 61/67/62 m for 3131/2906/3632; LIA spacings are 15.25/16.75/15.5 m. Do not use coordinates from one frame with the other bed.

The original exported constraint_rows/cols are retained exactly, before the optional 100 m operator exclusion.

## Ragged flowlines, NumPy only

Each flowline runs HEAD -> TERMINUS in native OGGM order. `flowline_offsets[i]:flowline_offsets[i+1]` slices all flattened node arrays. `flowline_flows_to[i]` is the parent flowline INDEX; -1 means no parent. `flowline_flows_to_indice[i]` is the join index in the PARENT's native order; -1 means none. Do not interpret the join index in reversed order.

`flowline_xy_native` and `flowline_xy_lia` are fractional (col,row) coordinates. Coordinates retain the original paths and have not been snapped, rerouted or extended. `flowline_dx_m[i]` is the original OGGM along-line spacing in metres, and flowline_distance_m resets to zero at each HEAD. The solver's downstream-to-upstream orientation must be produced by reversing each appropriate segment.

`flowline_bed_native_bilinear_m` / `flowline_bed_lia_bilinear_m` sample the corresponding 2-D bed. Bilinear sampling matches the production sampling helper; nearest samples are also provided explicitly.

`flowline_surface_oggm_m`: original OGGM 1-D surface.
`flowline_surface_dem_native_m` / `_lia_m`: DEM sampled on each frame.
`flowline_inversion_H_m`: legacy 1-D inversion thickness.
`flowline_inversion_bed_m`: original 1-D surface minus original inversion thickness, retained only for the legacy comparison.
`flowline_reference_H_native_m` / `_lia_m`: corresponding 2-D reference H sampled bilinearly.
`flowline_inversion_volume_m3`: per-node inversion volume; its sum reproduces inversion_1d_volume_km3.

The raw flows_to topology is exported, not an already split network. Use the existing explicit-connector adapter policy when reconstructing production-style branches; do not silently replace connectors by a collapsed join or alter outlet/head paths. Original connector, integration-refinement and calibration settings are carried in legacy_config_json.

## Important outlet convention

These original modern main-flowline ends are not zero-reference-thickness points: target-grid reference H is about 18.44 m (3131), 15.33 m (2906), 17.47 m (3632). The existing solve_flowline_network initialises a previously unseen outlet with start_surface_m=None, which solve_flowline interprets as S=B, H=0. Legacy modern tau calibration instead starts each profile at its stored modern reference surface. A full-chain replay must disclose and keep endpoint conventions consistent; otherwise its error also contains endpoint-condition mismatch. No endpoint correction or extension has been performed by this exporter.

## Calibration versus validation

`legacy_tau_kpa` and legacy_tau_row_json are the old 1-D-bed calibration results, not refitted values for a unified bed. `legacy_config_json` preserves original settings, including the native-grid 132 m calibration exclusion, distinct from the operator diagnostic's 100 m raster exclusion. Exported physical constants are rho=900 kg/m3, g=9.81 m/s2, shape factor=0.8 and maximum thickness=2000 m, matching the archived solver/configuration. Any change in objective, bed, boundary or exclusions must be disclosed in the replay. A modern self-replay checks consistency; temporal transfer to LIA remains untested.

The reference H is an inversion redistribution calibrated to F19 total volume, not independent observed H or raw F19 spatial thickness. Using B=DEM-Href makes a modern surface comparison a conditional consistency diagnostic, not independent validation of the reconstructed bed.

The received residual-analysis tables were retained and checked against our three A baselines; the new spline/oracle variants were not rerun. Two boundary scenarios provide a sensitivity interval, not rigorous uncertainty bounds. In the eastern glacier only about 30% of negative-H cells lie within 100 m of the raster boundary, so they cannot all be described as near-margin artifacts.

manifest.json records source/export hashes, array shapes and dtypes. sources_unchanged and same_operator_control_constraints are verified in the exporter.
