# Coordinate-corrected 0.5-degree reconstruction v1

## Question

What do the six archived reconstruction members look like on the public HMA
0.5-degree grid when they are interpolated directly from their verified
2-degree cell centers, without the misregistered GeoTIFF export chain?

## Frozen scope

- Source: `tas_result_matlab_for_py.mat`, read-only.
- Members: AO/HGAO/OEnKF crossed with proxy datasets 6 and 9.
- Source coordinates: 63, 65, ..., 109 E and 25, 27, ..., 51 N.
- Target coordinates: the declared public 0.5-degree cell-center grid.
- HMA mask: finite-all-years mask from `tas_ensemble_mean_rec_ref`.
- Target diagnostic: 33.75 N, 75.75 E for system 3584.
- Regional diagnostic: fixed intersection of the HMA mask and complete
  1901-2000 coverage in all five instrumental products.
- Reference period: 1961-1990; analysis period: 1901-2000.
- No OGGM simulation and no glacier curve enters a climate decision.

## Hard gates

- All six MAT members must share the verified source coordinates and years.
- Interpolation evaluated at every source center must reproduce the source.
- The target output coordinates and cell bounds must have no half-cell shift.
- The 3584 output node must equal direct interpolation at 33.75 N, 75.75 E.
- Output trends must equal trends interpolated from the 2-degree source nodes.
- The written six-member mean must close to the arithmetic member mean.
- The written HMA mask must exactly match the separately verified mask.
- Source hashes must remain unchanged.

The three HMA-mask cells at 24.75 N lie 0.25 degree south of the first source
center. They are retained with explicitly recorded linear boundary
extrapolation; all other retained HMA cells are inside the source-center
convex hull.
