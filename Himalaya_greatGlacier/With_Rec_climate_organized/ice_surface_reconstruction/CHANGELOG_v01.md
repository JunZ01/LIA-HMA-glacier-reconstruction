# Ice Surface Reconstruction v0.1

v0.1 is a diagnostic and robustness upgrade over the frozen v0 production
implementation. It does not change the accepted scalar-tau reconstruction
fields.

## Accepted audit corrections

- Stress diagnostics use the solver's upstream-segment finite-difference
  convention, segment-mean thickness, and the same integrated stress balance.
- Network solutions sampled back to original nodes recompute slope, driving
  stress, residual summaries, thickness maximum, and volume proxy on the
  sampled grid.
- Spatially varying tau / shape_factor profiles use trapezoidal segment
  averaging. The scalar production recurrence preserves the v0 operation order.
- Discriminant tolerance is an absolute square-metre tolerance plus a
  floating-point roundoff allowance, rather than a physical tolerance that
  scales with absolute elevation.
- Duplicate raster constraints are consolidated by stable sorting and grouping;
  conflict and averaging semantics are unchanged.
- xy_to_rc() centralizes OGGM (col, row) to raster (row, col) conversion.
- Modern validation thickness ratios return explicit not-evaluated statuses
  when the truth denominator is zero.
- NumPy 2 uses trapezoid; older NumPy retains a compatibility fallback.

## Deferred maintenance

- Jacobi and conjugate-gradient allocation optimization.
- Flowline refinement vectorization.
- Structured logging.
- Typed configuration models.
- Harmonic-extension result dataclass conversion.

## Verification

- Full module suite: 167 passed in oggm_env1.
- Production rerun: 191 systems, 377 successful bed members, 381 successful
  components, and 191 NetCDF files.
- v0 versus v0.1: all six scientific arrays are exactly equal in all 191 files;
  maximum absolute difference is 0.0.
- The component results, system results, mosaic audit, run manifest, and summary
  files are byte-for-byte identical to the accepted v0 production run.

The five Millan input gaps and the v0 zero-thickness candidate-divide boundary
policy remain unchanged scientific caveats.
