# Round 123 Southeast Tibet d18O Proxy Qualification

## Objective

Qualify the public MiMei and Xinpu annually resolved tree-ring d18O series as
new HMA reconstruction inputs before any assimilation experiment.

## Frozen boundaries

- Original project MAT, NetCDF, CSV, MATLAB, and prior experiment files are read-only.
- Results are written only to the independent Round 123 result directory.
- Candidate values for 1921-1939 are never decoded or used.
- The sealed external final-test reserve is not opened.
- Modern assessment uses only the frozen blocks 1940-1951, 1952-1963,
  1964-1975, 1976-1987, and 1988-2000.
- No OGGM experiment is run.

## Candidate evidence

- MiMei tree-ring d18O: PANGAEA 869839, 29.45 N, 96.43 E.
- Xinpu tree-ring d18O: PANGAEA 869840, 30.42 N, 95.07 E.
- Both records are documented as responding to summer precipitation, relative
  humidity, and temperature. Their raw proxy values, not published climate
  reconstructions, are assessed.

## Qualification tests

1. Verify source download hashes and documented metadata.
2. Measure spatial novelty against the reconciled project proxy ledger.
3. Extract candidate values only for 1940-2000 while explicitly skipping
   1921-1939 before value decoding.
4. At the nearest supported HMA grid cell, form frozen modern temperature and
   precipitation observation consensuses from the existing project products.
5. Fit temperature-only, precipitation-only, and bivariate linear PSMs in five
   blocked out-of-fold tests. Report OOF correlation, skill against a training-
   mean baseline, coefficient signs, and fold stability.

## Decision rule

A record may advance to an incremental reconstruction experiment only if it is
spatially nonduplicate and at least one physically documented PSM branch has
positive pooled OOF skill and positive pooled OOF correlation. Advancement is
experimental only and does not modify the production proxy network.
