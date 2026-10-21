# Round 149 NOAA Temperature Candidate Identity Contract

## Objective

Determine whether the newly acquired Eastern Tibet 2015 and Bhutan 2015
annual temperature products add a distinct numerical trajectory relative to
the frozen `proxy6` and `proxy9` networks.

## Frozen boundaries

- Treat downloaded NOAA files and all existing MAT/CSV files as read-only.
- Skip every 1921--1939 data row before tokenizing or decoding its values.
- Do not open the sealed Jionglacuo external-validation file.
- Do not run EnKF, OGGM, PSM fitting, or parameter tuning.
- Hash all sources before and after the audit and require exact equality.
- Write only to the new Round 149 result directory.

## Tests

1. Verify each raw NOAA file against the versioned acquisition manifest.
2. Export only 1000--1920 and 1940--2000 annual values.
3. Compare every published numeric series with every `proxy6` and `proxy9`
   record using correlation, affine residual, and normalized RMSE.
4. Flag exact affine aliases and near-exact standardized sequences using the
   established Round 145 thresholds.

## Scientific role

These are published regional reconstructions assembled from multiple tree-ring
records. Even if numerically distinct, they are external consistency benchmarks,
not independent validation records and not raw proxies eligible for direct EnKF
assimilation.
