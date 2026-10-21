# Round 238 contract: NPL065 documented winter Tmin modern OOF

## Question

Does the newly recovered, source-independent NPL065 Gaurishankar shrub-ring
network reproduce its documented positive December-February minimum-temperature
signal under blocked out-of-fold validation, and is that signal robust to a
second gridded temperature product?

## Frozen design

- Proxy source: the unmodified NOAA `npl065.rwl` recovered in Round 237.
- Site: 27.89 N, 86.37 E.
- Chronology input window: 1941-2000 only. Values before 1941 and all
  1921-1939 slots are skipped before numeric conversion.
- Exploratory chronology: modified negative exponential detrending, residual
  ratio indices, Tukey biweight mean, no prewhitening, using the already
  vendored dplpy 0.1.5 implementation.
- No equivalence to the paper's production chronology is claimed.
- Calibration/OFF validation years: 1960-2000, the intersection of the
  documented 1960-2013 calibration interval and the frozen local products.
- Exact target: CRU TS4.01 monthly minimum temperature, previous December plus
  current January-February, at the nearest grid cell.
- Robustness target: Berkeley Earth monthly mean temperature for the same
  three-month winter and nearest grid cell. This is not labelled Tmin.
- Folds: 1960-1967, 1968-1975, 1976-1983, 1984-1991, 1992-2000.
- Model: in each fold, fit only an intercept and chronology slope on the other
  four blocks; predict the held-out block.
- Null: all 40 non-zero circular shifts of the 41-year chronology. The frozen
  statistic is the smaller dual-product OOF correlation when all ten fold
  slopes are positive, otherwise zero.

## Admission gates

All gates must pass:

1. Complete 1960-2000 chronology with minimum sample depth at least 6.
2. Complete CRU Tmin and Berkeley winter targets.
3. At least 75% of CRU target month-cell slots report station support.
4. OOF correlation at least 0.30 for both products.
5. OOF skill against each training-fold climatology is positive.
6. At least four of five held-out fold correlations are positive for each
   product.
7. All ten fitted chronology slopes are positive.
8. Both 1960-1979 and 1980-2000 proxy-target correlations are positive for
   both products.
9. The two OOF prediction curves correlate at least 0.80.
10. Circular-shift empirical p-value is at most 0.05.

Thresholds are frozen before opening NPL065 ring-width magnitudes or local
climate magnitudes. A near miss is not relaxed.

## Boundaries

- Original NOAA, CRU, Berkeley, MAT, NetCDF, CSV, and source files are read-only.
- Round 238 writes only to its independent result directory.
- No 1921-1939 numeric value is converted or exported.
- No paleo chronology, reconstructed temperature curve, or historical fit is
  produced.
- Jionglacuo is not opened or hashed.
- No EnKF, production reconstruction, climate correction, or OGGM run occurs.
- Passing advances only to exact chronology-method recovery and historical
  transport testing; it does not admit a PSM into production.
