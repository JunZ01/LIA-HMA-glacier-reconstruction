# Round 250 NPL065 spline-method decomposition contract

## Frozen branches

- `modnegex`: Round 246 ModNegEx/residual baseline with linear fallback.
- `spline30`: residual indices from a fixed 30-year 50% frequency-response
  smoothing spline, matching the paper's stated starting stiffness.
- `spline50`: identical implementation with a 50-year period as a sensitivity
  control.

All branches use the same 1941-2013 modern-only raw series, Tukey biweight
chronology, no prewhitening, published four-cell CRU TS4.01 DJF Tmin target,
1960-2013 calibration, and six fixed nine-year blocked OOF folds. Also compute
all 25 rolling 30-year proxy-target correlations from 1960-1989 through
1984-2013.

This round does not implement signal-free iteration or the paper's
data-adaptive power transformation. It does not access values before 1941 or
locked 1921-1939 values, execute RCSigFree, or run production reconstruction,
EnKF, or OGGM.
