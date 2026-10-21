# Round 253 NPL065 ADS/power method decomposition contract

Use the validated Round 252 v3 direct port of dplR 1.7.9 age-dependent spline
and the fixed dplR Cook-Peters power-transform formula. Compare:

- `ads30_ratio`: raw widths divided by ADS30 curve, positive end slope allowed;
- `cookpow_ads30_ratio`: Cook-Peters transformed widths divided by ADS30 curve;
- `cookpow_ads30_ratio_nonpos`: same, constraining positive end slope.

Use the same 1941-2013 modern-only slots, Tukey biweight chronology, no
prewhitening, published four-cell CRU TS4.01 DJF Tmin, fixed six-block OOF, and
25 rolling 30-year windows. Retain Round 250 ModNegEx as a read-only baseline.

No signal-free iteration, RCSigFree binary, pre-1941 proxy values, locked
1921-1939 values, production reconstruction, EnKF, or OGGM.
