# Round 256 NPL065 modern signal-free sensitivity contract

Port the fixed-hash dplR 1.7.9 `ssf()` ratio algorithm to the allowed
1941-2013 modern-only matrix after Cook-Peters per-series power transforms.
Use ADS30 curves with two end-slope choices:

- `ssf_positive_end_allowed`;
- `ssf_positive_end_constrained`.

Use maximum 25 iterations and weighted high-frequency chronology MAD stopping
threshold `1e-4`. The iteration-zero chronology for each end-slope choice must
close its Round 253 non-signal-free counterpart to `1e-12` in calibration
metrics. Evaluate final chronologies against the same CRU TS4.01 four-cell DJF
Tmin, six blocked OOF folds, and 25 rolling 30-year windows.

This is a modern-only mechanism sensitivity, not exact RCSigFree reproduction.
No pre-1941 numeric proxy values, locked values, third-party binary execution,
production reconstruction, EnKF, or OGGM.
