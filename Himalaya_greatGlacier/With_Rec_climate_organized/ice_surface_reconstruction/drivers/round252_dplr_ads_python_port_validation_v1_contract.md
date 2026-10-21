# Round 252 dplR ADS Python-port validation contract

Port the fixed-hash dplR 1.7.9 `src/adsf.f95` age-dependent spline solver
line by line to Python. Validate the solver without climate data:

- constant and linear invariants;
- deterministic trend-plus-cycle vector;
- one NPL065 modern-only sample;
- for a constant stiffness vector of 30 years, numerical closure against the
  existing dplpy Cook-Peters 30-year spline.

The maximum constant-stiffness closure error must be at most `1e-8` before the
port can be used in a climate experiment. No binary compilation or execution,
pre-1941 proxy values, climate values, EnKF, reconstruction, or OGGM.
