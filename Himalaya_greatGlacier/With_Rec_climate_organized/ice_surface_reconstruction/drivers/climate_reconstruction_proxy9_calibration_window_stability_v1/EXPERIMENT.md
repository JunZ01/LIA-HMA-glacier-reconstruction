# Proxy9 Calibration-Window Stability v1

This stage moves one freedom only: the position of a fixed 61-year PSM
calibration window. The preregistered windows are 1901-1961, 1920-1980, and
the archived 1940-2000 baseline.

Records 1-290 retain the original annual bivariate temperature-precipitation
`psm_uni` model. Records 291-297 retain the original annual temperature-only
`psm_ul` model. All calls use the original standardized proxy chronologies,
locations, observation products, prior model, and monthly averaging. The same
MATLAB RNG state is reset before every bivariate window so stochastic
precipitation replacement cannot masquerade as a window effect.

This stage diagnoses effective sample counts, coefficient sign and magnitude
stability, PSM skill, residual variance, and HXb stability. It does not run an
EnKF reconstruction or select a production calibration window.
