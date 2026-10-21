# Proxy9 Residual Missingness Repair v1

## Question

Does the archived PSM residual encoding artificially increase proxy influence?
`psm_uni.m` initializes each 1940-2000 residual vector with zero and writes
regression residuals only for valid calibration years. A record ending before
2000 can therefore carry structural zeros into the error-variance estimate.

## Controlled Change

For every proxy9 record, use its archived observation chronology to identify
missing years in 1940-2000. Change an `Rvec` entry to `NaN` only where the
proxy observation is missing. Preserve all residuals at observed years. Run
the same archived RegEM implementation, and use the same diagonal EnKF error
matrix construction as the verified baseline replay.

Frozen inputs: proxy observations, all 297 `HXb` operators, prior, sample
indices, EnKF update, HMA mask, and instrumental comparisons. No OGGM is run.

This experiment diagnoses one software/statistical mechanism. It is not a
production reconstruction and does not authorize a calibration-window change.
