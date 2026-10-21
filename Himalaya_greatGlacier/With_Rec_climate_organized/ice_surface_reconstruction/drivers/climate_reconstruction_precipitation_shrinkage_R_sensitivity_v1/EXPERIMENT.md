# Precipitation Shrinkage-R Sensitivity V1

Date: 2026-08-13

## Purpose

Test one statistically defined alternative to the production diagonal
observation-error matrix for the reproducible
`proxy6 / EnKF / experiment 1` chain over 1901-1920.

## Frozen design

- Prior, 500-member experiment-1 sample, proxy observations, HXb, and residuals
  are unchanged.
- Production branch: diagonal residual variance, exactly as archived.
- Sensitivity branch: Ledoit-Wolf shrinkage correlation estimated from the
  standardized 202 x 61 frozen residual matrix.
- Shrinkage strength is estimated automatically; it is not scanned or tuned.
- The sensitivity matrix is rescaled to preserve every production diagonal
  residual variance exactly.
- No localization, inflation, proxy removal, PSM change, or OGGM.
- Years 1901-1920 only; 1921-1939 remains locked.

The diagonal branch must close against the archived experiment-1 posterior
before the shrinkage branch can be interpreted. This is sensitivity evidence,
not authority to replace the production R.
