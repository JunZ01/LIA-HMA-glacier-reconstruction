# Precipitation EnKF Experiment-1 Increment Attribution V1

Date: 2026-08-13

## Purpose

Attribute the excessive 1901-1920 HMA-mean precipitation decline in the
reproducible `proxy6 / EnKF / experiment 1` chain.

## Frozen inputs

- Recovered `336 x 848` precipitation prior.
- Archived experiment-1 500-member indices through the existing replay input.
- Frozen proxy6 observations and `HXb` through the existing replay input.
- Frozen RegEM residual matrix through the existing replay input.
- Original EnKF mean-update equations, `infl=1`, no localization.
- Years 1901-1920 only. The locked 1921-1939 final period is not accessed.

## Attribution

For each year, calculate the area-weighted HMA state increment and decompose it
as `sum_i[(w' K)_i * innovation_i]`, where `w` is the latitude-area weight,
`K` is the Kalman gain, and `innovation = y - mean(HXb)`.

The proxy contributions must close against the direct posterior-minus-prior
HMA-mean increment. This is a mechanism diagnosis, not authority to remove a
proxy, change a PSM, or tune EnKF parameters.
