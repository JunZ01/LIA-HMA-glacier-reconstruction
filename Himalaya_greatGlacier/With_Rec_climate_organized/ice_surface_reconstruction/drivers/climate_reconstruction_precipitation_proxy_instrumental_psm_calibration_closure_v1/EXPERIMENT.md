# Precipitation Proxy Instrumental PSM Calibration Closure V1

Date: 2026-08-13

## Purpose

Test whether the combined 0.5-degree Berkeley Earth and GPCC-derived fields,
sampled at the nearest grid cell, can reproduce the archived 1940-2000 proxy
PSM calibration behavior. This is required before interpreting the separate
1901-1920 out-of-sample proxy trend audit.

The archived bivariate formula and coefficients are frozen. No PSM is refit.
For each in-grid tree record, compare the archived proxy chronology with
`b0 + b1*T + b2*P` over 1940-2000 and compare the resulting correlation with
the archived `pcr` value.

This is calibration-period closure only, not independent validation. The
locked 1921-1939 period is not accessed.
