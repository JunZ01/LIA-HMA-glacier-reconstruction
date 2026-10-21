# Round 129 New d18O Seasonal-Prior Coupling Contract

## Purpose

Test whether the three documented-season operators advanced by Round 128, plus the retained annual Ganesh operator, have stable prior covariance with the annual HMA precipitation state. This is an operator-feasibility experiment only; it is not a production reconstruction.

## Frozen inputs

- CESM-LME010 monthly TREFHT, PRECC, and PRECL prior files.
- Prior member years 1001-1848 and the recovered 336-cell annual precipitation prior.
- Proxy locations and documented seasons from Round 128.
- Modern calibration years 1940-2000, with Keylong 1940 unavailable by construction.
- BEA+GPCC and CRU calibration products and the Round 128 OOF residual errors.

## Candidate operators

- MiMei: JJA temperature plus JJA precipitation.
- Xinpu: JJA temperature plus JJA precipitation.
- Ganesh: annual temperature plus annual precipitation; the rejected JJAS operator is not used.
- Keylong: current JJA temperature plus previous-October to current-April precipitation.

## Locked boundaries

- Do not decode or use candidate/current climate values from 1921-1939.
- Do not open the frozen external validation reserve.
- Do not modify source MAT, NetCDF, text, CSV, or MATLAB files.
- Do not alter production PSMs or run OGGM.

## Predeclared gates

The extraction preflight must reproduce local anomalies in the recovered annual target prior with correlation at least 0.995 and normalized RMSE at most 0.10. A candidate advances only if:

1. BEA+GPCC and CRU operator anomalies correlate at least 0.90.
2. Their 336-cell target-correlation maps correlate at least 0.80.
3. Both products have absolute correlation at least 0.10 with the annual HMA-mean prior.
4. In both products, at least 10% of HMA cells have absolute operator-target correlation at least 0.20.
5. In both products, operator variance divided by operator variance plus frozen OOF residual variance is at least 0.10.

Passing authorizes only a subsequent frozen EnKF addition sensitivity. Failure closes that candidate as a route to the annual precipitation reconstruction under the current prior.
