# Precipitation Observation-Error Network Audit V1

Date: 2026-08-13

## Purpose

Audit whether the production EnKF's diagonal observation-error matrix treats
strongly related proxy residuals as independent evidence.

The frozen RegEM-completed proxy6 residual matrix has 202 proxy records and 61
calibration residual years. Production computes the residual covariance and
then retains only its diagonal. This audit measures the discarded correlation
structure, effective rank, strong-correlation graph, duplicate-site context,
and the concentration of 1901-1920 HMA trend contributions.

No EnKF is rerun, no covariance estimator is selected, and no proxy or source
file is changed. The locked 1921-1939 period is not accessed.
