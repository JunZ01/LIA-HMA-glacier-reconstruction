# Precipitation Proxy Instrumental PSM Audit V1

Date: 2026-08-13

## Purpose

Test whether the widespread 1901-1920 proxy decline in the precipitation
reconstruction is supported by local instrumental climate, and separate the
production bivariate PSM's temperature and precipitation terms.

Production `psm_ot=7` calls `psm_uni` with Berkeley Earth temperature and the
field locally labeled GPCP but proven to be GPCC-derived precipitation. For the
201 tree records, the archived formula is:

`proxy = b0 + b1 * temperature + b2 * precipitation`.

Only proxies inside the combined 0.5-degree grid with finite 1901-1920
Berkeley Earth and GPCC-derived data are evaluated. Out-of-grid or incomplete
records are reported as unassessed. The 1921-1939 final period is not read.

This is a read-only mechanism audit. It does not refit PSMs, remove proxies,
change covariance, or rerun EnKF.
