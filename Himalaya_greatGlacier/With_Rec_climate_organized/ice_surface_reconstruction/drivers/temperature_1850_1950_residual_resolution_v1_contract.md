# Temperature 1850-1950 Residual Resolution v1 Contract

This contract was created from the user's explicit instruction on 2026-08-23
because no pre-existing file with this name was found in the project tree,
`E:\3spyderProject`, or the user directory.

## Objective

Run a new, independent eastern-HMA residual diagnostic comparing Baiyu 2022
and Haizi 2024 with the current six-member production temperature and the
rejected `-0.350 degC` diagnostic comparator over allowed portions of
1850-1950.

## Frozen periods and safety

- Allowed historical intervals are 1850-1920 and 1940-1950 only.
- Years 1921-1939 must be rejected by year before any climate or anchor value
  is accessed.
- Smoothing is performed separately within each allowed contiguous interval
  and never crosses the sealed gap.
- Original climate, proxy, workbook, raster, MAT, NetCDF, CSV, and project
  files are read-only.
- The experiment writes to a new result directory and does not fit or apply a
  correction, modify production climate, run EnKF, or run OGGM.

## Comparison boundary

- Audit production seasonality before computing residuals.
- If production contains calendar-annual temperature only, Baiyu 2022
  June-July temperature is a seasonal direction/standardized-shape diagnostic
  only. It must not estimate or authorize an annual-temperature correction.
- Haizi 2024 previous-October to current-September mean temperature may serve
  as a near-annual local anchor. Its finite source region and spatial
  representativeness must be reported; it cannot identify an HMA-wide offset.
- The `-0.350 degC` branch is a previously rejected diagnostic comparator, not
  an accepted baseline or a parameter to refit.

## Required metrics and outputs

- Trend sign and slope for 1850-1920 and 1940-1950 when complete data permit.
- Standardized low-frequency shape metrics calculated with within-segment
  smoothing only.
- Absolute-amplitude residuals only for seasonally comparable evidence; any
  non-comparable field must be explicitly marked prohibited rather than
  silently calculated.
- Separate evidence classifications for direction, standardized shape,
  absolute amplitude, seasonality, spatial support, and independence.
- Allowed-period curves, metrics, source manifest, synthetic locked-gap test,
  and `FINAL_RESULT.md` stating which baseline is supported.
