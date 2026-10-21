# Precipitation Instrumental Baseline V1

Date: 2026-08-13

## Purpose

Diagnose the existing production precipitation reconstruction against CRU TS
and the correctly identified GPCC-derived field without changing any
reconstruction parameter or source file.

## Frozen boundaries

- Production PSM calibration period: 1940-2000.
- Development out-of-sample period: 1901-1920.
- Locked final out-of-sample period: 1921-1939.
- Calibration-period consistency period: 1940-2000.
- GPCC-only early context: 1891-1900.
- The locked final period is not read into any metric, table, or figure in V1.
- `pr_obs_gpcp_05deg` is reported as GPCC-derived, never as GPCP.
- `rec_ref` is comparison-only.
- No OGGM, EnKF replay, tuning, method selection, or source modification.

## Questions

1. Is the 1901-1920 reconstruction error larger than CRU-GPCC disagreement?
2. Do all six production members share the same error, or is it method-specific?
3. Is the dominant mismatch in the HMA mean, spatial pattern, or local temporal
   variability?

## Outputs

The experiment writes only to a new independent result directory. It records
source hashes before and after, development metrics, calibration-consistency
metrics, a locked-period manifest, figures, and a decision summary.
