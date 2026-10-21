# Round 185 Central-JJA Historical Support Preflight Contract

## Purpose

Determine whether the 41-record `central_80_90` JJA network from Round 156 has
enough fixed-record historical support for a prospective, independent
low-frequency comparison against Chongce over 1550-1899.

This is a coverage and identity audit only. It does not estimate a climate
curve, fit a PSM, inspect proxy amplitudes, tune a correction, or authorize a
production reconstruction.

## Frozen inputs

- Round 13 `network_manifest.csv`.
- Original LMR `lmr_data.csv`.
- Central-JJA identity rule: `season_code == JJA` and `80 <= longitude < 90`.
- Expected selected count: 41 records, each matched exactly by
  `formal_identity` to one CSV header.

## Locked-period rule

For every CSV row, decode the year field first. If the year is 1921-1939,
skip the complete row before converting any selected proxy token. The output
must contain only counts and coverage metadata, never proxy values.

## Coverage summaries

- Paleo audit interval: 1550-1899, split into seven fixed 50-year windows.
- Modern support interval: 1940-2000.
- A long-core record must have at least 40 finite years in every paleo window
  and at least 40 finite modern years.
- Report the full 41-record network and the coverage-only long-core subset.
- The historical branch is support-feasible only if at least three long-core
  records exist and every paleo window has at least three of them represented.

## Boundaries

- Do not open Jionglacuo.
- Do not output annual proxy values or a paleo trajectory.
- Do not access any 1921-1939 proxy value.
- Do not run EnKF, a production correction, or OGGM.
- Leave all inputs unchanged and record SHA256 fingerprints before and after.

