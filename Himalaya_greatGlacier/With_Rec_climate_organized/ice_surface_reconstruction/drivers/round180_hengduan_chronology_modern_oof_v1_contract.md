# Round 180 Contract: Hengduan Chronology Modern OOF

## Purpose

Test whether the new Mendeley `889n6ccx4z` regional tree-ring chronology is
non-aliased and has transferable modern warm-season temperature skill against
both Berkeley Earth and CRU.

## Frozen candidate and target

- Predictor: published `SsfStb` regional chronology only.
- The published reconstructed temperature is lineage-only and must never be an
  OOF predictor.
- Target: mean May--September temperature across the five published sampling
  sites DLL, FGH, KQH, XYS, and YPL.
- Target years: 1940--1999.
- Five blocked 12-year folds: 1940--1951, 1952--1963, 1964--1975,
  1976--1987, and 1988--1999.

## Locked-period parser

- Decode column A (year) first.
- Reject 1921--1939 rows before reading column B.
- Read column B only for 1940--2000.
- Do not decode or inspect any paleo value outside the modern window.

## Gates

- No exact affine or `abs(r) >= 0.995` alias against proxy6/proxy9.
- OOF correlation at least 0.30 for both Berkeley Earth and CRU.
- Positive skill against training-fold climatology for both products.
- At least four of five fold correlations positive for both products.
- All ten fitted slopes positive.
- Positive early and late split correlations for both products.
- Berkeley/CRU OOF predictions correlate at least 0.80.

## Hard boundaries

- Do not inspect the paleo trajectory.
- Do not access any locked-period value or Jionglacuo value.
- Do not tune season, lag, sign, site weights, detrending, folds, or gate.
- Do not run EnKF, correction, production reconstruction, or OGGM.
