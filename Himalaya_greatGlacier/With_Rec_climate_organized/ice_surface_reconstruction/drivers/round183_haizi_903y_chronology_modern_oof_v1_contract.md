# Round 183 Contract: Haizi 903-Year Chronology Modern OOF

## Purpose

Test whether the new Haizi Mountain `Juniperus saltuaria` standard chronology
is non-aliased and transfers to gridded mean annual temperature in blocked
modern validation.

## Frozen candidate and target

- Predictor: supplement `Sheet1` column B (`std`) only.
- Published reconstructed temperature in column C is lineage-only and must not
  be an OOF predictor.
- Column D is sample depth only.
- Target: annual mean temperature at `29.41 N, 100.13 E`.
- Target years: 1940--1999.
- Five blocked 12-year folds: 1940--1951, 1952--1963, 1964--1975,
  1976--1987, and 1988--1999.

## Locked-period parser

- Decode column A (year) first.
- Reject 1921--1939 rows before reading B, C, or D.
- Read B, C, and D only for 1940--2000.
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
- Do not access locked-period or Jionglacuo values.
- Do not tune season, lag, sign, location, folds, or gate.
- Do not run EnKF, correction, production reconstruction, or OGGM.
