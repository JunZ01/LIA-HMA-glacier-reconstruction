# Round 192 Contract: East-HMA Dual-Chronology Thermal Index OOF

## Purpose

Test whether two individually weak but physically temperature-sensitive eastern
HMA chronologies jointly define a transferable regional thermal-state index.
This is a mechanism-candidate test, not field calibration or independent paleo
validation.

## Frozen predictor

- Hengduan published standard chronology from Round 180.
- Haizi published standard chronology from Round 183.
- Within every training fold, standardize each chronology using training-only
  mean and sample standard deviation, then average them with weights 0.5/0.5.
- Do not tune weights, signs, lags, seasons, sites, detrending, or folds.

## Frozen target

- Hengduan component: the already-audited five-site May--September mean
  temperature target.
- Haizi component: the already-audited local annual-mean temperature target.
- For each instrumental product and fold, standardize each target component
  using training-only statistics, then average them with weights 0.5/0.5.
- The result is dimensionless and must not be called a degrees-C temperature
  field.

## Validation

- Years 1940--1999.
- Five blocked 12-year folds: 1940--1951, 1952--1963, 1964--1975,
  1976--1987, and 1988--1999.
- Apply the unchanged Round 180/183 gates: OOF correlation at least 0.30 and
  positive climatology skill for both Berkeley Earth and CRU; at least four of
  five positive fold correlations per product; all fitted slopes positive;
  positive early/late split correlations; cross-product OOF prediction
  correlation at least 0.80.

## Hard boundaries

- Use only the modern, locked-safe derived series exported by Rounds 180/183.
- Do not inspect any paleo chronology or reconstruction value.
- Do not access or hash Jionglacuo or decode 1921--1939 values.
- A passing result permits a separately contracted historical-shape test only;
  it does not authorize EnKF, correction, production reconstruction, or OGGM.

