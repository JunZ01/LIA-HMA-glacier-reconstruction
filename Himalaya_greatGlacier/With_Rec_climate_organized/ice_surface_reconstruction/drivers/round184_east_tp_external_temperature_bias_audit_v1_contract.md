# Round 184 Contract: Eastern Tibetan Plateau External Temperature Bias Audit

## Purpose

Use the public 1210--2024 eastern Tibetan Plateau June--September temperature
reconstruction as external comparison evidence for the project's 1500--1899
warm bias. This record is not a PSM candidate and is not used for fitting.

## Frozen comparison

- External source: Mendeley Data `10.17632/s7t4zc34sr.2`, workbook column A
  year and column B reconstructed June--September temperature.
- Project source: the verified Round 23 corrected six-member HMA curve.
- Reference: the verified Round 23 four-family HMA central curve and family
  range.
- Common anomaly baseline: 1961--1990.
- Primary assessment: eight complete 50-year windows from 1500--1549 through
  1850--1899.
- Secondary allowed diagnostics: 1900--1920 and 1940--1989, reported separately
  and excluded from the primary decision.

## Locked-period parser

- Decode workbook column A (year) first.
- Reject rows dated 1921--1939 before decoding column B.
- Decode column B only for 1210--1920 and 1940--2000.
- Verify the Round 23 locked rows contain no numeric values.

## Decision rule

The external record supports the existing warm-bias diagnosis only if it is
colder than the project curve in at least six of eight primary windows and its
median absolute distance to the four-family central curve is smaller than its
median absolute distance to the project curve.

## Boundaries

- Spatial and seasonal target mismatch must be explicit: eastern Tibetan
  Plateau June--September versus HMA annual mean.
- Do not treat agreement as field validation or use the record to estimate a
  correction.
- Do not access Jionglacuo, run EnKF, modify production reconstruction, or run
  OGGM.
