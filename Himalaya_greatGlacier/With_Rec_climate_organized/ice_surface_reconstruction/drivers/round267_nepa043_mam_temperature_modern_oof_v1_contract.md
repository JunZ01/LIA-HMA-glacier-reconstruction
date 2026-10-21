# Round 267 NEPA043 MAM temperature modern OOF contract

## Scope

- Decode NEPA043 values only for 1940-2000 with the verified year-first RWL
  parser.
- NEPA043 is permanently excluded from the sealed 1921-1939 final test.
- Build one standard chronology using per-series ModNegEx detrending with a
  linear fallback, Tukey biweight averaging, and no prewhitening.
- Test the publication-defined same-year March-May temperature response at
  28.25 N, 85.38 E. The documented response direction is negative.
- Use monthly Berkeley Earth and CRU TS4.04 temperature as separate targets.
- This is evidence validation only. It is not EnKF, paleo reconstruction,
  climate correction, production, or OGGM.

## Cross-validation

- Target years: 1940-2000.
- Five contiguous held-out blocks: 1940-1951, 1952-1963, 1964-1975,
  1976-1987, and 1988-2000.
- Fit each product and block using all years outside the held-out block.
- Compare against the training-fold climatology, never a full-period target
  mean.

## Frozen gates

All gates must pass:

1. OOF correlation is at least 0.30 for both products.
2. OOF skill versus training-fold climatology is positive for both products.
3. At least four of five held-out block correlations are positive for each
   product.
4. All ten fitted chronology-to-temperature slopes are negative.
5. Raw chronology-target correlations are negative in both 1940-1969 and
   1970-2000 for both products.
6. Berkeley and CRU OOF predictions correlate at least 0.80.
7. Modern sample depth is at least 10 in every target year.

Passing admits NEPA043 only as a new modern-validated seasonal temperature
fitting candidate. Failure closes it as temperature fitting evidence. Neither
outcome authorizes production reconstruction or OGGM.
