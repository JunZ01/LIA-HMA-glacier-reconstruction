# Round 188 Central-JJA 1600 Chongce Shape Contract

## Candidate

Freeze the earliest Round 187 coverage cohort that passed both modern OOF
product gates: the 30 records with at least 40 finite years in every 50-year
window from 1600 through 1899 and at least 40 finite years in 1940-2000.

The selection rule is fixed from coverage and modern OOF only. Chongce is not
used to select records, fit coefficients, choose season, choose sign, or scale
amplitude. This is a prospective fixed-rule validation, not a claim of a fully
unseen analyst-blind test.

## Mapping

- Preserve the Round 156 central-band 41-site JJA target.
- Standardize the 30 predictors using 1940-2000 statistics, mean-impute missing
  standardized values to zero, average, and apply the centered 11-year filter.
- Fit separate full-period Berkeley Earth and CRU mappings over valid
  1945-1995 centers.
- Apply without refitting to 1600-1899 and export only six 50-year means.

## Gates

- Round 187 modern OOF gates must already pass for both products.
- Both candidate-Chongce six-window correlations must be at least `+0.10`.
- Both must improve on the current project-Chongce six-window correlation by
  at least `+0.20`.
- Both must agree with Chongce on at least three of five adjacent changes.
- Berkeley and CRU candidates must correlate by at least `0.80`.
- Both full-period modern calibration slopes must be positive.

Passing supports only a regional JJA low-frequency shape mechanism for
1600-1899. It does not validate absolute amplitude, 1500-1599, a correction
field, or production use.

## Boundaries

- Decode CSV year first; skip 1921-1939 before proxy-token conversion.
- Use the existing allowed Chongce window table. Do not open its workbook or
  Jionglacuo.
- Do not run EnKF, a production correction, or OGGM.
- Preserve all source fingerprints.

