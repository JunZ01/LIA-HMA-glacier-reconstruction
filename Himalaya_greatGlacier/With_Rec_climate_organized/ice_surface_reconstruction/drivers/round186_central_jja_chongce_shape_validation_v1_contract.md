# Round 186 Central-JJA Chongce Shape Validation Contract

## Question

Does the coverage-only 14-record long core selected in Round 185 retain the
modern out-of-fold skill of the Round 156 central-JJA mechanism, and does its
untuned 1550-1899 low-frequency shape agree with independent Chongce evidence?

This test can validate a regional low-frequency shape mechanism. It cannot
validate absolute temperature amplitude, produce a spatial correction field,
or authorize a production reconstruction.

## Frozen candidate

- Predictor identities are exactly the 14 Round 185 long-core records. Their
  selection used coverage only, never proxy values or Chongce agreement.
- Modern predictor values are the Round 14 observation matrix for 1940-2000.
- Historical predictor values are the exactly matched LMR CSV columns.
- The CSV and MAT modern values must agree to `1e-12` on every common finite
  value or the run stops.
- Standardize each predictor using its full 1940-2000 mean and standard
  deviation, replace missing standardized values by zero, average records,
  and apply the Round 156 centered 11-year filter.
- Fit separate Berkeley Earth and CRU linear mappings over valid 1945-1995
  centers to the original 41-site central-band JJA targets.
- Apply those frozen modern mappings to 1550-1899. Export only seven 50-year
  means, not annual proxy values or an annual candidate trajectory.

## Modern transport gates

Using the exact three purged Round 156 blocks for the 14-record subset, both
Berkeley Earth and CRU must satisfy:

- pooled OOF correlation at least `0.30`;
- absolute skill over the training-mean baseline at least `0.05`;
- positive fitted proxy slope in all three folds.

## Independent paleo-shape gates

The seven fixed windows are 1550-1599 through 1850-1899. Chongce is not used
for fitting or scaling. Because Chongce is local annual temperature and the
candidate is regional JJA temperature, only offset- and scale-invariant shape
metrics are admissible.

- Both product candidates must correlate with Chongce by at least `+0.10`.
- Both must improve on the existing project-Chongce correlation by at least
  `+0.20`.
- Berkeley and CRU candidate windows must correlate by at least `0.80`.
- Each candidate must agree with Chongce on at least four of six adjacent
  50-year change directions.
- Both full-period modern calibration slopes must be positive.

All modern and paleo gates must pass to advance the mechanism. Passing still
does not authorize amplitude correction or production use.

## Locked and source boundaries

- Decode the CSV year first and skip every 1921-1939 row before converting any
  selected proxy token.
- Use only the previously exported 1550-1899 Chongce window table; do not open
  the Chongce workbook or Jionglacuo.
- Do not run EnKF, a production correction, or OGGM.
- Preserve and verify all input fingerprints.

