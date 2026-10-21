# Round 232 Contract

## Question

Does the source-native Haizi STD chronology pass blocked modern OOF when the
instrumental target is constructed exactly as documented in Xu et al. (2024)?

## Correction to the superseded diagnostic

Round 183 used calendar-year gridded temperature at 29.41 N, 100.13 E. The
paper documents Daocheng station at 29.03 N, 100.18 E and reconstructs mean
temperature from previous October through current September. Round 232 changes
only those two target definitions and uses a new independent result directory.

## Frozen design

- Predictor: source-native Haizi STD chronology, column B of the official
  supplementary workbook.
- Target products: Berkeley Earth and CRU TS4.04 at the nearest grid cell to
  Daocheng station.
- Target season: previous October through current September.
- Evaluation years: 1941-2000. The 1940 target is excluded because its previous
  October-December months would fall in the sealed interval.
- Five blocked folds: 1941-1952, 1953-1964, 1965-1976, 1977-1988, 1989-2000.
- Each training fold fits only an intercept and chronology slope.
- Baseline: the training-fold target climatology.

## Frozen admission gates

All must pass for advancement:

1. OOF correlation is at least 0.30 for both products.
2. OOF skill against training-fold climatology is positive for both products.
3. At least four of five fold correlations are positive for both products.
4. All ten fitted slopes are positive.
5. Chronology-target correlations are positive in both 1941-1970 and
   1971-2000 for both products.
6. Berkeley and CRU OOF predictions correlate at least 0.80.

## Safety

The parser decodes year first. Values outside 1940-2000, including every
1921-1939 value, are skipped before chronology-cell conversion. No paleo
trajectory is exported or inspected. No Jionglacuo, EnKF, production climate,
or OGGM access is permitted.

