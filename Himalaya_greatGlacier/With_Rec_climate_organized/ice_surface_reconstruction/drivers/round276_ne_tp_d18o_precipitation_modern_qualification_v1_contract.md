# Round 276 northeastern TP d18O precipitation qualification contract

- Read only 1940-2000 values from the official 2025 Source Data workbook,
  skipping 1921-1939 before precipitation-value access.
- First compare the candidate against every frozen proxy6/proxy9 modern record
  with the Round-273 exact and near-alias definitions.
- Use the published previous-September through current-August precipitation
  season without testing alternatives.
- Test transport to GPCC and CRU over the fixed 35.5-38.0 N, 96.5-99.0 E box,
  which encloses the four published sampling areas.
- Evaluate 1941-2000 in the five fixed Round-271 12-year blocked folds.
- Advance to historical NETP validation only if there are zero aliases, both
  product OOF correlations are at least 0.30, both skills are positive, both
  products have at least four positive fold correlations, all ten slopes are
  positive, both early and late raw correlations are positive, and GPCC/CRU
  observed target correlation is at least 0.70.
- This candidate is inside existing NETP spatial support. Passing cannot by
  itself authorize a second spatial mode, production reconstruction, EnKF, or
  OGGM run.

