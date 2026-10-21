# Round 271 Xinzhu NGS precipitation modern OOF contract

- Read only Xinzhu values for 1941-2000. Decode the year before the value and
  skip 1921-1939 before numeric conversion.
- Use the published previous-November through current-February precipitation
  target without testing alternative seasons.
- Extract the nearest GPCC and CRU monthly precipitation cells to Xinzhu
  (27.25 N, 99.43 E) for 1940-2000 and sum the four monthly totals.
- Use five fixed contiguous 12-year held-out blocks: 1941-1952, 1953-1964,
  1965-1976, 1977-1988, and 1989-2000.
- Within each product and fold, fit only intercept plus Xinzhu reconstruction.
  Score held-out correlation, RMSE, and skill against the training-fold mean.
- Advance only if both pooled OOF correlations are at least 0.30, both skills
  are positive, each product has at least four positive fold correlations, all
  ten fitted slopes are positive, both products have positive raw correlations
  in 1941-1970 and 1971-2000, and observed GPCC/CRU correlation is at least
  0.70.
- This is a modern transport and temporal-stability diagnostic. It does not
  establish paleo accuracy or authorize a production reconstruction, EnKF, or
  OGGM run.

