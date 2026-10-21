# Round 115 Temperature Low-Rank OOF V2 Contract

V2 retains every scientific choice in the frozen v1 contract. V1 stopped
before fitting because six edge cells inside the 105-cell HMA state mask have
no value in any of the five instrumental products for any year in 1940-2000.

The technical support rule is now explicit and frozen before the v2 fit:

- Define observed HMA support as cells inside the established HMA mask where
  the unweighted five-product target is finite in every 1940-2000 year.
- Fit and score the candidate, equal-six legacy control, and zero-anomaly
  control on exactly this shared support.
- Do not fill the six all-product-missing cells, do not treat them as zero, and
  report their count as an unresolved spatial-coverage boundary.
- A later full 1000-2000 candidate, if authorized by a modern pass, must use a
  separately contracted spatial completion mechanism and may not call these
  six cells instrumentally validated.

All other frozen v1 choices remain unchanged: 226 documented-temperature
proxy9 evidence groups; five outer blocks; training-only area-weighted EOFs;
rank grid 1, 2, 3, 5, 8; ridge grid 0.1, 1, 10, 100, 1000; four inner blocked
validations; equal-six variable `tas_ensemble_mean_rec_result`; all Round 113
modern gates; no Round 114 external values; no 1921-1939 access; and no OGGM.
