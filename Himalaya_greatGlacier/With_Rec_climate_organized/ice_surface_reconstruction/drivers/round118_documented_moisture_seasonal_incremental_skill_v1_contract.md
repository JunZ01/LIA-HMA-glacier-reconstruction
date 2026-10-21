# Round 118 Documented-Moisture Seasonal Incremental-Skill Contract

## Question

Do the 25 documented-moisture proxy9 evidence groups contain transferable
summer-precipitation information beyond temperature when each record is tested
on its publication-defined season?

## Frozen Design

- Years: 1940-2000; outer blocks are the five Round 113 blocks.
- Proxy selection is identical to Round 117 and is independent of outcomes.
- Season is fixed from Round 110 metadata: JJA maps to June-August; Jun-Sep and
  JJAS_boreal_summer map to June-September. No season search is allowed.
- Product paths: Berkeley temperature + GPCC precipitation and CRU temperature
  + CRU precipitation. Fit temperature-only and temperature-plus-precipitation
  OLS models in every outer training set, then predict the held-out block.
- Also transfer each fitted model to the alternate product predictors without
  refitting. All proxy and climate values remain in their native physical or
  standardized units; only a 1961-1990 climate anomaly center is applied.

## Frozen Gates

- Median bivariate RMSE improvement over temperature-only is at least 1% in
  each of the four self/cross-product paths.
- At least 50% of records improve in all four paths and have positive
  self-product bivariate correlations.
- The fitted precipitation coefficient has the same sign across the two
  products in at least 70% of comparable outer fits.
- Median bidirectional product-migration loss is no more than 10%.

These thresholds reuse the project Round 05/06 seasonal-network boundaries.
A complete pass authorizes a separately contracted season-aware field method.
Failure closes the present tree/proxy9 precipitation-operator route. No
1901-1920 or 1921-1939 value and no paleo reserve value is accessed; original
sources are read-only; OGGM remains prohibited.
