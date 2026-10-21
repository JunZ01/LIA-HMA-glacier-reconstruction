# Proxy9 Forward-Memory PSM Transfer Test v1

This independent no-OGGM experiment tests whether emphasizing recent proxy-
climate relationships improves genuinely forward transfer.

Three internal forecasts are fixed in advance:

- train 1901-1920, test 1921-1940;
- train 1901-1940, test 1941-1960;
- train 1901-1960, test 1961-1980.

Candidate annual PSMs are expanding ordinary least squares and exponentially
weighted least squares with fixed 10-, 20-, and 40-year memory. Exponential
weights depend only on distance before the training endpoint. Predictor bias
correction uses the same weights as the regression. Every fit requires at
least 20 actual observations and full weighted-design rank. A failed weighted
fit falls back to the static expanding fit for that record and forecast; a
failed static fit retains the archived operator. No proxy is deleted and no
pseudoinverse or coefficient-sign constraint is used.

Method selection uses only the three stitched 1921-1980 forward forecasts,
minimizing fold-centered HMA RMSE with correlation as tie-breaker. Production
operators are then fit through 1980. The selected method is frozen before the
untouched 1981-2000 outer test is read. Archived RegEM error variance, proxy
observations, prior, samples, localization, annual predictors, and HMA mask
remain fixed. `rec_ref` is not used.
