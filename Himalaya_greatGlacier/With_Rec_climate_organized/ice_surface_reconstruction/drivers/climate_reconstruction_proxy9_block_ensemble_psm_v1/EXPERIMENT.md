# Proxy9 Block-Ensemble PSM Transfer Test v1

This independent no-OGGM experiment tests whether the fold-5 failure is caused
by a single pooled 1901-1980 calibration operator or by deeper temporal
nonstationarity in proxy-climate response.

The 1901-1980 interval is divided into four fixed 20-year blocks. Each block
fits the archived annual bivariate/univariate PSM type independently. For each
inner held-out block, operators from the other three blocks are aggregated by
an equal-weight mean or a pointwise median and compared with the existing
pooled-complement OLS control. A local fit requires at least 20 observations
and full column rank. A record with fewer than two eligible local operators
falls back to the pooled operator for that fold; no proxy is deleted and no
pseudoinverse is used.

The production mean and median operators use all four 1901-1980 local blocks.
Selection among pooled OLS, block mean, and block median is frozen from the
stitched 1901-1980 cross-fit HMA centered RMSE, with correlation as the declared
tie-breaker. Only after selection is frozen are the predeclared production
branches evaluated on untouched 1981-2000 data. Archived RegEM error variance,
proxy observations, prior, samples, localization, annual predictors, and HMA
mask remain fixed. `rec_ref` is not used.
