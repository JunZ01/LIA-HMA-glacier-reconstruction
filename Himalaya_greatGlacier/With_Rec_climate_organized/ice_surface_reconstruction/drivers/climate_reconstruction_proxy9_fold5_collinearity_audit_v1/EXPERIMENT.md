# Proxy9 Fold-5 Predictor Collinearity Audit v1

This no-OGGM audit tests whether temperature/precipitation predictor
collinearity explains the widespread coefficient changes and 1981-2000
holdout failure of the bivariate proxy9 PSMs.

For records 1-290, it extracts the exact annual Berkeley Earth temperature and
GPCP precipitation-anomaly predictors used by the archived PSM at each proxy
location. Metrics are computed for the fold-5 training period 1901-1980 and
for the archived 1940-2000 calibration period: predictor correlation, VIF,
standardized design-matrix condition number, effective sample count, and
temperature/precipitation standard deviation.

The audit changes no PSM, EnKF, proxy, or source file. It is diagnostic only.
