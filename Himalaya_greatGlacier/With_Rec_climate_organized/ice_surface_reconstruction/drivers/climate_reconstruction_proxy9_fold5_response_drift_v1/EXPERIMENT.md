# Proxy9 Fold-5 Response-Drift Audit v1

This no-OGGM audit tests whether the failed 1981-2000 holdout reflects a
change in the observed proxy-to-local-climate relationship.

For every proxy9 record with sufficient held-out observations, it compares:

- the observed proxy trend and local annual temperature/precipitation trends;
- the archived and fold-5 PSM predictions evaluated on held-out observations;
- the trend in held-out prediction residuals;
- the record's change in HMA EnKF trend contribution.

All predictors and proxy observations are read from the same archived PHYDA
sources in MATLAB R2024b. No PSM, EnKF, source record, or error model is
modified. This is a diagnostic of temporal response stability only.
