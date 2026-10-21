# Proxy9 Calibration-Window Group Interaction v1

This no-OGGM experiment runs all 32 subsets of the five mutually exclusive
proxy groups defined by the grouped calibration-window attribution v2.

For each subset, only those groups receive the delta-anchored 1920-1980 PSM
operator; all remaining records retain the archived 1940-2000 operator. Proxy
observations, prior, samples, localization, archived RegEM-completed diagonal
error variances, model type, seasonality, HMA mask, and instrumental products
are fixed.

The complete factorial permits an exact inclusion-exclusion decomposition of
the HMA annual curve into group main effects and interactions. `rec_ref` is
comparison-only. The experiment diagnoses network behavior and does not select
a production operator, delete records, constrain coefficients, or combine the
residual-missingness repair.
