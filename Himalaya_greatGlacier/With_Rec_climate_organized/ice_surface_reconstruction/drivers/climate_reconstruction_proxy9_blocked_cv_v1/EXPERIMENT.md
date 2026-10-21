# Proxy9 Five-Fold Blocked Calibration Validation v1

This no-OGGM experiment tests whether a 1901-2000 PSM calibration design has
out-of-sample temporal skill without selecting records or windows from the
final HMA reconstruction curve.

Five contiguous 20-year test blocks cover 1901-2000. For each fold, every PSM
is fitted on the remaining 80 years using its archived model type and annual
temperature/precipitation predictors. The fold-specific EnKF reconstruction is
used only in its held-out 20-year block. These blocks are stitched into a
cross-fitted 1901-2000 reconstruction for validation against instrumental
products.

A full 1901-2000 operator is also generated as a production hypothesis. Its
in-period fit is not an acceptance test; acceptance depends on the stitched
cross-fitted reconstruction and numerical/source-purity gates.

Proxy observations, prior, sample indices, localization, archived
RegEM-completed diagonal error variances, model type, seasonality, HMA mask,
and instrumental products are frozen. `rec_ref` remains comparison-only. The
residual-missingness repair is not combined in this experiment.
