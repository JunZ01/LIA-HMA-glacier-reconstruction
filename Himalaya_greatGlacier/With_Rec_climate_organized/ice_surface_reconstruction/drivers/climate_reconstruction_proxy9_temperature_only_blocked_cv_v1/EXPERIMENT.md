# Proxy9 Temperature-Only Blocked Validation v1

This no-OGGM experiment tests whether the precipitation predictor in the 290
archived bivariate tree-ring PSMs causes temporal transfer failure.

The same five contiguous 20-year holdout blocks used in the approved blocked
validation are repeated. Records 1-290 are fitted with annual temperature-only
linear PSMs; records 291-297 remain annual temperature-only as in the archived
model. Each fold is trained on the other 80 years and used only in its held-out
20-year block. A full 1901-2000 temperature-only operator is retained only as a
production hypothesis.

Proxy observations, prior, sample indices, localization, archived
RegEM-completed diagonal error variances, HMA mask, annual seasonality, and
instrumental comparisons are frozen. The experiment changes only the inclusion
of precipitation in records 1-290. It does not select or delete proxies, force
coefficient signs, combine residual repairs, or run OGGM.
