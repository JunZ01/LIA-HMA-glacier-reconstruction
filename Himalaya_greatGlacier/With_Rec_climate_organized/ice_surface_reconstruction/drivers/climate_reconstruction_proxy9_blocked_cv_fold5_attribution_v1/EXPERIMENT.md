# Proxy9 Blocked-CV Fold-5 Attribution v1

This no-OGGM experiment attributes the failed 1981-2000 temporal holdout from
the five-fold blocked calibration validation.

It reruns only the archived 1940-2000 operator and fold 5, whose PSMs were fit
on 1901-1980 and evaluated on the held-out 1981-2000 block. Proxy
observations, prior, samples, localization, archived RegEM-completed diagonal
error variances, HMA mask, and all source arrays remain frozen.

The output ranks each record by its change in HMA trend contribution over
1981-2000 and verifies that the sum of record-level contribution changes
closes the regional trend change. This is attribution, not permission to
delete records or modify coefficients.
