# Proxy9 Nested Error-Weight Holdout Test v1

This no-OGGM experiment tests whether record-wise PSM reliability estimated
without seeing 1981-2000 can prevent the fold-5 modern holdout failure.

Within the fold-5 training interval 1901-1980, four contiguous 20-year inner
test blocks are used. Each record's archived annual bivariate/univariate PSM
type is fitted on the other 60 years and evaluated on the inner test block.
The pooled inner out-of-sample mean squared error defines a reliability
variance.

For EnKF, the tested diagonal variance is
`max(archived RegEM variance, nested out-of-sample MSE)`. This can only reduce
a record's weight relative to the archived error model; no record is removed
and no variance is reduced. The 1981-2000 outer holdout is not used in fitting,
weight construction, or branch selection.

A 2x2 control compares archived/fold-5 operators with archived/nested-inflated
error variances. Proxy observations, prior, samples, localization, HMA mask,
annual seasonality, model type, and all source data remain frozen. `rec_ref` is
not used.
