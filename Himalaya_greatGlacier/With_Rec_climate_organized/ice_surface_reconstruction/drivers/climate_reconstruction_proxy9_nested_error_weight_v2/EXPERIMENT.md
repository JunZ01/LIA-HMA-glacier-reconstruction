# Proxy9 Nested Error-Weight Holdout Test v2

This independent no-OGGM experiment supersedes v1 only for construction of
the nested-CV reliability variance. V1 files and outputs remain unchanged.

The scientific design remains frozen: four 20-year inner test blocks within
1901-1980 estimate record-wise out-of-sample PSM error, while 1981-2000 stays
untouched as the outer test. Proxy observations, annual predictor definition,
PSM type, prior, samples, localization, HMA mask, and source data are frozen.

V2 explicitly audits every inner-fold design matrix before regression. It
saves matrix rank, singular-value condition number, and exclusion reason. A
rank-deficient fold is not solved and does not enter pooled MSE. Each proxy
must retain at least three valid inner folds before its nested MSE is eligible
for error inflation. No pseudoinverse, proxy deletion, variance reduction, or
outer-test tuning is allowed.

After these gates pass, the tested variance remains
`max(archived RegEM variance, eligible nested-CV MSE)`. The later 2x2 outer
test compares archived/fold-5 operators with archived/nested-inflated error
variances. `rec_ref` is not used.
