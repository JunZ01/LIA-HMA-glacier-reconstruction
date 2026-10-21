# HMA Spatial Instrument-Anchor Nested Validation v1

This independent no-OGGM experiment selects a spatial instrumental error model
without reading 1981-2000.

The proxy branch is frozen to the proxy9 fold-5 operator trained on 1901-1980
with archived RegEM proxy variances. Spatial instrumental observations are the
86 HMA state-grid cells with complete four-product residual coverage in
1901-1980. Four held-out products are crossed with four held-out 20-year blocks.

For each validation cell, the three-product mean minus held-out-product
covariance is estimated from the other 60 years. Because the sample covariance
is rank deficient and spatial disagreement has effective rank about 4.59, the
tested covariance is

`scale * ((1 - shrinkage) * sample_covariance + shrinkage * diagonal)`.

Shrinkage candidates are `0.1, 0.25, 0.5, 0.75, 1.0`; scale candidates are
`0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0`. Selection minimizes pooled
grid-year RMSE over the 16 cells, with median grid temporal correlation as
tie-breaker. The choice is frozen before the 1981-2000 spatial outer test.

No product is treated as independent truth, no proxy is deleted, originals
remain read-only, and `rec_ref` is not used.
