# HMA Spatial Instrument-Anchor Nested Validation v2

V1 selected shrinkage `0.25` and the smallest tested scale `0.03125` using
only 1901-1980. The untouched 1981-2000 spatial outer test had not been read.

This independent v2 retains the same 86 HMA state cells, four product-by-four
time-block validation design, covariance estimator, shrinkage candidates, and
selection rule. It extends only the lower scale side to
`0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0`.
V1 files remain unchanged. `rec_ref` is not used.
