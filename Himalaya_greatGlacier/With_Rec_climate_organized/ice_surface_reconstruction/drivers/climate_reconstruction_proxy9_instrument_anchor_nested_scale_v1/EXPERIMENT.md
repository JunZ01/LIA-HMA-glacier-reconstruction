# Proxy9 Instrument-Anchor Nested Error-Scale Test v1

This independent no-OGGM experiment calibrates the scalar HMA instrumental
observation error without using 1981-2000.

The proxy analysis is fixed to the fold-5 operator trained on 1901-1980 with
archived RegEM proxy error variances. The instrumental observation is the HMA
mean of three products, while the fourth product is validation-only. Within
1901-1980, four fixed 20-year temporal test blocks are crossed with the four
held-out products, yielding 16 validation cells.

For each cell, the base instrumental error variance is the mean squared
difference between the three-product mean and held-out product in the other 60
years. Candidate multipliers are fixed at `0.125, 0.25, 0.5, 1, 2, 4`. The
multiplier is selected by minimum pooled centered RMSE over the 16 held-out
cells, with mean correlation as tie-breaker. The selection is frozen before
1981-2000 is read.

The outer test repeats leave-one-product-out assimilation in 1981-2000 using
the selected multiplier and base variance estimated from all 1901-1980 years.
Instrument observations remain absent before 1901, and the pre-instrumental
analysis must close to proxy-only. Products are not independent, so this is an
error-scale and architecture test, not final independent validation or a
production reconstruction. `rec_ref` is not used.
