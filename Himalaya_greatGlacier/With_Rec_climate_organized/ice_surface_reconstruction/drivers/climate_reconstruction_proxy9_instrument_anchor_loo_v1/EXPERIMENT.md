# Proxy9 Instrument-Anchor Leave-One-Product-Out Pilot v1

This independent no-OGGM pilot tests whether a separately declared modern
instrumental analysis layer can repair the proxy-only modern transfer failure
without changing pre-instrumental years.

The proxy component is frozen to the fold-5 annual operator trained on
1901-1980, with archived RegEM proxy error variances. A single additional
linear observation is the HMA area-weighted temperature mean. For each of four
instrumental products in turn, the other three products form the assimilated
annual HMA mean and the omitted product is validation-only. The scalar
instrument error variance is frozen from the 1901-1980 mean squared difference
between the three-product mean and the omitted product. No 1981-2000 omitted-
product data enter the error estimate.

The joint EnKF uses the prior covariance between the HMA mean and every state
grid cell, so the regional observation updates the spatial field coherently.
Instrument observations are absent before 1901; the joint analysis must close
exactly to the same fold-5 proxy-only analysis in those years. The untouched
1981-2000 omitted product tests transfer. Products are not assumed independent,
so this is an architecture and consistency pilot, not final validation or a
production reconstruction. `rec_ref` is not used.
