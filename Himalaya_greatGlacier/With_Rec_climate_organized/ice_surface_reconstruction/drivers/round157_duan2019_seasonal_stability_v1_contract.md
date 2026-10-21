# Round 157 Duan 2019 Seasonal Stability Contract

## Question

Does the published Duan et al. (2019) August-September temperature curve carry
stable modern seasonal information across the HMA, central HMA, and eastern HMA
when evaluated against Berkeley Earth and CRU?

## Frozen Design

- Use only the Round 156 allowed export and 1940-2000.
- Reuse the Round 116 99-cell common observed HMA support and its fixed regions:
  HMA, west `<80E`, central `80-90E`, and east `>=90E`.
- Extract August-September temperature at those cells from the existing Berkeley
  Earth and CRU monthly inputs using the established Round 118 readers.
- Fit one intercept and slope from the Duan curve to each regional target inside
  each of the five established held-out blocks. No smoothing, trend predictor,
  product weighting, region weighting, or hyperparameter is allowed.
- Primary regions are HMA, central, and east. West is an external-transfer
  diagnostic because the 17-site MXD network is concentrated in southeastern TP.

## Gates

For every primary region and both products: OOF correlation must be at least
0.30, skill versus training-fold climatology must be positive, at least four of
five held-out fold correlations must be positive, all five fitted slopes must be
positive, and both 1940-1969 and 1970-2000 correlations must be nonnegative.
Berkeley-versus-CRU OOF predictions must correlate at least 0.80 in each primary
region.

Passing releases the curve only as a source-derived seasonal operator candidate.
It remains neither a raw proxy nor independent final validation. Failure closes
this candidate. No 1901-1920 trajectory, 1921-1939 value, Jionglacuo value,
production EnKF, or OGGM run is allowed.
