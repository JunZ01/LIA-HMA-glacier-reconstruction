# Round 154 Spatially Balanced Adaptive-Background OOF Contract

## Question

Does one fixed longitude-balanced strategy-C analog operator retain or improve
modern out-of-fold temperature skill relative to the archived global-pattern
operator?

## Frozen candidate

- Networks remain proxy6 and proxy9 and are evaluated separately before their
  production background contributions are combined.
- Longitude bands are fixed at west `<80E`, central `80--90E`, and east
  `>=90E`.
- In each band, center the active proxy observation pattern and each modeled
  proxy pattern, compute their cosine correlation, and average the three band
  correlations with equal one-third weights.
- Require at least three active proxies in every band. Do not delete, tune,
  reweight, or sign-flip individual proxies.
- Keep the archived 50 member pools, top `100/500` selection per pool, prior,
  proxy values, and all other strategy-C operations unchanged.
- The control is the original single global-pattern correlation rebuilt with
  the same fold-native PSM and the same active proxy mask.

## Modern-only validation

- Read only 1940--2000 proxy and instrumental values.
- Reuse the five frozen outer blocks: 1940--1951, 1952--1963, 1964--1975,
  1976--1987, and 1988--2000.
- For every held-out block, use only the existing R2024b fold-native PSM fitted
  outside that block. Reconstruct all 848 modeled proxy values from the saved
  coefficients and archived model predictors.
- Score HMA, west, central, and east regional curves against Berkeley Earth,
  CRU, GISTEMP, HadCRUT5, and NOAA. Center each field using that outer fold's
  training years.
- Report 1940--1969 and 1970--2000 OOF metrics separately. These are stability
  diagnostics, not additional fitted branches.

## Gates

The candidate advances to an allowed-period 1901--1920 trajectory test only if:

- median region-product RMSE improvement is at least 5%;
- no region-product RMSE worsens by more than 5%;
- at least 75% of block-region-product comparisons improve;
- median annual-correlation change is no worse than -0.05;
- median absolute skill versus a zero-anomaly prediction is positive and
  median annual correlation is at least 0.30;
- proxy6 and proxy9 each have nonnegative median RMSE improvement;
- 1940--1969 and 1970--2000 each have nonnegative median RMSE improvement;
- every longitude band has at least three active proxies in every evaluated
  year, the archived control closes, and all source fingerprints remain
  unchanged.

Failure closes this equal-band operator. No alternate band boundary, band
weight, proxy subset, or member-selection count may be tried under this
contract.

## Boundary

Do not read or decode 1901--1939, do not open Jionglacuo, do not run an EnKF
production reconstruction, and do not run OGGM. Passing authorizes only the
separate allowed-period trajectory diagnostic named above.
