# Round 136 NETP Spatial-Group Leave-One-Out Contract

## Purpose

Test whether the Round 134 NETP low-frequency trajectory is distributed across
the proxy network or depends on one spatial group.

## Frozen design

- Use the same 19 training proxies, 15 rounded 0.25-degree spatial groups,
  GPCC/CRU NETP MJJ targets, 11-year proxy smoothing, product-specific modern
  orientation, scalar calibration, and 31-year paleo smoothing as Round 134.
- Reproduce the full Round 134 curves first.
- Create exactly 15 branches, each omitting one complete spatial group.
- Refit only the already defined scalar calibration after an omission. Do not
  change seasons, periods, smoothing, signs, weights, proxy membership, or
  target products.
- Proxy 15 and QT-Prec remain validation-only and cannot affect a branch.

## Allowed years

- Proxy paleo values: 1000-1920 only.
- Proxy and instrumental calibration: 1940-2000 only.
- No 1921-1939 value and no frozen external-reserve value may be read.

## Gates

The fixed temporal candidate is network-robust only if all conditions pass:

1. The reproduced full GPCC, CRU, and marginalized curves close against Round
   134 to a maximum absolute difference no larger than 1e-12.
2. All 15 branches complete, and every product-specific modern full-fit
   correlation is at least 0.70.
3. Every branch has correlation at least 0.90 with the full marginalized curve
   over both 1515-1900 and 1800-1900.
4. Every branch retains the positive full-curve trend over 1800-1900.
5. At least 13 of 15 branches remain closer to withheld proxy 15 than QT-Prec
   by the frozen 0.10 correlation-difference threshold over 1800-1900.

Passing advances the temporal candidate to method/uncertainty validation. It
does not yet authorize spatialization, EnKF assimilation, production climate
replacement, or OGGM.
