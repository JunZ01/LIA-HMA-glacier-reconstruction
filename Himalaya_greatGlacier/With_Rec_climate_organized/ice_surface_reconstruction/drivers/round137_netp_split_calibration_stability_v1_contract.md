# Round 137 NETP Split-Calibration Stability Contract

## Purpose

Test whether the Round 134 paleo trajectory depends on calibrating the proxy
composite across the full 1940-2000 period.

## Frozen branches

Construct exactly four paleo branches:

- GPCC calibrated on 1940-1969.
- GPCC calibrated on 1970-2000.
- CRU calibrated on 1940-1969.
- CRU calibrated on 1970-2000.

Each branch uses only its fit block to smooth the modern values, standardize
proxies, choose proxy signs, and fit the one scalar regression. It then applies
those fixed parameters to the unchanged 1000-1920 proxy values. The 19 proxies,
15 equal-weight groups, MJJ target, 11-year and 31-year smoothers, and all other
Round 134 choices remain fixed.

## Validation-only evidence

The Round 134 full-calibration curves, withheld proxy 15, and QT-Prec are never
used to fit a split branch. They are read only after branch construction.

## Gates

All conditions must pass:

1. Every split branch has fit-period correlation at least 0.70.
2. Every split branch correlates at least 0.70 with its same-product Round 134
   full-calibration curve over both 1515-1900 and 1800-1900.
3. Early- versus late-calibrated branches for each product correlate at least
   0.70 over both periods.
4. All four branches retain a positive 1800-1900 trend.
5. All four branches remain closer to withheld proxy 15 than QT-Prec by the
   frozen 0.10 correlation-difference threshold over 1800-1900.

Passing advances the temporal candidate to structural uncertainty-envelope
construction. It does not authorize production replacement, spatialization,
EnKF, or OGGM.

No 1921-1939 value and no frozen external-reserve value may be read.
