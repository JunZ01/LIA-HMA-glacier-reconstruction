# Round 135 Withheld Proxy 15 Arbitration Contract

## Purpose

Use the NETP tree record excluded before Round 134 to determine whether the
preinstrumental low-frequency disagreement is closer to the Round 134
19-proxy composite or to the independent QT-Prec reconstruction.

## Frozen evidence

- Withheld record: proxy6 index 15, 37.93 N, 101.53 E.
- Public metadata identity: `Asi-GHEGAN.PAGES2k.2013`.
- Candidate reconstruction: Round 134 product-marginalized NETP MJJ curve.
- External reconstruction: QT-Prec, never used for fitting or orientation.
- Modern orientation targets: the same GPCC and CRU NETP MJJ means used in
  Round 134.

## Allowed years

- Proxy and instrumental orientation: 1940-2000 only.
- Paleo proxy values: 1000-1920 only.
- No value from 1921-1939 may be read or decoded.
- The frozen external validation reserve remains unopened.

## Fixed processing

1. Apply the Round 134 centered 11-year smoother separately to the withheld
   proxy and each modern MJJ target.
2. Orient the proxy from the sign of its 1940-2000 relation to each product.
   QT-Prec and the Round 134 paleo curve play no role in orientation.
3. Standardize the proxy with its own allowed modern smoothed mean and sample
   standard deviation.
4. Apply a centered 31-year mean to the paleo oriented proxy, matching the
   two-stage low-frequency treatment used for the Round 134 proxy composite.
5. Compare only common finite years for 1515-1900 and 1800-1900. No
   interpolation is allowed.

## Informativeness gate

The withheld proxy can arbitrate only if GPCC and CRU give the same full-period
orientation, the smaller absolute full-period correlation is at least 0.20,
and at least three of four within-block correlations (1940-1969 and
1970-2000, evaluated with smoothing confined to each block) have that sign.

## Arbitration rule

For each comparison period compute
`r(proxy15, QT-Prec) - r(proxy15, Round134 composite)`.

- Delta at least +0.10: favors QT-Prec.
- Delta at most -0.10: favors the Round 134 composite.
- Otherwise: nondiscriminating.

If both periods favor the same side, or the 1800-1900 conflict period favors
one side while the longer period is nondiscriminating, that side is supported.
Opposite period decisions are classified as mixed. Trend signs are reported
as corroborating diagnostics and do not override the frozen correlation rule.

## Scope

This is an independent diagnostic, not a reconstruction update. It does not
change proxy weights, seasons, smoothing windows, PSMs, EnKF settings, climate
files, or OGGM inputs.
