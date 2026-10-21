# Round 140 East-HMA External Identity and Role Closure

## Purpose

Determine whether the safely exported Biru/Jiali precipitation and streamflow
series are aliases of the current proxy6 network and assign their admissible
scientific roles before any east-HMA reconstruction is fit.

## Frozen inputs

- Round 139 V2 allowed-value CSVs only.
- Proxy6 values requested only for 1000-1920 and 1940-2000.
- Existing proxy6 metadata and source-family mapping.

## Identity tests

For every external series and every proxy6 record with at least 20 common
allowed years, calculate correlation and the residual after an affine mapping.
An exact sequence alias requires absolute correlation at least 0.999999 and
maximum affine residual no larger than 1e-8 in external-series units.

For the direct precipitation reconstructions also calculate distance from the
published source tree sites to the nearest proxy6 coordinate. Site independence
requires no proxy6 coordinate within 1 degree and no exact sequence alias.

## Frozen roles

- Biru previous-September to current-June precipitation and Jiali
  previous-July to current-April precipitation may become validation-only
  benchmarks if site independence passes.
- The same-workbook streamflow curve is secondary hydroclimate evidence, not an
  independent precipitation target.
- The 1000-2018 three-rivers streamflow reconstruction is a secondary
  hydroclimate phase benchmark with source-network independence unresolved. It
  cannot train or directly validate precipitation amplitude.
- Sparse helper/chart columns in the three-rivers sheet are excluded after
  confirming they contain no annual series.

No external series may influence proxy selection, season selection, weights,
or calibration. No 1921-1939 value or frozen-reserve value may be read. No EnKF
or OGGM run is authorized.
