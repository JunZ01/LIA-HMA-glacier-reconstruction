# Round 141 Southern Tibetan Plateau Cross-Year Precipitation Reconstruction V1

## Objective

Test whether the fixed regional low-frequency proxy-composite method that was robust in the NETP transfers to the southern Tibetan Plateau and independently reproduces the Biru and Jiali precipitation reconstructions.

## Frozen design

- Region: 27-35 N, 90-102 E.
- Proxy inventory: every proxy6 tree record inside the box with first observed year no later than 1500 and last observed year no earlier than 2000. The expected inventory is 38 records.
- Spatial weighting: records are grouped by coordinates rounded to 0.25 degrees; groups receive equal weight and records within a group receive equal weight.
- Seasons are fixed from the source metadata, not selected from results:
  - previous September through current June (`pSep_cJun`), validated only against Biru;
  - previous July through current April (`pJul_cApr`), validated only against Jiali.
- Modern targets: cosine-latitude-weighted GPCC and CRU precipitation totals over 30 fixed cells (latitudes 27, 29, 31, 33, 35 N; longitudes 91, 93, 95, 97, 99, 101 E).
- Calibration years: 1941-2000. The first cross-year season uses only months from 1940 and 1941; 1939 is never accessed.
- Blocked OOF folds: fit 1941-1970 and test 1971-2000, then reverse. Eleven-year smoothing is confined inside each fit/test block. Proxy standardization, sign orientation, and scalar regression are fit-only.
- Full reconstruction: fit 1941-2000, reconstruct 1000-1920, then apply a 31-year centered low-frequency mean.
- Biru and Jiali values are validation-only. They cannot affect proxy selection, season definition, sign, weight, scaling, calibration, or product marginalization.
- External validation uses only effective common years ending in 1900. Raw amplitude is diagnostic only because a point reconstruction and regional target have different spatial support; phase correlation and trend sign are formal.

## Frozen gates

- Every one of eight product/season/fold test correlations is positive.
- For every season/product pair, concatenated OOF correlation is at least 0.50 and OOF skill versus fold climatology is positive.
- GPCC-versus-CRU OOF prediction correlation is at least 0.80 for each season.
- GPCC-versus-CRU paleo 31-year curve correlation is at least 0.80 for each season.
- The product-marginalized matching branch has correlation at least 0.25 with its held-out external reconstruction over the full effective common period and at least 0.20 over 1800-1900.
- The model and matching external reconstruction have the same 1800-1900 trend sign.

All gates must pass to advance. Modern success with external failure closes the transfer. No season, boundary, proxy, weight, or threshold may be tuned after results are seen.

## Prohibitions

No value from 1921-1939 may be decoded or used. The frozen final validation reserve remains unopened. No source file may be modified. No EnKF or OGGM run is authorized.
