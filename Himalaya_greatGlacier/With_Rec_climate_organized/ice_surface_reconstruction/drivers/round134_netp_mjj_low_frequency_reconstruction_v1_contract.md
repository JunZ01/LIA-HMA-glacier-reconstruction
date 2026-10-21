# Round 134 NETP MJJ Low-Frequency Reconstruction V1

## Objective

Test one fixed regional-seasonal temporal reconstruction design before any spatial EnKF implementation. The target is the area-weighted May-July precipitation mean over 35-42 N, 92-108 E.

## Frozen Proxy Design

- Start from the 20 archived proxy6 tree records inside the NETP box.
- Exclude proxy index 15 (37.93 N, 101.53 E) from all fitting and reconstruction because it is nearest the independent QT-Prec Langya/Pingqiang sampling area. QT-Prec remains validation-only.
- Group the remaining 19 records on coordinates rounded to 0.25 degrees, yielding 15 spatial groups. Within-group and across-group averages have equal weight.
- Use only 1000-1920 and 1940-2000 proxy values. Never read proxy values from 1921-1939.
- Smooth proxy records and MJJ targets with a centered 11-year mean. In blocked OOF, smoothing is performed separately inside each fit/test block so no value crosses the 1969/1970 boundary.
- Inside each fold, standardize and sign-align each proxy using fit data only. Fit one scalar linear calibration from the equal-weight group composite to regional MJJ precipitation. No ridge parameter, proxy deletion, or weight tuning is allowed.

## Instrumental And Product Design

- Products: GPCC and CRU precipitation, aggregated over May-July and area-weighted across the same 32 NETP state coordinates.
- Calibration years: 1940-2000.
- Blocked OOF folds: fit 1940-1969/test 1970-2000 and fit 1970-2000/test 1940-1969.
- OOF climatology predictions use only the corresponding fit-target mean.

## Paleo And Independent Validation

- Fit each product branch on all allowed 1940-2000 data, reconstruct 1000-1920, and form a product-marginalized mean.
- Compare centered 31-year candidate curves with QT-Prec over 1500-1920 and 1800-1920.
- QT-Prec values, Round 133 metrics, and Round 133 baseline curves are read only after the model design is frozen. They are never used to choose proxy signs, groups, coefficients, windows, or products.
- Shenqi Cave is not used because it is outside the NETP target region.

## Advance Gates

All conditions must pass:

1. Both GPCC and CRU concatenated blocked-OOF correlations are at least 0.30 and both OOF skills relative to fold climatology are positive.
2. All four product-by-fold test correlations are positive.
3. GPCC and CRU OOF predictions correlate at least 0.80; their 31-year paleo reconstructions correlate at least 0.80 over 1500-1920.
4. At least five proxy spatial groups are active in every year from 1500-1920.
5. The product-marginalized QT-Prec correlation improves over the Round 133 baseline by at least 0.05 in 1500-1920, is not lower than baseline in 1800-1920, and has the correct 1800-1920 trend sign.
6. Neither individual product branch differs from the marginalized QT-Prec correlation by more than 0.10 in either validation period.

Passing authorizes a separate spatial MJJ reconstruction experiment using the exact Round 134 V2 prior. Failure closes this fixed equal-weight temporal-composite design. It does not authorize tuning. No final frozen reserve, EnKF, or OGGM run is allowed.

