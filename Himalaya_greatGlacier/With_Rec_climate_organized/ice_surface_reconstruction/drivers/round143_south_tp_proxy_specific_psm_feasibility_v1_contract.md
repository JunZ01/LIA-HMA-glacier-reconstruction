# Round 143 Southern Tibetan Plateau Proxy-Specific PSM Feasibility V1

## Objective

Run one final, bounded modern cross-validation test of whether proxy-specific reliability information can repair the southern Tibetan Plateau seasonal precipitation PSM after Round 142 diagnosed severe temporal instability.

## Frozen method

- Reuse exactly the Round 141 region, 38 proxies, 19 coordinate groups, two documented cross-year seasons, 30 grid cells, GPCC/CRU targets, years 1941-2000, and two 30-year blocked folds.
- Eleven-year smoothing remains confined independently to each fit and test period.
- Within each fold, standardize every proxy on fit years only. Its fit-period correlation with the target defines both orientation and a nonnegative reliability weight equal to the absolute correlation.
- Reliability weights are normalized only among records in the same 0.25-degree group. Spatial groups remain equal weight, so duplicate-rich sites cannot dominate.
- Fit one scalar regression from the reliability-weighted group composite to the target. Apply all centers, scales, signs, weights, and coefficients unchanged to the held-out period.
- No proxy is selected, removed, thresholded, or tuned. There is no regularization parameter and no season or boundary search.

## Frozen gates

- All eight season/product/fold test correlations must be positive.
- All four concatenated OOF correlations must be at least 0.50.
- All four OOF skills versus fit-fold climatology must be positive.
- GPCC-versus-CRU OOF prediction correlation must be at least 0.80 for each season.

All gates must pass to permit paleo reconstruction and Biru/Jiali validation. Failure closes the current modern-correlation seasonal PSM family for this region; additional weighting, subset selection, boundary search, or model complexity is not authorized.

No 1921-1939 value, external validation value, frozen reserve, EnKF, or OGGM use is permitted. Source files remain read-only.
