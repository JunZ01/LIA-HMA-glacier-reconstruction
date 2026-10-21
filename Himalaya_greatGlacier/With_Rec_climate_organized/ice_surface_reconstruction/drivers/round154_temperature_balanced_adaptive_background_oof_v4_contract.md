# Round 154 Contract: Spatially Balanced Adaptive Temperature Background

## Decision

Test one predeclared repair to the AO/HGAO strategy-C analogue selector: replace the proxy-count-weighted global pattern correlation with an equal-longitude-band mean of correlations calculated separately for west HMA (`lon < 80 E`), central HMA (`80 <= lon < 90 E`), and east HMA (`lon >= 90 E`).

## Frozen inputs and branches

- Years: 1940-2000 only for the modern gate, in held-out blocks 1940-1959, 1960-1979, and 1980-2000.
- PSMs: the already generated fold-native proxy6 and proxy9 operators. Each held-out block is excluded from its PSM fit.
- Prior: the archived 848-member temperature prior and all 50 archived 500-member experiment pools; 100 members are selected per pool and year.
- Proxy support: only records with a finite held-out observation, finite positive fold-native residual variance, and finite model estimates for all prior members.
- Candidate: within each proxy network and year, calculate member-wise Pearson correlation inside each longitude band. A band contributes only with at least three active records and nonzero observed variance. Average available band correlations with equal weight and select the top 100 members in each pool.
- Controls: the archived global-correlation selector reconstructed with the same fold-native PSMs, and the fixed 500-member-pool background.
- Products: CRU, GISTEMP, HadCRUT5, and NOAA are primary. Berkeley Earth is secondary because it enters PSM fitting.
- Regions: HMA and the same three longitude bands.
- Frequencies: annual and within-block-centered 11-year low-pass components.

There is no longitude weight, member count, threshold, proxy subset, or score transformation to tune after seeing results.

## Frozen promotion gates

The candidate passes only if all conditions hold for the primary-product, low-pass, 50-pool comparison against the current global selector:

1. Median RMSE improvement across region-product-fold pairs is at least 5%.
2. At least 60% of pairs improve.
3. The eastern-HMA median improvement is positive.
4. No longitude region has median regression worse than 5%.
5. At least two of three held-out blocks have positive median improvement, and no block regresses by more than 5%.
6. At least three of four primary products have positive median improvement.
7. The experiment-1-only median improvement has the same positive direction.

Passing authorizes an immediate allowed-period 1901-1920 trajectory comparison using the unchanged archived full-period PSM and all 50 pools. Failure closes this balanced selector and forbids a paleo candidate from this experiment.

## Boundaries

- Never read or decode 1921-1939.
- Do not access Jionglacuo.
- Do not modify source MAT, NetCDF, CSV, MATLAB, or prior files.
- Do not run a production EnKF, change gain/R, delete proxies, fit longitude weights, or run OGGM.
- This is an operator/background diagnostic, not a production climate reconstruction.
