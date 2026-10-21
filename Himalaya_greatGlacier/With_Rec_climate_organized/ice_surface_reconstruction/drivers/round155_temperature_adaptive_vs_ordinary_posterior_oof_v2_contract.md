# Round 155 Contract: Adaptive AO Versus Ordinary-EnKF Posterior OOF

## Question

When the same fold-native PSM and RegEM residual variance are used, does removing strategy-C analogue selection improve the complete temperature posterior rather than only its background?

## Frozen comparison

- Years: 1940-2000 only.
- Outer blocks: 1940-1951, 1952-1963, 1964-1975, 1976-1987, and 1988-2000.
- Networks: archived proxy6 and proxy9, evaluated separately and as an equal network mean.
- Prior: archived temperature experiment-1 pool of 500 unique members.
- Adaptive control: current global strategy-C pattern correlation, top 100 of the same 500 members each year, followed by the unchanged fold-native EnKF update.
- Candidate: ordinary EnKF using all 500 experiment-1 members and the same fold-native PSM, observations, and residual variance. Reuse the verified v5 `full_increment_alpha1` OOF posterior and require regional-curve closure.
- Products: Berkeley Earth, CRU, GISTEMP, HadCRUT5, and NOAA.
- Regions: HMA, west `<80E`, central `80-90E`, and east `>=90E`.
- No alpha shrinkage, localization, proxy filtering, spatial weight, R change, or amplitude adjustment is allowed.

## Frozen gates

Ordinary EnKF advances to a 50-pool stability test only if all conditions hold relative to adaptive AO:

1. Median region-product RMSE improvement is at least 5%.
2. No region-product RMSE regresses by more than 5%.
3. At least 75% of block-region-product comparisons improve.
4. Median annual-correlation change is no worse than -0.05.
5. Median ordinary-EnKF absolute skill versus zero is positive and median annual correlation is at least 0.30.
6. Proxy6 and proxy9 each have nonnegative median RMSE improvement.
7. The 1940-1969 and 1970-2000 median improvements are both nonnegative.
8. The reused ordinary-EnKF regional curves close within `1e-10 degC`, all five folds are complete, and all source fingerprints remain unchanged.

Failure closes fixed-background ordinary EnKF as the direct replacement for current adaptive AO. Passing authorizes only a separate 50-pool/HGAO-aware modern test, never a paleo trajectory or production reconstruction.

## Boundaries

- Do not read or decode 1901-1939.
- Do not access Jionglacuo.
- Do not modify source data or prior results.
- Do not run a production EnKF or OGGM.
