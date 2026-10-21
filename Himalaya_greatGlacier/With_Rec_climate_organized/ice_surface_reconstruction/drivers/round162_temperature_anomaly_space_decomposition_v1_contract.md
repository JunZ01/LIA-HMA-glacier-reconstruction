# Round 162 Temperature Anomaly-Space Decomposition Contract

## Objective

Explain the Round 161 perfect-model amplitude loss by separating three mechanisms:

1. finite 500-member prior-span limitation;
2. full-period PSM-operator observability limitation;
3. attenuation from archived RegEM residual variance and adaptive member selection.

## Frozen inputs

- The archived 848-member temperature prior is read-only.
- The experiment-1 500-member reconstruction pool is unchanged.
- The 24 held-out truth members are copied exactly from Round 161 v2.
- Proxy6 and proxy9 use the same full-period PSM coefficients, predictor matrices,
  active-row rules, and archived RegEM residual variances as Round 161 v2.
- The native 14 x 24 HMA grid, masks, and cosine-latitude weights are unchanged.
- No actual proxy observations, instrumental values, 1901-1939 values, production
  EnKF run, OGGM run, or Jionglacuo data are permitted.

## Diagnostics

All state diagnostics use anomalies relative to the fixed all-500 background mean.

1. Compute weighted prior singular spectra and participation effective rank for HMA
   and each frozen subregion.
2. Project each held-out truth anomaly onto the weighted 500-pool anomaly column
   space and report representable and orthogonal variance fractions.
3. Reproduce noise-free ordinary and adaptive analyses from Round 161 v2.
4. Compute a diagnostic `R -> 0` least-squares operator limit with the same PSM rows.
   It is a theoretical observability bound only and cannot become a production
   parameter or candidate reconstruction.
5. For every truth, network, region, and method report anomaly amplitude slope,
   norm ratio, cosine similarity, and residual norm relative to both full truth and
   the pool-representable truth component.
6. Report proxy-space innovation representability for the `R -> 0` solution.

## Interpretation

- Low pool representability identifies a prior-span limitation.
- High pool representability but weak `R -> 0` recovery identifies a PSM-operator
  observability limitation.
- Material improvement from ordinary to `R -> 0` identifies residual-variance
  shrinkage.
- Worse adaptive than ordinary recovery identifies selection-induced attenuation.
- Multiple mechanisms may be reported; no tuning is performed in this round.

## Outputs

The independent result directory must contain the contract, source manifest,
prior spectrum table, truth decomposition, method decomposition, summary table,
JSON summary, final result note, and a scientific diagnostic figure.

