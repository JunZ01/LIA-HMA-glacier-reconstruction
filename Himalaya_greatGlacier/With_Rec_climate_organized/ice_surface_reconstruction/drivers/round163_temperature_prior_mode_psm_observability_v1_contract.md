# Round 163 Temperature Prior-Mode PSM Observability Contract

## Objective

Map every weighted prior temperature mode to the residual-standardized full-period
PSM observation space. Determine whether the low-dimensional observation-coupled
boundary in canonical Round 162 is caused primarily by structural operator
blindness or by the archived diagonal RegEM residual weighting.

## Frozen inputs

- Reuse the canonical Round 161/162 archived 848-member prior, experiment-1
  500-member pool, 24 disjoint held-out truth members, native 14 x 24 HMA mask,
  cosine-latitude weights, full-period proxy6/proxy9 PSMs, and RegEM variances.
- Use proxy6, proxy9, and their fixed concatenated joint network. Do not select,
  delete, reweight, or refit proxies, modes, regions, members, or PSM coefficients.
- No actual proxy observation, instrumental value, 1901-1939 value, Jionglacuo
  value, production EnKF run, or OGGM run is permitted.

## Diagnostics

For HMA and the frozen west, central, and east regions:

1. Compute the weighted prior SVD and retain its complete numerical-rank basis.
2. Standardize proxy ensemble anomalies by the archived diagonal RegEM residual
   standard deviations.
3. For every prior mode, compute the linear-Gaussian posterior variance reduction
   with archived `R` and the pseudoinverse `R -> 0` structural upper bound.
4. Report prior variance share, cumulative variance, cross-covariance energy, and
   both variance-reduction curves for proxy6, proxy9, and the joint network.
5. Project each of the 24 held-out truth anomalies onto the prior modes and report
   actual-`R` and `R -> 0` observability-weighted resolved-energy fractions.
6. Report proxy-network complementarity and the prior-variance fraction carried by
   structurally blind modes (`R -> 0` reduction below 0.20).

## Frozen interpretation

- If joint-network `R -> 0` prior-variance-weighted reduction is below 0.50,
  structural PSM blindness is primary.
- Otherwise, if archived-`R` reduction is below 60% of the `R -> 0` reduction,
  residual-standardization attenuation is primary.
- Both mechanisms may be material. This round does not tune `R`; any residual-model
  continuation must first use modern OOF data under a separate contract and must be
  mode-aware rather than a scalar gain correction.

## Outputs

Write an independent result directory containing the contract, mode table,
held-out-truth table, summaries, JSON decision, scientific curves, final result,
and unchanged-source manifest.

