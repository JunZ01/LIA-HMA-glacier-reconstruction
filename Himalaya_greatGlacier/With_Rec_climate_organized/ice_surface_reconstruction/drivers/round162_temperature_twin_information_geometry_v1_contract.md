# Round 162 Temperature Twin Information Geometry Contract

## Question

Why did the Round 161 perfect-model temperature twin recover only about 28% of HMA state RMSE even though the pseudo-observations exactly obey the archived PSM?

## Frozen inputs

- Reuse the Round 161 experiment-1 500-member reconstruction pool.
- Reuse the same 24 held-out truth members, full-branch proxy operators, RegEM residual variances, seed, and five noise replicates.
- Reuse the archived prior and project HMA/native-grid masks without modification.
- Do not use actual proxy observations, instrumental target values, 1921-1939 values, Jionglacuo data, production EnKF, or OGGM.

## Diagnostic decomposition

For the ordinary equal-network update, and secondarily each proxy network:

1. Measure prior anomaly numerical rank, participation-ratio effective rank, and the ranks containing 50%, 80%, 90%, 95%, and 99% of weighted variance.
2. Project each held-out truth-minus-background anomaly onto the complete weighted prior anomaly span and its 99%-variance practical span.
3. Build the observation-coupled state subspace from residual-standardized state/proxy cross covariance. Project each truth anomaly onto its complete and 99%-energy spans.
4. Measure the actual update/truth norm ratio, cosine alignment, and the best result obtainable by multiplying the actual update by one oracle scalar.

## Interpretation

- The complete prior-span projection is the upper bound imposed by the sampled prior state space.
- The observation-coupled projection is an optimistic upper bound for the state directions visible to the selected proxy operator.
- Oracle scalar rescaling tests whether the actual update direction is useful but attenuated.
- These are perfect-model mechanism diagnostics, not validation scores and not permission to tune against instrumental truth.

## Required outputs

- `prior_rank_metrics.csv`
- `case_decomposition.csv`
- `summary_metrics.csv`
- `information_geometry.png`
- `summary.json`
- `source_manifest.json`
- `FINAL_RESULT.md`

