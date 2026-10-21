# Round 116 Temperature Low-Rank Residual OOF Contract

## Question

Round 115 showed that a proxy-only annual low-rank field loses the transferable
modern trend carried by the equal-six reconstruction. Can the same frozen proxy
network instead predict only the instrumental residual around equal-six and
produce a genuine out-of-fold improvement?

## Frozen Design

- Reuse the Round 115 v2 years, five outer blocks, 99-cell common observed HMA
  support, five instrumental products, 226 documented-temperature proxy9
  evidence groups, training-only proxy normalization, missing-value policy,
  area-weighted EOF algorithm, inner folds, rank grid, ridge grid, scoring
  functions, equal-six control, zero control, and all Round 113 gates.
- The only scientific change is the target. For each fit, target the training
  residual `five-product mean - equal-six`. The held-out candidate is
  `equal-six + predicted residual`.
- Equal-six values in a held-out year are a frozen reconstruction input, not an
  instrumental target. No held-out instrumental value enters its residual
  prediction.
- No blend factor, trend term, time predictor, localization, proxy deletion,
  product weight, gate change, or result-directed hyperparameter is added.
- The stitched candidate is recentered to 1961-1990 using candidate predictions
  only, consistent with Round 115.

## Boundary And Decision

No 1921-1939 value and no Round 114 external series may be opened. Original
sources remain read-only; outputs are independent. A pass of every Round 113
modern gate advances this residual method to a separately contracted full
candidate. Failure closes annual linear low-rank correction of equal-six with
the current proxy9 temperature network and moves the mainline away from further
annual proxy-regression variants. OGGM remains prohibited.
