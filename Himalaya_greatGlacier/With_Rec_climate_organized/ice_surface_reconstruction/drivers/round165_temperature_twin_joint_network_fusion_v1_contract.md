# Round 165 Temperature Twin Joint-Network Fusion Contract

## Question

Does averaging separately assimilated proxy6 and proxy9 posterior means discard complementary information compared with a single stacked-network update?

## Frozen design

- Reuse the exact Round 161-164 prior pool, 24 held-out truths, full PSM operators, residual variances, and native HMA weights.
- Use exact noise-free pseudo-observations.
- Compare proxy6, proxy9, equal posterior averaging, and one stacked proxy6+proxy9 update at archived R and the predeclared near-zero diagnostic scale `0.0001`.
- Treat stacked assimilation as a mechanism bound only because cross-network residual independence has not been established.
- Use no real proxy observations, instrumental values, 1921-1939 values, Jionglacuo, production reconstruction, or OGGM.

## Decision rule

- Joint archived-R skill at or above 50% identifies posterior averaging as sufficient to explain the prior gate failure.
- Joint near-zero-R skill at or above 50%, with archived joint below it, identifies a combined residual-shrinkage and fusion boundary.
- Failure even at joint near-zero R closes network fusion as sufficient.

