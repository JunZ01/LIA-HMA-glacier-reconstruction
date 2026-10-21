# Round 120 External Temperature Chronology Incremental OOF Contract

## Question

Round 119 permits reopening temperature reconstruction only with new,
low-frequency-preserving HMA temperature chronologies that are distinct from
the frozen validation reserve. Do the independently identified Hengduan
regional tree-ring chronology and Haizi Mountain STD chronology repair the
specific modern OOF weaknesses of the Round 116 equal-six residual candidate?

## Frozen Design

- Reuse Round 116 years (1940-2000), five outer blocks, nested inner blocks,
  99-cell common HMA support, five instrumental temperature products, target,
  equal-six and zero controls, training-only standardization, missing-value
  policy, area-weighted EOF algorithm, rank grid, ridge grid, scoring functions,
  and every Round 113 gate.
- Reuse the exact 226 documented-temperature proxy9 evidence groups selected in
  Round 116. No existing proxy is deleted, reweighted, or relabeled.
- Test three predeclared additions: Hengduan only (227 predictors), Haizi only
  (227 predictors), and both chronologies (228 predictors).
- Extract only 1940-2000 chronology values from the source workbooks. The
  1921-1939 chronology values are not decoded, retained, compared, or written.
- The Hengduan predictor is the published five-site regional tree-ring-width
  chronology. The Haizi predictor is the published `std` chronology, not the
  authors' reconstructed temperature series.
- Every branch independently repeats nested rank/ridge selection. No
  localization, blend factor, season search, trend term, time predictor,
  product weighting, gate change, or result-directed tuning is allowed.

## Decision

The combined two-record branch is the predeclared promotion branch. It advances
only if it passes every frozen Round 113 modern gate. The two single-record
branches provide attribution and cannot be selected post hoc for production.
Failure closes these two chronologies as a sufficient repair of the annual
linear low-rank residual method; it does not invalidate their archive identity.
Original sources remain read-only, outputs are independent, the frozen external
validation reserve remains sealed, and OGGM remains prohibited.
