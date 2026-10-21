# Round 164 Temperature Twin Residual-Variance Ablation Contract

## Question

Does archived RegEM residual variance suppress otherwise observable temperature modes enough to explain the low Round 161/162 perfect-model state recovery?

## Frozen design

- Reuse the exact Round 161/162 500-member pool, 24 held-out truths, full proxy6/proxy9 operators, and native HMA weights.
- Use exact noise-free full-PSM pseudo-observations only.
- Scale the assimilation residual variance by the predeclared factors `1`, `0.25`, `0.0625`, `0.01`, and `0.0001`.
- The factors are a mechanism ablation, not candidates for production tuning or selection.
- Do not use actual proxy observations, instrumental values, locked 1921-1939 values, Jionglacuo, production reconstruction, or OGGM.

## Decision rule

- If near-zero residual variance raises equal-network median HMA state-RMSE skill to at least 50%, residual-error shrinkage is sufficient to explain the Round 161 gate failure under exact PSM truth.
- Otherwise the dominant remaining limit lies in PSM/operator directionality or the equal-network combination, not residual amplitude alone.

## Required outputs

- `case_state_skill.csv`
- `summary_state_skill.csv`
- `modal_transfer.csv`
- `residual_variance_ablation.png`
- `summary.json`
- `source_manifest.json`
- `FINAL_RESULT.md`

