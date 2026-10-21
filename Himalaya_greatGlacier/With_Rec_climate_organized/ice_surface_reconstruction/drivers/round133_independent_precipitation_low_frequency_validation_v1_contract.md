# Round 133 Independent Precipitation Low-Frequency Validation

## Purpose

Test whether the Round 130 and Round 132 Xinpu-Keylong precipitation branches improve independent low-frequency agreement relative to the frozen baseline. This is validation only; it does not tune, assimilate, or select proxy observations.

## Frozen Inputs

- Round 130 posterior branches: `baseline`, `both_bea`, and `both_cru`.
- Round 132 product-marginalized posterior branch.
- Round 129 exact state-grid coordinates.
- Round 122 public external benchmark values, which already exclude 1921-1939.
- No final external validation reserve and no OGGM run.

## Spatial And Temporal Rules

- QT-Prec is compared with the area-weighted state mean over the paper-defined NETP box: 35-42 degrees N, 92-108 degrees E.
- Shenqi Cave is compared with the nearest state cell to 28.93 degrees N, 103.10 degrees E (29 degrees N, 103 degrees E).
- Formal periods are 1500-1920 and 1800-1920 only.
- Annual model and QT-Prec curves use a centered 31-year mean.
- Shenqi PI remains on its observed five-year dates, receives a centered seven-sample mean (approximately 35 years), and is paired only with model values at the same integer years. No external series is interpolated.
- Because the state is annual precipitation while QT-Prec is May-July total precipitation and Shenqi is a monsoon precipitation index, comparisons use period-standardized phase correlations and standardized trend directions. Raw amplitudes are not compared.

## Predeclared Decision Gate

QT-Prec is the primary independent archive; Shenqi is secondary because it is an isotope-derived hydroclimate record.

A candidate has independent low-frequency support only if all conditions hold:

1. Its mean QT-Prec correlation improvement over baseline across 1500-1920 and 1800-1920 is at least +0.05.
2. Neither QT-Prec period correlation is degraded by more than 0.02.
3. Its mean Shenqi correlation is not degraded by more than 0.05.
4. It does not turn a baseline-correct trend sign into a wrong trend sign in any benchmark-period pair.

If no candidate passes, close the current annual-Gaussian precipitation-assimilation route for Xinpu and Keylong. A future explicitly temporal/low-frequency operator would require new development and validation rather than promotion of these branches.

