# Round 163 Temperature PSM Mode Observability Contract

## Question

Which weighted HMA prior temperature modes are visible to the archived proxy6/proxy9 operators, and which truth-relevant modes are suppressed by the residual-standardized observation geometry?

## Frozen scope

- Reuse the Round 161/162 experiment-1 500-member pool, 24 held-out truth members, full archived PSM branches, RegEM residual variances, and native HMA grid.
- Use no actual proxy observation, instrumental target value, locked 1921-1939 value, or Jionglacuo value.
- Run no production reconstruction, EnKF reconstruction, or OGGM simulation.
- Make no localization, proxy selection, gain tuning, or bias correction.

## Method

1. Compute the area-weighted HMA SVD of the frozen prior anomalies.
2. For every prior mode, calculate prior variance, held-out truth-error energy, residual-standardized proxy information, exact linear-Gaussian posterior variance reduction, and state/proxy cross-covariance coupling energy.
3. Report proxy6, proxy9, and their stacked diagnostic union separately.
4. Map the first four prior modes only to interpret the dominant geometry; do not treat these maps as reconstructed climate.

## Required outputs

- `mode_observability.csv`
- `network_spectrum_summary.csv`
- `mode_observability.png`
- `summary.json`
- `source_manifest.json`
- `FINAL_RESULT.md`

