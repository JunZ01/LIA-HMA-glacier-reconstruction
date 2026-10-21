# Early nineteenth-century phase attribution v1

## Question

Why does the coordinate-corrected `rec_result` remain partly out of phase with
`rec_ref` during the early nineteenth century, and which reconstruction
members create that behavior?

## Frozen scope

- Corrected six-member 0.5-degree NetCDF from the independent coordinate
  regeneration experiment.
- `rec_ref` is a comparison reconstruction, not truth.
- Target: system 3584 at 33.75 N, 75.75 E.
- Regional diagnostic: the verified 1724-cell HMA mask with cosine-latitude
  weights.
- Primary periods: 1801-1850 and 1801-1900.
- Filters: centered 15-year and 31-year means.
- Lag scan: -30 to +30 years. Positive lag means `rec_result` leads `rec_ref`.
- No instrumental selection, no OGGM run, and no climate or glacier tuning.

## Diagnostic separation

The experiment distinguishes:

1. trend-sign conflict;
2. phase reversal after detrending;
3. timing displacement recoverable by a bounded lag;
4. local behavior at 3584 versus regional HMA behavior;
5. proxy6 versus proxy9 and method-family contributions.

Source hashes must remain unchanged.
