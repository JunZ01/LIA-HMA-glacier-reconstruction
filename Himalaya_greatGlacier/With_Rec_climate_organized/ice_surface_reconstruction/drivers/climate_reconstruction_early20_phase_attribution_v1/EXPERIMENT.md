# Early twentieth-century phase attribution v1

## Question

Is the 1901-1950 phase mismatch between coordinate-corrected `rec_result` and
`rec_ref` at system 3584 a robust local reversal, a bounded timing offset, or
a member/group-specific behavior? Does it occur across the HMA mask?

## Frozen scope

- Corrected six-member product and the public `rec_ref` field only.
- Primary period: 1901-1950. Context period: 1901-2000.
- Target: 3584 at 33.75 N, 75.75 E; regional HMA mean uses the verified mask
  and cosine-latitude weights.
- Centered 15-year and 31-year low-pass diagnostics; lag scan from -30 to
  +30 years, where positive lag means `rec_result` leads `rec_ref`.
- `rec_ref` is a comparison product, not truth.
- No OGGM run, no instrumental selection, and no reconstruction tuning.

## Required evidence

- raw, detrended, first-difference, and low-pass correlations;
- trend sign and centered-sign agreement;
- zero-lag versus best bounded-lag correlation;
- six-member, proxy6/proxy9, and method-family decomposition;
- 3584 versus HMA spatial correlation maps;
- source-purity hashes before and after.
