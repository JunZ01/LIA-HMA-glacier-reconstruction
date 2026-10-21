# Round 227: Yang 2010 / PAGES2k source-lineage closure

## Purpose

Use the official PAGES2k v2 Supplementary Table 1 to determine the publication,
site, and archive lineage of project records `Asia-MAINJT::Asia_222` and
`Asia-NYINJT::Asia_223`, then test whether either record is the same source or
sample collection as Yang et al. (2010), DOI `10.1177/0959683609350388`.

## Allowed reads

- `LIA_PROJECT_FACTS.md` for its hash only.
- Round 226 v2 derived metadata summary.
- Europe PMC's official supplementary-files endpoint for `PMC5505119`.
- The metadata-only workbook `sdata201788-s3.xlsx` extracted byte-for-byte from
  that official ZIP.

## Forbidden reads and actions

- Do not open PAGES2k LiPD, CSV, MAT, R, pickle, proxy values, annual series, or
  reconstruction values.
- Do not access any value from 1921-1939.
- Do not access Jionglacuo data or metadata.
- Do not modify source files, production climate, EnKF inputs, MATLAB files, or
  OGGM files.
- Do not overwrite an existing result directory.

## Outputs

Create one independent result directory containing the unmodified official ZIP,
the unmodified extracted metadata workbook, matched metadata rows, hashes, a
machine-readable decision, and a concise final result.
