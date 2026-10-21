# Round 181 Contract: Haizi 903-Year Temperature Supplement Acquisition

## Purpose

Recover the official supplementary spreadsheet for the 903-year Haizi Mountain
mean-annual-temperature reconstruction, DOI `10.1038/s41598-024-79096-6`.

## Frozen source

- NCBI PMC record `PMC11554784`.
- Resolve the OA package through the official `oa.fcgi` response.
- Replace only the transport scheme of the returned NCBI FTP URL with HTTPS.

## Operations

- Preserve the OA response and complete package unchanged.
- Inventory TAR member metadata.
- Extract spreadsheet members byte-for-byte into `raw_unopened`.

## Hard boundaries

- Do not open, parse, render, or preview a spreadsheet.
- Do not access annual, locked-period, or Jionglacuo values.
- Do not run identity matching, OOF, EnKF, correction, production
  reconstruction, or OGGM.
