# Round 181 v2 Contract: Haizi 903-Year Supplement Acquisition

Version 1 retained the NCBI OA metadata response but the current HTTPS OA
package mirror returned 404. Version 2 uses the official Springer static
supplement endpoint for DOI `10.1038/s41598-024-79096-6` and changes no
scientific criterion.

## Operations

- Preserve the NCBI OA response unchanged.
- Preserve the official Springer XLSX unchanged and record SHA256.
- Verify only that the file is a ZIP container by magic bytes.

## Hard boundaries

- Do not open, parse, render, or preview spreadsheet contents.
- Do not access annual, locked-period, or Jionglacuo values.
- Do not run identity matching, OOF, EnKF, correction, production
  reconstruction, or OGGM.
