# Round 199 East Asia proxy identity metadata audit

## Objective

Extract only `Proxies_Ensemble.xlsx` from the remote Zenodo `Data.zip`, parse
only workbook rows 1-14 defined by the deposited code as metadata, and compare
the proxy identities and coordinates with the project's reconciled 499-record
proxy ledger.

## Frozen boundary

- Fetch exactly one compressed ZIP member by HTTP byte range.
- Preserve and hash the extracted workbook, but decode only rows 1-14.
- Do not iterate, convert, summarize, or export annual proxy rows 15 onward.
- Do not inspect model, CRU, reconstruction, 1921-1939, or Jionglacuo values.
- Do not run reconstruction, correction, EnKF, MATLAB, or OGGM.
- Write only to the independent Round 199 directory.

## Interpretation

Exact normalized identity and exact coordinate/time metadata matches establish
source-family overlap. Coordinate-only matches are candidates for later source
review, not proof of identity. This round cannot validate the temperature field
or authorize its numerical use.

