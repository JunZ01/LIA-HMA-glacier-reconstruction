# Round 139 East-HMA External Workbook Boundary Preflight

## Purpose

Inspect the two already downloaded east/south HMA hydroclimate workbooks,
identify their sheets and series, and export only values allowed by the project
boundary.

## Source policy

- Both source XLSX files are read-only.
- Parse the XLSX ZIP/XML structure directly; do not load the workbook through
  an API that materializes every cell.
- Determine a data-row year from the first populated cell only.
- For rows dated 1921-1939, inspect cell type attributes but do not decode any
  value other than the year used to reject the row.
- If any locked row stores a non-year value as shared, inline, or formula text,
  stop the affected workbook before resolving strings.
- Export only 1000-1920 and 1940-2000 data rows. Preserve pre-data header rows
  only for column identity.

## Outputs

- Workbook/sheet inventory and boundary counts.
- One independent allowed-values CSV for each data sheet.
- Header and first/last allowed-row previews.
- Source fingerprints before and after.

This preflight does not establish proxy independence, climate-variable
validity, or model skill. Those require a separately frozen identity closure.
It does not run EnKF or OGGM and does not open the frozen external validation
reserve.
