# Round 139 East-HMA XLSX Boundary Preflight V2

This contract is scientifically identical to V1. V1 stopped on a Windows
output-path-length error before statistics. V2 changes only the result directory
and workbook labels used in CSV filenames.

- Source workbooks remain read-only.
- Determine the year from the first populated numeric cell.
- Reject 1921-1939 rows before decoding any other value.
- Stop if a locked row contains shared, inline, or formula string values.
- Export only 1000-1920 and 1940-2000.
- Do not open the frozen reserve or run EnKF/OGGM.
