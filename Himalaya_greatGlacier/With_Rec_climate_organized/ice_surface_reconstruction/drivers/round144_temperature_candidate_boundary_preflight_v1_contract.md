# Round 144 Temperature-Candidate Boundary Preflight V1

## Objective

Create allowed-period working copies for the unsealed Hengduan and Haizi temperature candidates before identity/chronology testing, while preserving all downloaded source files and the sealed Jionglacuo external validation record.

## Frozen scope

- Parse the five Hengduan Tucson-style raw tree-ring measurement files and the Haizi supplementary workbook.
- Inspect the Hengduan upload-information workbook only for workbook/sheet structure unless a sheet has an explicit numeric year axis.
- Reject every 1921-1939 row or Tucson value slot before its non-year value is decoded.
- Export only years 1000-1920 and 1940-2000 to the independent Round 144 result directory.
- Record source hashes before and after execution.
- Do not open or parse `jionglacuo_temperature_reconstruction.xlsx`; it remains sealed for later external validation.

This round performs no identity decision, PSM fit, reconstruction, EnKF, or OGGM run. Allowed exports advance only to exact identity, chronology, site, measurement, and source-role closure.
