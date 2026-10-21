# Round 169 Huang 2025 southeastern-TP MXD public-data recovery

## Objective

Determine whether Huang et al. (2025), DOI `10.1016/j.quascirev.2025.109266`, publicly provides machine-readable annual MXD chronology or August-September minimum-temperature reconstruction data extending from 1246 into the modern period.

## Scientific role

- The study is a potential new eastern-HMA temperature-evidence candidate because its documented MXD reconstruction starts in 1246 and explains 42.3% of instrumental variance.
- This recovery round does not establish source independence, modern transfer skill, or permission to inspect a paleo trajectory.
- The next stage is authorized only if an official annual numeric table/file is found.

## Protected acquisition

- Save the official Elsevier supplementary DOCX unchanged in an independent `raw_unopened` directory and record SHA256.
- Inspect the OOXML member inventory, table count, table dimensions, nonnumeric headers, and first-column year coverage only.
- If a table row begins with a year from 1921 through 1939, discard the complete row before integer or floating-point conversion and before any output.
- Do not render figures, digitize curves, OCR images, or export annual values in this round.

## Decision

- `advance_huang2025_annual_series_to_source_identity_audit` if the official supplement contains a machine-readable annual chronology or reconstruction table.
- `close_huang2025_anonymous_public_numeric_recovery` otherwise.
- No paleo comparison, production reconstruction, EnKF, Jionglacuo access, or OGGM run is authorized.
