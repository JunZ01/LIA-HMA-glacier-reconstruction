# Round 182 Contract: Haizi 903-Year Workbook Structure

## Purpose

Identify sheet structure and safe modern-only parser boundaries without
reading any numeric spreadsheet value.

## Operations

- Reuse the verified Round 179 OOXML structural inspector.
- Read ZIP members, sheet names, dimensions, cell types, and string cells in
  rows 1--10 only.
- Verify source SHA256 before and after.

## Hard boundaries

- Do not read numeric cell values.
- Do not access annual, locked-period, or Jionglacuo values.
- Do not run identity matching, OOF, EnKF, correction, production
  reconstruction, or OGGM.
