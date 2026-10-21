# Round 179 Contract: Mendeley Temperature Structure and Identity

## Purpose

Determine the safe parser boundary and test exact raw-sample aliases before
opening any annual proxy or reconstruction value from Round 178.

## Frozen operations

- For the five raw tree-ring files, decode only bytes 1--12 of each row:
  sample identifier and segment start year. Never decode ring-width tokens.
- Compare normalized sample identifiers among the five candidates and against
  existing project `.rwl` files.
- For XLSX files, inspect ZIP members, workbook relationships, sheet names,
  dimensions, row/cell counts, cell types, and string cells in rows 1--10.
- Never read a numeric spreadsheet cell value.

## Hard boundaries

- Do not decode, export, plot, or correlate annual proxy or climate values.
- Do not access 1921--1939 values or Jionglacuo.
- Do not run OOF, EnKF, correction, production reconstruction, or OGGM.
- Do not modify any Round 178 file.

## Decision

Use the observed workbook structure to write a separately contracted parser
that rejects locked rows before numeric value conversion. Sample-ID matches
must be resolved before either candidate is treated as independent evidence.
