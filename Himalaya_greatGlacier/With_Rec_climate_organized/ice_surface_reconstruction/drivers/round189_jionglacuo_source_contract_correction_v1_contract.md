# Round 189 Jionglacuo Source-Contract Correction

## Purpose

Resolve a newly discovered source-identity conflict without reopening any
candidate workbook or numeric series.

Mendeley dataset `10.17632/s7t4zc34sr.2` links to Nature article
`10.1038/s41612-025-01254-y`. The official article must be checked for the
sampling-site identity. If it identifies Jionglacuo, all Round 184 scientific
results derived from that workbook are invalid under the project's explicit
Jionglacuo prohibition.

## Allowed inputs

- The unchanged Mendeley metadata snapshot from Round 178.
- Round 184 `summary.json` and `source_manifest.json` only.
- The official Nature article HTML.

## Forbidden access

- Do not open, hash, parse, copy, or inspect the Jionglacuo workbook.
- Do not inspect any annual or window value derived from it.
- Do not access the locked period, run EnKF, perform a correction, or run OGGM.

## Decision rule

If the Mendeley metadata links the exact Nature DOI and the official article
identifies its tree-ring sampling site as Jionglacuo:

1. mark Round 184 invalid and retract its external warm-bias support;
2. prohibit all future use of `s7t4zc34sr.2` and its local aliases;
3. retain Round 185-188 because they use LMR, instrumental products, and
   Chongce rather than this source;
4. update the canonical facts before any new east-HMA experiment.

