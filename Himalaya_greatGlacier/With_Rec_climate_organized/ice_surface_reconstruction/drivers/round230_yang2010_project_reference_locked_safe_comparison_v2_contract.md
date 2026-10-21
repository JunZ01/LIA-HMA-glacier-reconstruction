# Round 230 v2 Contract

The scientific design and frozen gates are identical to Round 230 v1. Version
2 exists only because the inherited helper import chain requires `openpyxl`
even though no workbook function is used by this experiment. The wrapper
provides a fail-closed placeholder for that unused import and redirects all
outputs to an independent v2 result directory. If any workbook reader is
called, execution must stop.

