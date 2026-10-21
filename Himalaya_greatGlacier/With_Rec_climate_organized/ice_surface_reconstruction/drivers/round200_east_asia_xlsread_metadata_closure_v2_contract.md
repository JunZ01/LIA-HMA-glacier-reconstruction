# Round 200 V2 syntax correction

V1 stopped at MATLAB parse time before opening the workbook because the
multi-line `table` call lacked a continuation token after the opening line.
V2 changes only that syntax and writes to an independent output directory.
The V1 metadata-only range and all scientific boundaries remain frozen.

