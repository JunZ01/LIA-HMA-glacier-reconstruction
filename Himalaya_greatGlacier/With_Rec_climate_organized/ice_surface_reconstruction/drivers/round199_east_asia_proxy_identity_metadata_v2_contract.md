# Round 199 V2 first-sheet correction

V1 extracted and CRC-verified the workbook, then stopped before reading cells
because the file has three worksheets. The deposited reconstruction code calls
`xlsread(proxy_file, 1)`, explicitly selecting worksheet index 1. V2 therefore
changes only the workbook adapter so that the unchanged V1 parser sees and
decodes rows 1-14 from the first worksheet. Worksheets 2 and 3 are not parsed.
All V1 scientific and locked-data boundaries remain frozen.

