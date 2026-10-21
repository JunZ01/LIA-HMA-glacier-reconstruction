# Round 200 V4 synthetic workbook preparation correction

V3 stopped before `xlsread` because MATLAB `readcell` returned `missing` objects
that `writecell` would not serialize. V4 prepares the same metadata-plus-year-
labels synthetic workbook with openpyxl, iterating only physical source rows
1-14. MATLAB R2024b then applies the deposited full-sheet `xlsread` expressions
to that synthetic file. No original annual proxy value is decoded.

