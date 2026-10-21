# Round 200 V3 synthetic full-sheet xlsread closure

V2 read only `A1:PH14` and stopped after observing that `xlsread` cropped the
all-nonnumeric leading row/column, so its numeric-array shape could not emulate
the deposited full-sheet call. V3 constructs an independent synthetic workbook
containing the unchanged physical rows 1-14 plus only the deposited year labels
1-2019 in column A. All annual proxy cells remain blank.

MATLAB R2024b then runs the deposited full-sheet `xlsread` expressions against
this synthetic workbook. This reproduces numeric bounding-box semantics without
reading any original annual proxy value. V3 records both metadata closure and
the year vector produced by `num(15:end,1)`.

