# Round 225 Zhu 2011 browser-download ingest

Admit only an exact browser-downloaded `Zhuetal2011P3.xls` or
`Zhuetal2011P3.xlsx` file from the user's Downloads directory (or an explicitly
provided path). Copy its bytes unchanged into a new independent result
directory, verify the OLE/ZIP spreadsheet signature, and record SHA256 and file
metadata.

Do not open a workbook, enumerate sheets, decode cells, parse years or values,
access 1921-1939 or Jionglacuo, or run PSM, reconstruction, correction, EnKF,
MATLAB, or OGGM. Reject partial downloads, HTML/challenge responses, unrelated
filenames, and files without a spreadsheet signature. Create no result
directory unless a valid completed spreadsheet is found.
