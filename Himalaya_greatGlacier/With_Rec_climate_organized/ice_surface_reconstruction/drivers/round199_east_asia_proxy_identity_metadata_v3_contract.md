# Round 199 V3 workbook-row semantic correction

Inspection of the V2 metadata export showed that the physical OOXML rows are:
row 4 source ID, rows 6-8 latitude/longitude/elevation, rows 10-11 start/end
year, rows 12-14 archive/measurement/publication title. The deposited MATLAB
comments label latitude/longitude as rows 5-6 and start/end as rows 9-10.

V3 uses only the already decoded V2 rows 1-14, corrects their physical-row
semantics, and reruns metadata overlap. It does not reopen the workbook or any
annual row. The mismatch is recorded as a potential deposited-code indexing
risk requiring a separate exact MATLAB `xlsread` closure before the external
field can be trusted.

