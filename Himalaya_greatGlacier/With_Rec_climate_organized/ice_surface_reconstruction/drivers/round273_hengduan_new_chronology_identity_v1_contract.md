# Round 273 Hengduan new-chronology identity contract

- Read only chronology values for 1940-2000 from the official Zenodo workbook.
  Decode each XML row's year first and discard 1921-1939 before accessing the
  chronology cell.
- Compare all 30 clustered chronologies against every record in the frozen
  proxy6 and proxy9 1940-2000 matrices.
- Classify the nine privately contributed 2018 collections prospectively as:
  northern HLS, JYC, YZC; southern BLX, CYS, FNPTS, GDCTS, GDCYS, LGQ.
- Exact affine alias: overlap at least 50 years, absolute correlation at least
  0.999999, and affine residual RMSE fraction at most 1e-6.
- Near alias: overlap at least 50 years and absolute correlation at least 0.995.
- Advance if at least six of the nine new chronologies have at least 55 modern
  values and no exact or near alias, both clusters retain at least two admitted
  new chronologies, and at least one ITRDB chronology is detected as an alias
  (positive-control sensitivity).
- Do not read either published paleo reconstruction, run EnKF, modify a
  production reconstruction, or run OGGM.

