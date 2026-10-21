# Round 200 MATLAB xlsread metadata closure

## Objective

Verify how MATLAB R2024b `xlsread` maps the deposited workbook's physical rows
to the `num` and `text` arrays used by the reconstruction code.

## Boundary

- Read worksheet 1 range `A1:PH14` only.
- Do not read or decode worksheet row 15 onward.
- Compare the deposited expressions `num(5:10,2:end)` and
  `text(1,2:end)`, `text(12,2:end)` with the Round 199 V3 physical-row
  metadata export.
- Do not read annual proxy, climate, model, reconstruction, locked-period, or
  Jionglacuo values.
- Use MATLAB R2024b only and write to the independent Round 200 directory.
- Do not run reconstruction, correction, EnKF, or OGGM.

