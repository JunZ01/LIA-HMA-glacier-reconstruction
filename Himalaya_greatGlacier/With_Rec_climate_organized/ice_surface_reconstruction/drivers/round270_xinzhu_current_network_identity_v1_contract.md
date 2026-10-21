# Round 270 Xinzhu current-network identity contract

- Decode only Xinzhu reconstruction values for 1940-2000.
- Skip 1921-1939 and all outside-scope rows before value conversion.
- Compare the allowed Xinzhu sequence against every record in the frozen
  proxy6 and proxy9 1940-2000 matrices.
- Exact alias: overlap at least 50 years, absolute correlation at least
  0.999999, and affine RMSE fraction at most 1e-6.
- Near alias: overlap at least 50 years and absolute correlation at least
  0.995.
- Any exact or near alias quarantines Xinzhu as non-independent. Zero aliases
  admits it only to modern precipitation OOF.
- No paleo trajectory, locked-period validation, EnKF, production
  reconstruction, climate correction, or OGGM is authorized.

