# Round 241 contract: NPL065 same-version CRU versus Berkeley attribution

## Question

Does the Round 239 CRU-versus-Berkeley winter-temperature skill difference
persist when CRU Tmin and CRU Tmean both use TS4.01?

## Frozen design

- Preserve the Round 238 chronology, 1960-2000 years, five blocked folds,
  regression, metrics, and 0.30 diagnostic threshold.
- Preserve the exported CRU TS4.01 Tmin and Berkeley Tmean targets.
- Replace only the Round 239 CRU TS4.04 Tmean branch with official CRU TS4.01
  Tmean from Round 240.
- Use previous December plus current January-February at the same nearest CRU
  grid cell.
- Verify every Round 240 NetCDF hash against its acquisition manifest.
- Require Round 238 metric closure within 1e-12.

If both TS4.01 CRU variables pass while Berkeley Tmean fails, the local product
representation difference is confirmed for this candidate. This does not
relax or overwrite the failed Round 238 admission gate and does not establish
which gridded product is correct.

Original sources are read-only. No 1921-1939 value, paleo proxy value,
Jionglacuo, EnKF, production reconstruction, correction, or OGGM run is
allowed.
