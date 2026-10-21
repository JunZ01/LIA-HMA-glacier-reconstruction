# Round 239 contract: NPL065 Tmin versus Tmean failure attribution

## Question

Why did NPL065 pass the exact CRU winter Tmin screen but fail the Berkeley
winter Tmean robustness threshold in Round 238: temperature-variable
specificity or gridded-product disagreement?

## Frozen design

- Reuse only the allowed 1960-2000 chronology and target curves exported by
  Round 238 v4. Do not reopen NPL065 ring-width magnitudes.
- Preserve the Round 238 folds, regression, OOF metrics, and 0.30 correlation
  diagnostic threshold exactly.
- Add one target only: previous-December/current-January-February mean
  temperature from the nearest CRU TS4.04 grid cell.
- Read the CRU mean-temperature source only for 1959-2000, required to form
  winter targets for ring years 1960-2000.
- Recompute Round 238 CRU Tmin and Berkeley Tmean OOF results from the exported
  curves and require numerical closure within 1e-12.

## Interpretation

- CRU Tmin passes and both CRU/BEA Tmean fail: evidence for Tmin-specific proxy
  sensitivity.
- CRU Tmin and CRU Tmean pass but Berkeley Tmean fails: evidence for a gridded
  product representation difference.
- All pass: the Round 238 Berkeley failure does not reproduce.
- Any other combination remains mixed.

This diagnostic does not relax or overwrite the failed Round 238 dual-product
admission gate. CRU TS4.04 Tmean differs in both variable and version from CRU
TS4.01 Tmin; a product-difference result therefore requires a same-version
follow-up.

## Boundaries

- Original files are read-only; outputs go only to an independent Round 239
  result directory.
- No 1921-1939 value, paleo proxy value, or historical reconstruction is read.
- Jionglacuo is not opened or hashed.
- No EnKF, production reconstruction, climate correction, or OGGM run occurs.
