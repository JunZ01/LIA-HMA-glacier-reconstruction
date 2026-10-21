# Round 134 NETP MJJ Exact-Prior Preflight V1

## Purpose

Create the May-July precipitation prior required for a NETP seasonal low-frequency reconstruction without changing the production loader or any existing source/output file.

## Frozen Construction

- MATLAB R2024b only.
- Original PHYDA `load_S_720360` and `modelyrs` implementations.
- CESM-LME010 (`cesm_lme010`).
- HMA bounds 24-51 N and 62-110 E.
- Member years 1001-1848 and RNG seed 4236 reset before every loader call.
- Annual precipitation: months 1-12.
- MJJ precipitation: months 5-7.
- NETP diagnostic mask: 35-42 N and 92-108 E, area weighted.

## Hard Gates

1. The exported annual 336 x 848 precipitation field must be exactly equal to the recovered production prior: maximum difference, RMSE, and changed-value count all zero.
2. Grid coordinates and member years must exactly match Round 129 V3.
3. MJJ must be finite, have shape 336 x 848, nonzero variance in every state cell, and not be byte/value-identical to JJA or annual precipitation.
4. The NETP mask must contain exactly 32 native state cells.
5. Existing source fingerprints must remain unchanged.

This preflight does not read proxy values, instrumental climate, QT-Prec values, Shenqi values, or 1921-1939 values. It does not open the frozen external validation reserve and does not run EnKF or OGGM. Passing only authorizes the later Round 134 reconstruction-design stage.

