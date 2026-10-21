# HMA Spatial-Sector Holdout Pilot v2

This independent no-OGGM experiment supersedes the v1 sector-holdout score.
Version 1 compared the absolute EnKF state with instrumental anomalies, which
created 7-12 degC offsets, and its NetCDF read included years after 1980. Those
v1 values are implementation-audit evidence only and cannot select a spatial
model.

Version 2 reads only 1901-1980 from the instrumental NetCDF. For every held-out
20-year block, each product and grid cell is referenced to its own remaining
60 training years. Validation uses temporal-mean-centered grid-cell RMSE and
annual spatial-anomaly correlation, so static climatological offsets cannot
drive the decision.

The 86 common HMA cells are divided into four mutually exclusive longitude
sectors. The experiment crosses four held-out products, four held-out time
blocks, and four completely unassimilated sectors, for 64 validation cells per
branch. It compares proxy-only, visible-sector regional anchoring, the frozen
boundary spatial-covariance mechanism, and a direct three-product benchmark.

Frozen controls:

- regional error multiplier: `0.0625`;
- spatial shrinkage: `0.25`;
- spatial boundary scale: `0.00390625`.

The spatial mechanism advances only if its pooled centered RMSE beats both
proxy-only and regional anchoring and its spatial-anomaly correlation beats
proxy-only. `rec_ref`, OGGM, precipitation reconstruction, and 1981-2000 are
not used.
