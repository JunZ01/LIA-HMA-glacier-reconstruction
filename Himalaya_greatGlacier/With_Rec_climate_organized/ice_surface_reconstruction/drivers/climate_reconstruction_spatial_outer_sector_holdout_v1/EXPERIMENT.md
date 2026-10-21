# Frozen HMA Joint Temporal-Spatial Outer Test

This independent no-OGGM experiment tests the frozen temperature spatial
architecture on both a new period and a completely unassimilated region.

The experiment crosses four held-out instrumental products with four held-out
longitude sectors. Instrument-error statistics use only 1901-1980 and only
cells outside the held-out sector. During 1981-2000, the held-out product and
held-out sector are absent from the EnKF observations and are used only for
scoring.

Frozen settings:

- fold-5 proxy operator;
- regional error multiplier `0.0625`;
- spatial covariance shrinkage `0.25`;
- spatial covariance scale `0.00390625`;
- the four-sector partition from sector-holdout v2.

The spatial mechanism receives joint outer support only if it beats proxy-only
and visible-sector regional anchoring in pooled centered RMSE, improves annual
spatial-anomaly correlation over proxy-only, and improves RMSE over proxy-only
in every held-out sector. The direct three-product mean in the hidden sector is
reported only as a benchmark. `rec_ref`, precipitation reconstruction, OGGM,
and outer-period parameter selection are excluded.
