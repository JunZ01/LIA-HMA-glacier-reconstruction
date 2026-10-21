# Frozen HMA Spatial Outer Test, 1981-2000

This independent no-OGGM experiment evaluates the spatial temperature
architecture frozen before opening the 1981-2000 outer period. It performs no
covariance, scale, proxy, or method selection.

For each of four instrumental products, the other three products are averaged
and assimilated. Every product and grid cell is referenced only to its own
1901-1980 mean. Instrument-error covariance is estimated only from 1901-1980
residuals. The held-out product's 1981-2000 values are used only for final
scoring.

Frozen settings:

- fold-5 proxy operator;
- regional error multiplier `0.0625`;
- spatial covariance shrinkage `0.25`;
- spatial covariance scale `0.00390625`;
- 86 common HMA temperature cells.

Branches:

- proxy-only;
- regional instrumental anchor;
- frozen spatial instrumental anchor;
- direct three-product mean benchmark.

The outer period supports the spatial mechanism only if the spatial branch
beats both EnKF controls in pooled centered RMSE and beats proxy-only in annual
spatial-anomaly correlation. This remains product cross-validation rather than
independence among products. `rec_ref`, precipitation reconstruction, and OGGM
are not used.
