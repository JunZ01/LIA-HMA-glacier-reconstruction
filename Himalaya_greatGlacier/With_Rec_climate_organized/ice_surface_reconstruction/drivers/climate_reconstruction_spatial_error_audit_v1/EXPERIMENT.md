# HMA Instrumental Spatial-Error Audit v1

This independent no-OGGM audit determines how the four instrumental
temperature products may enter a 336-state spatial EnKF without treating
strongly correlated grid cells as independent observations.

All products are mapped to the existing 2-degree EnKF state grid and expressed
as 1961-1990 anomalies. For each product, the other three-product mean minus
the held-out product is computed over 1901-1980. The resulting product-by-time
residual fields diagnose coverage, per-grid variance, spatial correlation,
covariance eigenvalue spectrum, participation-ratio effective rank, and the
number of modes needed for 80/90/95/99 percent variance.

No EnKF branch is run and no spatial error model is selected from 1981-2000.
Original NetCDF and mask inputs remain read-only. `rec_ref` is not used.
