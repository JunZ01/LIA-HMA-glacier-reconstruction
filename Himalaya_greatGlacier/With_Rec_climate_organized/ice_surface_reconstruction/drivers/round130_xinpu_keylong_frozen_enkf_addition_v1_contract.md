# Round 130 Xinpu-Keylong Frozen EnKF Addition Contract

Run a mechanism sensitivity for reconstruction years 1000-1920 only. The locked 1921-1939 interval and frozen external validation reserve remain unopened.

Freeze the recovered annual precipitation prior, 500 replay member indices, 202 archived proxy observations/operators, RegEM residual matrix, and `M_update`. Add only Xinpu and Keylong because they passed every Round 129 exact seasonal-prior coupling gate. Use Round 129 full-calibration coefficients and exact seasonal fields. New-proxy R is diagonal and fixed to the corresponding Round 128 self-product OOF RMSE squared.

Branches are baseline, Xinpu-only, Keylong-only, and both records, with BEA+GPCC and CRU coefficient versions for every added-proxy branch. No localization change, proxy deletion, production overwrite, or OGGM run is permitted. This experiment measures influence and cross-product robustness; it does not by itself establish reconstruction accuracy.
