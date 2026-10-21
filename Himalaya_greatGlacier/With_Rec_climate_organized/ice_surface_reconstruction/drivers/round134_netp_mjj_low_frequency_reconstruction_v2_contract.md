# Round 134 NETP MJJ Low-Frequency Reconstruction V2

This contract preserves every scientific and validation rule in V1. The V1 driver was never executed. Static review found that its two disjoint blocked-smoothing arrays were combined by ordinary addition, so finite values would be added to NaN outside each block. V2 changes only that implementation expression to elementwise finite-value combination (`numpy.fmax`) in the two OOF calculations and the matching plot. No data, periods, proxy selection, grouping, model, or gate changes.

The frozen design remains: 19 training proxies after excluding proxy 15, 15 equal-weight 0.25-degree groups, within-fold 11-year smoothing, fit-only sign/scale estimation, one scalar calibration, GPCC and CRU products, 1000-1920 reconstruction, QT-Prec validation-only, and all V1 advance gates. Values from 1921-1939, the final reserve, EnKF, and OGGM remain prohibited.

