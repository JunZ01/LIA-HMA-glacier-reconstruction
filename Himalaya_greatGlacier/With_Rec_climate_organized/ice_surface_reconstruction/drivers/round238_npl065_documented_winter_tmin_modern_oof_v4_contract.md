# Round 238 v4 contract: NPL065 documented winter Tmin modern OOF

This contract preserves the v1 scientific design, thresholds, and safety
boundaries, plus the v3 byte-level RWL header handling.

The sole implementation change from v3 is a prospectively declared per-series
detrending fallback. Each modern-only series is first fit with `ModNegEx`. If
and only if the optimizer raises a non-convergence `RuntimeError`, that series
is detrended with a linear curve using the same residual-ratio method. The
sample ID and exception type are recorded. No series is deleted for this
reason, and no target value is consulted when selecting the fallback.

This implements the conventional negative-exponential-or-straight-line rule.
It still does not claim exact equivalence to the paper's production chronology.
V3 stopped during chronology construction before climate magnitudes were read.

Original sources remain read-only. No 1921-1939 numeric value, paleo curve,
Jionglacuo value, EnKF, production reconstruction, correction, or OGGM run is
allowed.
