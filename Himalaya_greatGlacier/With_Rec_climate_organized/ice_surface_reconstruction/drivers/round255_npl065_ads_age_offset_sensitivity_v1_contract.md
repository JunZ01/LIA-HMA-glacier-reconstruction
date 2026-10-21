# Round 255 NPL065 ADS age-offset sensitivity contract

Compare the Round 253 best modern-only method under two ADS stiffness clocks:

- `reset_at_modern_start`: start every observed modern segment at stiffness 30;
- `inherit_structural_age`: start at `30 + modern_first_year - original_first_year`,
  where original first years come from the zero-value-decoding Round 254 audit.

Both branches use the same Cook-Peters transformation estimated only from
allowed 1941-2013 values, ADS30 ratio indices, constrained positive end slope,
Tukey biweight chronology, no prewhitening, CRU TS4.01 four-cell DJF Tmin, six
blocked OOF folds, and rolling 30-year metrics.

The reset branch must close Round 253 to `1e-12`. No pre-1941 numeric values,
locked values, signal-free iteration, RCSigFree execution, EnKF,
reconstruction, or OGGM.
