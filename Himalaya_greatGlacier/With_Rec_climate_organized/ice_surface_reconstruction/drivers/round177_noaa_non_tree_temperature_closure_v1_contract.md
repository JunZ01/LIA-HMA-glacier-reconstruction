# Round 177 Contract: NOAA Non-Tree Temperature Closure

## Purpose

Adjudicate the physical meaning of the only Round 176 v3 tables that pass the
row-level modern-frequency screen.

## Frozen evidence

- Round 176 v3 time-axis audit and summary.
- Round 175 unchanged NOAA study metadata JSON.
- No proxy, isotope, temperature, precipitation, or climate value.

## Decision rule

A record explicitly documented as precipitation or hydroclimate is not a
temperature candidate even when its sampling frequency passes. No orientation
may be inferred from isotope sign alone.

## Hard boundaries

- Do not decode any non-time annual value.
- Do not access locked-period non-time values or Jionglacuo.
- Do not run OOF, EnKF, production reconstruction, correction, or OGGM.
