# Round 150 NOAA Temperature Trajectory Extension Contract

## Question

Do the newly acquired Eastern Tibet 2015 and Bhutan 2015 JJA temperature
reconstructions support the low-frequency trajectory of the frozen eastern-HMA
temperature reconstruction, especially during 1901--1920?

## Frozen design

- Reuse only the allowed-year regional curve exported by Round 122.
- Use only the primary published reconstruction column from each Round 149
  export; exclude uncertainty bounds, sample counts, low-pass duplicates, and
  observed calibration targets from the candidate set.
- Compare 1500--1920, 1901--1920, and 1940--2000 exactly as in Round 122.
- Use raw annual correlation, centered 31-year correlation where defined, and
  linear trend direction. Standardize only for plotting.
- Never access 1921--1939 values and never open Jionglacuo.
- Do not run EnKF, OGGM, PSM fitting, or tuning.
- Treat both products as source-overlapping external trajectory benchmarks, not
  independent validation and not raw proxies.

## Decision

The experiment may strengthen or weaken confidence in a regional low-frequency
phase. It cannot promote either product into the assimilation network and cannot
by itself identify which reconstruction is correct.
