# Round 193 Contract: Dual Thermal Index Historical Shape

## Purpose

Test whether the Round 192 modern-passing eastern-HMA thermal index transports
to 1500--1899 low-frequency shape when compared with Wang et al. (2015).

## Frozen series

- Predictor: the equal-weight Hengduan and Haizi chronology index selected in
  Round 192. Each chronology is standardized to its complete 1940--2000 mean
  and sample standard deviation; no historical fitting is allowed.
- Benchmark: Wang et al. (2015) ETP JJA reconstruction, standardized only for
  plotting and correlation. It is a partially source-overlapping regional
  trajectory benchmark, not independent validation.

## Frozen diagnostics and gate

- Annual, centered 31-year, and eight non-overlapping 50-year means over
  1500--1899.
- The historical shape gate requires 50-year correlation at least 0.30,
  centered 31-year correlation at least 0.20, and at least five of seven
  adjacent 50-year changes with the same sign.
- Individual chronology diagnostics are reported but do not change the gate.

## Locked parser

- Decode year from column A first.
- Skip 1921--1939 rows before decoding column B.
- Decode chronology values only for 1500--1899 and 1940--2000.

## Hard boundaries

- Do not access or hash Jionglacuo or any alias.
- Do not use Wang to calibrate amplitude, weights, signs, seasons, or lags.
- Passing retains a mechanism candidate only; it does not authorize a degrees-C
  correction, EnKF, production reconstruction, or OGGM.

