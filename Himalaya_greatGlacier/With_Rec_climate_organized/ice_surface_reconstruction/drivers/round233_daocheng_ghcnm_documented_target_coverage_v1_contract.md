# Round 233 Contract

## Question

Does the public GHCN-M QCU Daocheng station record have enough quality-controlled
monthly coverage to adjudicate the Haizi source-native chronology directly,
without relying on gridded Berkeley or CRU targets?

## Frozen parser

- Station ID: `CHXLT574247`, metadata location 29.05 N, 100.30 E, 3729 m.
- Target year uses previous October-December plus current January-September.
- Decode station ID, year, and element first. Skip 1921-1939 before examining
  any monthly field.
- Inspect only missing-value tokens and QCU quality flags. Do not convert or
  retain temperature magnitudes in this round.
- Candidate years: 1959-2017, matching the published Haizi calibration period.

## Frozen coverage gate

Advance to direct-station blocked OOF only if all conditions hold:

1. At least 45 complete cross-year targets in 1959-2017.
2. At least 18 complete targets in both 1959-1988 and 1989-2017.
3. Each of five chronological blocks (1959-1970, 1971-1982, 1983-1994,
   1995-2006, 2007-2017) contains at least six complete targets.
4. No run of missing target years exceeds three consecutive years.

No threshold may be relaxed after observing coverage.

## Safety

No temperature magnitude, paleo chronology value, 1921-1939 value,
Jionglacuo value, EnKF, production climate, or OGGM state may be accessed.

