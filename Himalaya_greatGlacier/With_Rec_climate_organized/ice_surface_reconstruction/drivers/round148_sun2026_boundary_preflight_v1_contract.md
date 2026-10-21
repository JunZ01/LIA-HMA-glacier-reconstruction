# Round 148 Sun 2026 NOAA Boundary Preflight V1

## Objective

Create an allowed-period working copy of the newly acquired NOAA Sun 2026 Tibetan temperature reconstruction without decoding any 1921-1939 value.

## Frozen parser

- Read the raw text one line at a time.
- Classify a data row only when its first whitespace-delimited token is an integer year from 500 through 2500.
- If that year is 1921-1939, discard the complete line before splitting or decoding any remaining token.
- Decode and export numeric fields only for 1000-1920 and 1940-2000.
- Copy non-data header/metadata lines separately; they contain no annual value row by definition.
- Stop if an allowed data row has a nonnumeric field or if row width changes.
- Record source hashes before and after execution.

The file remains a PSM-target candidate, not independent validation. This round does not inspect Jionglacuo, fit a PSM, reconstruct climate, run EnKF, or run OGGM.
