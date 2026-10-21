# Round 237 Contract: NPL065 Source Acquisition Preflight

## Purpose

Acquire the official NOAA NPL065 raw RWL file unchanged and decide whether its
identity and structural modern coverage permit a boundary-safe blocked OOF test.

## Frozen candidate

- NOAA/WDS study 40542, xmlId 81629, ITRDB NPL065.
- Site: Gaurishankar Conservation Area, 27.89 N, 86.37 E.
- Species: Rhododendron campanulatum.
- Published target: December--February minimum temperature.
- Published calibration: 1960--2013; reported r=0.781 and 61% explained variance.
- Official raw URL:
  `https://www.ncei.noaa.gov/pub/data/paleo/treering/measurements/asia/npl065.rwl`.

## Allowed operations

- Download and store the HTTP payload unmodified in `raw_unopened`.
- Hash the payload and record HTTP metadata.
- Parse only Tucson series identifiers, row start years, token positions, and
  end-of-series markers to inventory structural coverage.
- Skip 1921--1939 slots before any numeric conversion.
- Read the existing proxy evidence ledger metadata only and compute spatial
  distances and chronology-bound identity matches.

## Frozen advance gates

- HTTP success and nonempty source file.
- At least 30 distinct source series.
- Structural coverage has at least one non-marker slot for every year in
  1940--2000.
- NOAA metadata coverage reaches at least 2000.
- No exact existing formal source identity or coordinate/chronology-bound match.

## Hard boundaries

- Do not parse any ring-width magnitude in this round.
- Do not inspect or export any paleo trajectory.
- Do not access numeric values for 1921--1939.
- Do not open or hash Jionglacuo.
- Do not run a PSM, EnKF, production reconstruction, correction, or OGGM.
- Preserve all existing input hashes.

