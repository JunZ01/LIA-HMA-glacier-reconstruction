# Temperature Station Annual Anomaly Benchmark v1

## Objective

Construct a station-anchored annual temperature-anomaly benchmark for the 15
Round-277 GHCN-M stations and compare it with the six frozen production
temperature members. CRU and Berkeley Earth are background processing products,
not independent votes.

## Frozen data boundary

- Historical values may be read only for 1850-1920 and 1940-1950.
- The 1961-1990 interval may be read only to calculate source-native baselines.
- Years 1921-1939 must be rejected from the value path after decoding station,
  element, and year, but before decoding any monthly value or flag.
- Original GHCN-M archives, production GeoTIFFs, climate NetCDF, CSV, and project
  files are read-only.
- No correction is fitted or applied. No production climate, EnKF, or OGGM run
  is authorized.

## Comparison contract

- GHCN-M qcf is primary. qcu is a homogenization-sensitivity series from the
  same physical observations and is never counted as independent evidence.
- The GHCN-M fixed-width integer scale is 0.01 degree Celsius. A synthetic unit
  and locked-year poison test must pass before source values are parsed.
- A station-series baseline is valid only when every calendar month has at
  least 20 accepted values in 1961-1990.
- Annual station anomalies are day-weighted means of the 12 monthly anomalies.
  A station-year is retained only when all 12 months are accepted.
- The production reconstruction is annual. Therefore the primary comparison is
  annual only; no monthly or seasonal equivalence is claimed.
- Production members and background products are sampled bilinearly at the
  official station coordinates and centered on their own 1961-1990 values at
  the same coordinates.

## Required outputs

- Unit, locked-gap, source-hash, and baseline-coverage audits.
- Monthly station climatology and annual qcf/qcu anomalies.
- Station-year values for all six production members and their mean.
- Station-year Berkeley and CRU background values where available.
- Per-station qcf/qcu sensitivity and product metrics.
- Fifteen station curves, west/central/east matched-support curves, source
  manifest, machine-readable summary, and FINAL_RESULT.md.
