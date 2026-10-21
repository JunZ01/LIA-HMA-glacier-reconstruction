# Temperature Early-Station Homogeneity Audit v1

## Objective

Determine whether the 1880-1920 warm residual found by the station anomaly
benchmark is a common project-field feature or an artifact of one station,
one country, one source lineage, or GHCN-M homogenization.

## Frozen scope

- Select stations only from the completed benchmark using at least five matched
  qcf/project years in 1901-1920. Do not select from residual magnitude.
- Use only derived allowed-period values from the benchmark and Round 277.
- Assert that no input row contains 1921-1939.
- qcf remains primary; qcu is same-observation homogenization sensitivity.
- CRU and Berkeley are not used in this homogeneity decision.
- No production climate, correction, EnKF, or OGGM run is authorized.

## Required tests

- Exact NOAA inventory identity and legacy-XLT status.
- Official monthly source-flag composition.
- qcf-qcu annual adjustment level, trend, and bounded one-split diagnostic.
- Project-minus-qcf and project-minus-qcu residuals on identical station-years.
- Leave-one-station and leave-one-country residual robustness.
- Per-station evidence grade based on coverage and qcf/qcu stability.
- Station curves, adjustment curves, tables, manifest, and final decision.
