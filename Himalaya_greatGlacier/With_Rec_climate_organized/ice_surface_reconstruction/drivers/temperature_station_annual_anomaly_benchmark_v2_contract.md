# Temperature Station Annual Anomaly Benchmark v2

This contract inherits every scientific and safety condition in
`temperature_station_annual_anomaly_benchmark_v1_contract.md`.

V2 changes one pre-execution implementation assumption only: GHCN-M is allowed
to omit station-year rows when a station has no submitted values. Missing rows
are measured as coverage gaps and are not treated as parser failures. The
parser must still access complete 12-cell source rows only, exercise the
1921-1939 pre-value guard, and apply every baseline and annual-completeness gate
from v1.
