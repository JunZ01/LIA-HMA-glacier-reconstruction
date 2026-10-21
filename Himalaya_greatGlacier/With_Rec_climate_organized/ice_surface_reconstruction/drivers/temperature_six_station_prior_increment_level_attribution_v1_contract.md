# Six-station prior/increment level attribution contract

- Diagnose the six independently selected western/central-western GHCN-M stations from the completed homogeneity audit.
- Use GHCN-M `qcf` as the primary station target and `qcu` only as a homogenization sensitivity.
- Express station observations, posterior fields, and recovered backgrounds as anomalies relative to their own 1961-1990 climatology.
- Decompose every archived project member exactly as `posterior = background + assimilation increment` at each station.
- Assess 1880-1900, 1901-1920, and 1940-1950 only where station years are finite; do not infer 1850-1879 without direct support.
- Read only 1880-1920, 1940-1950, and 1961-1990 from posterior and proxy sources. Never request or read 1921-1939.
- Reuse the archived fixed and strategy-C adaptive background definitions already validated by Round 25. This is a read-only decomposition, not an EnKF rerun.
- Do not modify production reconstruction, source data, or OGGM inputs.

