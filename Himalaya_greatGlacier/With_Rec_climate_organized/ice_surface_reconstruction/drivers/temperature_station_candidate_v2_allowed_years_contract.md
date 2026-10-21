# Temperature Candidate v2 allowed-year contract

- Build an independent research candidate only for 1880-1920 and 1940-1950; omit 1921-1939 entirely.
- Do not infer or correct 1850-1879 because the strict direct-station benchmark has insufficient support.
- Use qcf as the primary target and qcu only as a sensitivity diagnostic.
- Use a common western residual amplitude with a fixed 500 km Gaussian distance taper from eligible stations.
- Fit a linear amplitude across the contiguous 1880-1920 block from the independently summarized 1880-1900 and 1901-1920 station residual means.
- Estimate 1940-1950 independently and do not interpolate through the sealed gap.
- Apply the same correction to all six archived members, preserving their spread.
- Write a new allowed-year NetCDF; do not modify production GeoTIFFs, EnKF inputs, or OGGM inputs.

