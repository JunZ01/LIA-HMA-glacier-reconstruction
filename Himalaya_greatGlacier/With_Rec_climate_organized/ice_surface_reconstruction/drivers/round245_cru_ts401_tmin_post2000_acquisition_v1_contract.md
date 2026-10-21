# Round 245 contract: CRU TS4.01 Tmin post-2000 acquisition

## Purpose

Acquire the official CRU TS4.01 Tmin files required to reproduce the published
NPL065 1960-2013 calibration interval without touching the sealed period.

## Frozen scope

- Download only `2001-2010` and `2011-2016` NetCDF gzip archives from the
  official CRU TS4.01 `tmn` directory.
- Preserve official gzip bytes unmodified in `raw_unopened`; decompress into a
  separate `extracted` directory.
- Verify gzip/NetCDF magic, size, and SHA256.
- Parse no climate magnitude in this acquisition round.

No file containing 1921-1939 is requested. Original files remain read-only.
No proxy, Jionglacuo, EnKF, production reconstruction, correction, or OGGM work
is allowed.
