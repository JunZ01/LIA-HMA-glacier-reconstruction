# Round 240 contract: CRU TS4.01 Tmean same-version acquisition

## Purpose

Acquire the exact-version CRU monthly mean-temperature files needed to
separate the Round 239 CRU-version difference from the Berkeley product
difference.

## Frozen scope

- Official source: CRU TS4.01 `tmp` decade archives.
- Download only 1951-1960, 1961-1970, 1971-1980, 1981-1990, and 1991-2000.
- Preserve each official gzip byte stream unmodified in `raw_unopened`.
- Decompress each archive byte-for-byte into a separate `extracted` directory.
- Verify gzip and NetCDF magic, size, and SHA256.
- Do not parse any climate magnitude in this acquisition round.

No file containing 1921-1939 is requested. Original project files are
read-only. No proxy, Jionglacuo, EnKF, production reconstruction, correction,
or OGGM work is allowed.
