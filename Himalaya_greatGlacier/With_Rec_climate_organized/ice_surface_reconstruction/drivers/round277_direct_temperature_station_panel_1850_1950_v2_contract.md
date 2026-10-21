# Round 277 direct temperature station panel v2 contract

This contract is identical to the v1 time-access, source, station, and
interpretation contract, with one mandatory correction:

- GHCN-M v4 fixed-width monthly temperature integers are hundredths of degrees
  Celsius and must be converted using `value / 100.0`.

The v1 derived panel used `value / 10.0` and is invalid for all temperature and
qcu-qcf adjustment magnitudes. V2 must be built in a new independent result
directory. It must not overwrite or silently repair v1. Source archives and
their hashes remain valid; the error is confined to the v1 derived unit scale.

All locked-period requirements remain unchanged: allow only 1850-1920 and
1940-1950, discard 1921-1939 after decoding year and before reading any monthly
cell, and prove the guard with a synthetic poison test.

