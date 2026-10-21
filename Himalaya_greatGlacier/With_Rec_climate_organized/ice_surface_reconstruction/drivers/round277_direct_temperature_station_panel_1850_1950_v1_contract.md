# Round 277 direct temperature station panel contract

## Purpose

Acquire and safely preprocess a direct GHCN-M v4 temperature station panel for
diagnosing the HMA temperature reconstruction over 1850-1920 and 1940-1950.
This experiment does not alter, correct, or rerun the production reconstruction.

## Frozen sources

- NOAA GHCN-M v4 latest unadjusted monthly mean temperature archive (`qcu`).
- NOAA GHCN-M v4 latest adjusted monthly mean temperature archive (`qcf`).
- NOAA GHCN-M country metadata.
- Inventory metadata embedded in both official archives.

The downloaded bytes are stored under `raw_unopened`, hashed, and made
read-only. Extracted source members are stored under `prepared`, independently
hashed, and made read-only. A latest URL is acceptable only because response
headers, byte count, retrieval UTC time, and SHA256 are frozen in the manifest.

## Time-access contract

- Allowed value intervals: 1850-1920 and 1940-1950, inclusive.
- Locked interval: 1921-1939, inclusive.
- Every fixed-width data line must decode station ID and year first.
- A locked-year line must be discarded before the element field, monthly cell,
  monthly value, or monthly flag is read.
- The parser must pass a synthetic poison test in which all monthly cells in a
  locked-year line are deliberately nonnumeric. The same poison in an allowed
  year must raise, proving the test exercises the guarded value path.

## Stations

Tier A: Leh, Shimla, Darjeeling, Murgab, Mukteswar, Srinagar, Khorog, Lhasa.

Tier B: Naryn, Quetta, Dera Ismail Khan, Yunnan Fou, Hotan, Qamdo, Murree.

## Interpretation

`qcu` and `qcf` are two processing states of the same GHCN-M station evidence,
not independent climate records. Their difference diagnoses homogenization and
quality-control sensitivity. Gridded products that reuse these stations must
not be counted as additional independent observations.

## Outputs

The independent result directory must contain the preserved sources, extracted
members, station metadata, long-form qcu/qcf monthly panel, coverage summary,
qcu-qcf lineage diagnostics, parser audit, poison-test result, source manifest,
summary, lineage notes, and `FINAL_RESULT.md`.

