# Round 242 contract: NPL065 paper method recovery

## Purpose

Recover the exact chronology construction, instrumental target, calibration,
and verification definitions behind the published NPL065 winter-temperature
relationship.

## Source and handling

- Download the open institutional-repository PDF associated with DOI
  `10.1016/j.dendro.2020.125799`.
- Preserve the received PDF byte stream unmodified in `raw_unopened` and record
  its URL, size, SHA256, and PDF page count.
- Extract page text into a separate `extracted_text` directory for method
  audit. Text extraction is derivative; the raw PDF is never modified.
- Do not infer an unpublished chronology or target value from a figure.

This round reads no project proxy, climate, reconstruction, locked-period, or
Jionglacuo value and runs no EnKF, correction, production reconstruction, or
OGGM simulation.
