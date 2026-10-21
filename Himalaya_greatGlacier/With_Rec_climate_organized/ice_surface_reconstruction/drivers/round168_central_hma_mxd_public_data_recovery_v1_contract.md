# Round 168 central-HMA 2026 MXD public-data recovery

## Objective

Determine whether Nahtz et al. (2026), DOI `10.1007/s11676-026-02104-0`, publicly provides an annual central-HMA MXD chronology or temperature-reconstruction sequence suitable only for independent validation over allowed years 1775-1920 and 1940-2000.

## Frozen role

- This record cannot be a production reconstruction candidate because its reliable reconstruction begins in 1775, later than the frozen 1500 gate.
- If an annual numeric series is publicly available, it may enter only as independent 1775-1920 validation evidence after source-overlap auditing.
- It must never be fitted, assimilated, used to choose a correction, or used to inspect 1921-1939.

## Acquisition and inspection

- Save official publisher files unchanged in an independent `raw_unopened` directory and record SHA256.
- Inspect PDF metadata, page count, embedded-file inventory, and sanitized text headings/data-availability statements only.
- Any extracted text line containing a year from 1921 through 1939 is discarded before output or numeric conversion.
- Do not digitize curves, OCR figures, infer annual values, or treat a plotted reconstruction as machine-readable data.

## Decision

- `advance_public_annual_series_to_identity_audit` only if an official annual numeric chronology/reconstruction file is found.
- `retain_as_metadata_only_external_evidence` if only article figures or nonnumeric supplementary material are public.
- No EnKF, production reconstruction, Jionglacuo access, OGGM, or paleo comparison is authorized in this round.
