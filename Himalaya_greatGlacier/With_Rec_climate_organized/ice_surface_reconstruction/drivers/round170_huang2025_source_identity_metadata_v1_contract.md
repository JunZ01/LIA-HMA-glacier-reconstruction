# Round 170 Huang 2025 Source-Identity Metadata Contract

## Question

Can the public Huang et al. (2025) supplementary DOCX establish whether the
Ranwu maximum-latewood-density chronology is newly sampled evidence or overlaps
an already assessed Sygera/Ranwu/Duan source family?

## Frozen Inputs

- Round 169 official Elsevier supplementary DOCX, unchanged.
- Project metadata and previously verified identity summaries only.

## Locked-Period Protection

- Any OOXML paragraph containing a standalone year from 1921 through 1939 is
  discarded before output or numeric conversion.
- No figure is rendered, OCR-processed, or digitized.
- No annual proxy, climate, reconstruction, or glacier value is parsed.

## Outputs

- Safe keyword-matched paragraph inventory.
- OOXML media names, hashes, pixel dimensions, and relationship metadata.
- A source-identity decision limited to what the public metadata establishes.

## Decision Rule

- `distinct_source_established` only if the public metadata identifies new
  samples/sites and excludes overlap with the existing source families.
- `overlap_established` only if a direct sample/site/accession match exists.
- Otherwise `source_identity_unresolved_request_raw_data_and_crosswalk`.

This round cannot validate a reconstruction, select a PSM, inspect a paleo
trajectory, or authorize EnKF/OGGM work.
