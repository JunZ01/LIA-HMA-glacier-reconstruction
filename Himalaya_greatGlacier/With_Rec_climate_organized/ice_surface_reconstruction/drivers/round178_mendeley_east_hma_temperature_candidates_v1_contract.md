# Round 178 Contract: Mendeley East-HMA Temperature Candidate Acquisition

## Purpose

Recover two public eastern-HMA temperature datasets that satisfy the temporal
and physical metadata gates and may contain genuinely new raw evidence.

## Frozen sources

- Mendeley Data `889n6ccx4z`, version 1, DOI `10.17632/889n6ccx4z.1`.
- Mendeley Data `s7t4zc34sr`, version 2, DOI `10.17632/s7t4zc34sr.2`.
- Use only the repository's anonymous `public-api` snapshot, file list, and
  file download URLs.

## Operations

- Save repository metadata responses unchanged.
- Save every published file unchanged in a dataset-specific `raw_unopened`
  directory.
- Verify every file against the publisher-supplied SHA256.

## Hard boundaries

- Do not open, render, parse, preview, or inspect any downloaded file.
- Do not access any annual value, locked-period value, or Jionglacuo value.
- Do not run identity matching, OOF, EnKF, correction, production
  reconstruction, or OGGM in this round.

## Decision

Hash-verified files advance to a separately contracted structural and source
identity audit. Neither dataset is independent evidence until that audit
passes.
