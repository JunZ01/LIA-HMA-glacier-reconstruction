# Round 171 NOAA HMA Raw-Temperature Inventory Contract

## Question

Does the current anonymous NOAA tree-ring catalogue contain an unassessed raw
central/eastern HMA temperature-sensitive chronology that spans at least
1500-2000 and can proceed under the Round 166 gate?

## Frozen Search

- NOAA data type: tree ring (`18`).
- Reconstructions excluded.
- Latitude 25-40 N, longitude 80-105 E.
- Study coverage must span 1500-2000 CE.
- Current NOAA headers-only API response.

## Allowed Inputs

- NOAA study headers and ITRDB site metadata only.
- The verified Round 110 proxy-evidence metadata ledger.
- Existing project decisions for Qamdo2, Hengduan, Maxiong, and Rara.

## Protection

- No annual proxy or climate value is downloaded or parsed.
- No 1921-1939 or Jionglacuo value is accessed.
- No candidate is selected from paleo behavior.

## Decision

A record may advance only if it is a site-level raw chronology, is absent from
the existing network and closed candidate families, has a documented positive
temperature interpretation and season, and meets the temporal coverage gate.

This inventory cannot authorize EnKF, OGGM, or production reconstruction.
