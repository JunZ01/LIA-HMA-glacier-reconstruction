# Round 236 Contract: Central/East HMA NOAA Candidate Metadata Audit

## Purpose

Resolve the data type, documented climate role, coverage metadata, and public
file availability of ten central/eastern HMA records omitted by the Round 171
full-1500-span query.

## Frozen studies

- NEPA001 / NOAA 4422 / xmlId 5632
- NEPA002 / NOAA 4453 / xmlId 6975
- NEPA037 / NOAA 3794 / xmlId 4254
- NEPA043 / NOAA 22394 / xmlId 20522
- NEPA055 / NOAA 22408 / xmlId 20550
- NPL065 / NOAA 40542 / xmlId 81629
- Ganesh oxygen isotope / NOAA 22551 / xmlId 20722
- Humla oxygen isotope / NOAA 22547 / xmlId 20714
- Southern Himalaya oxygen isotope / NOAA 23830 / xmlId 21974
- Wache Bhutan oxygen isotope / NOAA 22548 / xmlId 20716

## Allowed operations

- Fetch each exact NOAA full-study JSON by xmlId.
- Save each response unmodified and hash it.
- Extract scalar metadata paths and values for study notes, publications,
  sites, variables, temporal coverage, and linked-file descriptions.
- Record linked URLs as metadata strings only.

## Hard boundaries

- Do not follow or open any linked data-file URL.
- Do not download or parse annual proxy or climate values.
- Do not access numeric values for 1921--1939.
- Do not open or hash Jionglacuo.
- Do not run a PSM, EnKF, production reconstruction, correction, or OGGM.
- Preserve all existing input hashes.

