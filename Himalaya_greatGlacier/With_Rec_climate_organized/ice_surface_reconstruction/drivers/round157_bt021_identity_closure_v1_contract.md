# Round 157 BT021 Identity and Role Closure Contract

## Objective

Determine whether the newly acquired NOAA ITRDB `BT021` Ura-Dhur chronology is
numerically distinct from every archived proxy6/proxy9 sequence, and close its
scientific role relative to the already parsed Bhutan 2015 temperature curve.

## Boundary and Parsing

- Use `bt021-crn-noaa.txt` as the canonical annual chronology source.
- Decode the first tab-delimited token as year. For 1921-1939, skip the row
  before converting any chronology or sample-count token to a number.
- Export only 1000-1920 and 1940-2000. Values after 2000 are outside this audit
  and are not decoded.
- Compare on exact common allowed years using the established Round 145 affine
  and near-exact standardized thresholds.
- The downloaded raw files and all archived proxy files remain unchanged.

## Decision

- Any exact or near-exact proxy6/proxy9 sequence match quarantines BT021 as an
  existing-network alias.
- If no match exists, BT021 may advance only as a new eastern-HMA fitting
  candidate to a separately contracted modern OOF test.
- Because Bhutan 2015 is derived from the Ura-Dhur/BT021 material, its published
  reconstruction remains same-source evidence and cannot independently validate
  a BT021-fitted candidate, regardless of numerical affine equivalence.

No PSM, paleo reconstruction, EnKF, OGGM, or locked-period analysis is allowed.
