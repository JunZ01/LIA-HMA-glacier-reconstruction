# Round 191 Contract: CHIN041--Hengduan Sample Identity

## Purpose

Resolve whether the Wang/PAGES CHIN041 chronology and the five-site Hengduan
candidate reuse any exact tree/core sample identifiers.

## Frozen source

- Official NOAA ITRDB raw measurement URL:
  `https://www.ncei.noaa.gov/pub/data/paleo/treering/measurements/asia/chin041.rwl`
- The five preserved Hengduan raw RWL files from Round 178.

## Parser boundary

- Read only the first 12 ASCII bytes of each RWL line: eight-byte sample ID and
  four-byte segment-start year.
- Do not parse, convert, retain, or inspect any ring-width token.
- Segment-start years are used only to count lines whose decades intersect the
  locked period; the associated value fields remain unread.

## Decision boundary

- Any normalized sample-ID match proves direct sample reuse.
- Zero exact matches excludes direct sample reuse but does not make the nearby
  chronologies statistically independent climate evidence.

## Hard boundaries

- Do not access or hash Jionglacuo or any alias.
- Do not decode 1921--1939 ring-width values.
- Do not run OOF fitting, EnKF, correction, production reconstruction, or OGGM.

