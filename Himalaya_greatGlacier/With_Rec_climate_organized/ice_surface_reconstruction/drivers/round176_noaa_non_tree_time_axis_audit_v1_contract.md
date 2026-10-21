# Round 176 Contract: NOAA Non-Tree Time-Axis Audit

## Purpose

Close false-positive Round 175 metadata candidates using temporal resolution
and physical-variable metadata before any proxy or climate value is inspected.

## Frozen operations

- Download only official NOAA text files linked by Round 175 candidates.
- Preserve every downloaded file unchanged and record SHA256.
- Parse the NOAA data header and one identified time token per row only.
- Convert BP time to CE as `1950 - BP`.
- Skip rows mapped to 1921--1939 before any non-time token conversion.
- Count observations in 1940--2000; require at least 20 to advance.
- Exclude declared 100-year averages, tables without a row-level time axis, and
  oxygen-isotope products explicitly interpreted as precipitation or
  hydroclimate.

## Hard boundaries

- Do not convert, export, plot, correlate, or inspect any non-time data value.
- Do not access any locked-period non-time value or Jionglacuo value.
- Do not run OOF, EnKF, production reconstruction, correction, or OGGM.
- Do not modify Round 175 or any downloaded source.

## Decision

Frequency-qualified direct-temperature tables may advance to identity and
blocked modern OOF. Frequency-qualified oxygen-isotope tables require a
separate physical-orientation and identity decision before OOF.
