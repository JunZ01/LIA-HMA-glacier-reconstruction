# Round 175 v2 Contract: NOAA HMA Non-Tree Temperature Inventory

## Purpose

Test whether the current NOAA catalogue contains a genuinely new central or
eastern HMA non-tree temperature-sensitive record that can satisfy the frozen
Round 166 candidate gates.

## Frozen scope

- Data types: ice cores, paleolimnology, speleothems, and other collections.
- Geographic screen: 25--40 N and 80--105 E; broad regional polygons larger
  than five degrees are excluded from candidate status.
- Temporal metadata screen: the table must cover both 1500 and 2000 CE.
- Temperature screen: metadata must explicitly identify temperature or stable
  oxygen-isotope information.
- Existing proxy6/proxy9 coordinate-and-span matches do not count as new
  evidence.

## Hard boundaries

- Download and preserve NOAA JSON metadata only.
- Do not download or parse any annual data file in this round.
- Do not access any 1921--1939 or Jionglacuo value.
- Do not inspect a paleo trajectory.
- Do not run an EnKF reconstruction, production correction, or OGGM.
- Do not modify any declared source.

## Decision

Only metadata candidates advance to a separately contracted raw-file,
identity, temporal-resolution, and blocked-modern-OOF audit. Stable oxygen
isotopes are not presumed to be temperature proxies merely because they pass
this metadata screen.
