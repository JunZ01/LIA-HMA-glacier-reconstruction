# Round 235 Contract: NOAA HMA Tree-Ring Time-Filter Gap Inventory

## Purpose

Identify central/eastern HMA tree-ring studies omitted by Round 171 because its
query required a record to span the full 1500--2000 interval. The motivating
case is the 2026 Central Himalayan MXD network, which starts after 1500 but can
directly inform the unresolved 1850--1950 temperature-shape problem.

## Frozen query and comparison

- NOAA/WDS Paleo Search data type 18, reconstructionsOnly=N.
- Spatial box: 25--40 N, 80--105 E.
- No time filter.
- `headersOnly=true`; do not request or open linked data files.
- Compare returned NOAA study IDs with the exact Round 171 candidate inventory.
- Flag names containing MXD, density, temperature, Nahtz, Rara, Chola,
  Cholangpati, Nepal, Himalaya, Tibet, Bhutan, or Hengduan for metadata review.

## Hard boundaries

- Save the raw API response unmodified and hash it.
- Do not download or parse annual proxy or climate values.
- Do not access numeric values for 1921--1939.
- Do not open or hash Jionglacuo.
- Do not run a PSM, EnKF, production reconstruction, climate correction, or
  OGGM simulation.
- Preserve all existing source hashes.

