# Round 167: CHIN074 Qamdo 2 identity and chronology recovery

## Purpose

Determine whether CHIN074 provides genuinely new eastern-HMA temperature evidence or reuses the CHIN046/CHIN047 Qamdo material already represented in the archived proxy network.

## Frozen scope

- Acquire official NOAA/ITRDB CHIN074, CHIN046, and CHIN047 files into a new experiment directory.
- Preserve downloaded bytes unchanged under `raw_unopened` and record hashes and URLs.
- Audit source identity from headers, sample identifiers, species, site metadata, and publication metadata.
- Do not inspect or emit numeric observations from 1921-1939.
- Do not inspect a paleo reconstruction trajectory in this identity stage.
- Do not modify original MAT, NetCDF, CSV, MATLAB, or downloaded source files.
- Do not run production EnKF, production reconstruction, or OGGM.

## Decision gate

Reject CHIN074 as new evidence if its constituent samples are wholly reused from CHIN046/CHIN047 or if an exact/near-alias chronology is already present in the project proxy network. If material independence remains plausible and an official standardized chronology is available, advance only to a modern-period OOF test using allowed observations from 1940-2000 and the frozen Round 166 dual-product criteria.

## Interpretation limit

Passing this stage establishes source novelty only. It does not establish temperature skill, PSM stability, or eligibility for production reconstruction.
