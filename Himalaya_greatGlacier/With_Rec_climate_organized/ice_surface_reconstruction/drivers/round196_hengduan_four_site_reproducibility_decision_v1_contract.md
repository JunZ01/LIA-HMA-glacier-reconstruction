# Round 196 Contract: Hengduan Four-Site Reproducibility Decision

## Purpose

Decide whether the DLL-excluded four-site Hengduan chronology can be constructed
without introducing an uncontrolled chronology algorithm.

## Required equivalence

Any reconstruction method must first reproduce the published five-site
`SsfStb` chronology under the documented RCSigFree settings before it may be
applied to FGH+KQH+XYS+YPL.

## Evidence

- Publisher method: power transform; Friedman variable-span detrending with
  alpha 7 in a signal-free workflow; biweight robust mean; Osborn-style
  variance stabilization; 51-year moving EPS/Rbar; EPS >= 0.85 and at least six
  cores from three trees.
- Official Cambridge Windows suite acquired in Round 195 v3.
- Static binary metadata only; do not execute the binary in this round.

## Decision boundary

If no versioned control file, example, or documented non-interactive interface
is present, do not substitute a newly written approximation. Close the four-site
branch pending explicit execution approval or an author-supplied chronology or
control file.

