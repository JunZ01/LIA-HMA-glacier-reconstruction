# Round 195 Contract: SigFree Toolchain Preflight

## Purpose

Determine whether the official Windows SigFree program suite provides a
versioned, scriptable path to reproduce the published five-site Hengduan
chronology before any DLL-excluded chronology is constructed.

## Source

- Official University of Cambridge dendrochronology software page.
- Windows suite URL:
  `https://www.geog.cam.ac.uk/wp-content/uploads/Windows7_Prgms-1.zip`

## Scope

- Save the ZIP unchanged in an independent Round 195 directory and record its
  SHA256.
- Inspect archive member names and extract with path-traversal protection.
- Do not execute any binary in this round.
- Do not read any tree-ring, chronology, climate, locked-period, or Jionglacuo
  value.

## Decision

Advance only if SigFree and sufficient documentation/example control files are
present to predeclare all chronology settings and run reproducibly.

