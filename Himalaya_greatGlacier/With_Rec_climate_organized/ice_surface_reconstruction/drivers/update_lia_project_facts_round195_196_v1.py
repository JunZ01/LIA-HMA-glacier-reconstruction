from __future__ import annotations

from pathlib import Path


SECTION = r'''

## 322. Official SigFree toolchain and four-site reproducibility boundary (2026-08-21)

- Round 195 v1 was an incomplete HTTP transfer and v2 completed the official
  Cambridge Windows suite download but failed only while extracting long macOS
  metadata paths. Version 3 reused the complete v2 archive read-only, passed the
  ZIP CRC, and is the valid preflight. The official archive is 16,961,998 bytes
  with SHA256 `a87cec8786f24ec67b9373d8ac222e60036a28a0797059caaf1238566c6f5e1d`.
- The SigFree folder contains `RCSigFree_2019_MRWE.exe` (SHA256
  `e8a380642ae94a72c136abfc344392f74215acdcefb3368b90d957dc0960485e`)
  and the Melvin 2008 method paper, but no example data, control file, or
  parameter template. Static inspection shows an interactive Fortran program
  with graphics prompts and no Windows version resource. The binary was not
  executed.
- The published Hengduan chronology requires a data-adaptive power transform,
  Friedman variable-span detrending with alpha 7 in RCSigFree, biweight robust
  averaging, Osborn-style variance stabilization, and a 51-year EPS/Rbar gate.
  These choices directly affect low-frequency behavior.
- Round 196 therefore closes the proposed DLL-excluded FGH+KQH+XYS+YPL branch
  for the current workflow. It may reopen only after explicit authorization to
  execute the official binary and first reproduce the published five-site
  `SsfStb` chronology, or after an author-supplied control file or four-site
  chronology is obtained. A hand-written approximate signal-free chronology is
  not admissible because algorithmic low-frequency differences could be
  mistaken for climate signal.
- No tree-ring, chronology, climate, 1921-1939, or Jionglacuo value was read in
  Rounds 195-196, and no EnKF or OGGM run occurred. Evidence is in
  `round_195_sigfree_toolchain_preflight_v3/` and
  `round_196_hengduan_four_site_reproducibility_decision_v1/`.
'''


def main() -> int:
    root = next(parent for parent in Path(__file__).resolve().parents if (parent / "LIA_PROJECT_FACTS.md").exists())
    path = root / "LIA_PROJECT_FACTS.md"
    text = path.read_text(encoding="utf-8")
    if "## 322. Official SigFree toolchain" in text:
        raise RuntimeError("round195-196-facts-already-present")
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Last updated:"):
            lines[index] = "Last updated: 2026-08-21 (Rounds 190-196: dual index closed; exact four-site chronology awaits authorized SigFree reproduction)"
            break
    path.write_text("\n".join(lines).rstrip() + SECTION + "\n", encoding="utf-8")
    print("UPDATED_LIA_PROJECT_FACTS_ROUNDS_195_196")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

