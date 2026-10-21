"""Record the reviewed 1077 and 1616 curve freezes in project facts."""

from __future__ import annotations

import os
from pathlib import Path


OLD_HEADER = b"Last updated: 2026-08-15 (formal dual-gate inventory completed)"
NEW_HEADER = b"Last updated: 2026-08-15 (systems 1077 and 1616 curves frozen)"

SECTION = """## 23. Systems 1077 and 1616 frozen curves (2026-08-15)

- After reviewing the separate complete spin-up and 1470-2000 length plots,
  the user approved continuing. Systems `1077` and `1616` are therefore frozen
  as provisional historical curves alongside the already frozen system `991`.
  This copies completed trajectories and states only; no OGGM model was
  advanced and these are not final parameter estimates.
- System `1077` freezes task
  `1077-B_curve_loop_delta_refine_1077_r1_v1-72d31c217d404a04` at
  `delta_t=-0.03125 K` with Stage-2 physical parameters. Its formal separation
  errors are `0.131/0.093`, total-length errors are `1.432%/1.150%`, amplitude
  ratio is `0.776`, and spin-up stability is `0.319%`.
- System `1616` freezes task
  `1616-B_delta_boundary_extension_v1-a3f867943d22c213` at
  `delta_t=+0.75 K` with Stage-2 physical parameters. Its formal separation
  errors are `0.122/0.027`, total-length errors are `3.360%/1.041%`, amplitude
  ratio is `0.851`, and spin-up stability is `1.451%`.
- Both legacy source trajectories lack per-checkpoint observation-length
  columns. The completed fingerprint-level observation-domain audit proves
  both chains remain within the valid domain, so their saved raw model length
  equals the observation length used for freezing. Neither chain is clipped or
  altered.
- The versioned frozen registry now contains exactly systems `991`, `1077`,
  and `1616`. Each package contains the complete timeseries, compact
  length/area/volume trajectory, spin-up and year-2000 states, source task and
  summary, displayed plot, candidate record, and file hashes. All recorded
  hashes reproduce without mismatch.
- Registry:
  `validation_data/results/paleo_surface_validation_v0/lia_191_frozen_curve_registry_v1/`.
  Reusable freeze code:
  `Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/freeze_lia_191_candidates_1077_1616_v1.py`.
"""


def main() -> None:
    path = Path(__file__).resolve().parents[4] / "LIA_PROJECT_FACTS.md"
    original = path.read_bytes()
    section_bytes = SECTION.encode("utf-8")
    if section_bytes in original:
        print("1077/1616 freeze facts section already present")
        return
    if original.count(OLD_HEADER) != 1:
        raise ValueError("expected exactly one preceding Last updated header")
    updated = original.replace(OLD_HEADER, NEW_HEADER, 1)
    updated = updated.rstrip(b"\r\n") + b"\n\n" + section_bytes + b"\n"
    temporary = path.with_name(path.name + ".tmp_freeze_1077_1616_v1")
    temporary.write_bytes(updated)
    os.replace(temporary, path)
    print(f"updated {path}")


if __name__ == "__main__":
    main()
