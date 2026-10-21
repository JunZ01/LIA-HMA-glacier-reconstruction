"""Append the formal Core-191 dual-gate inventory result to project facts."""

from __future__ import annotations

import os
from pathlib import Path


OLD_HEADER = b"Last updated: 2026-08-14 (analog glen_a block completed and closed)"
NEW_HEADER = b"Last updated: 2026-08-15 (formal dual-gate inventory completed)"

SECTION = """## 22. Core-191 formal dual-gate inventory after analog blocks (2026-08-15)

- A read-only inventory merged 2428 deduplicated saved controller chains,
  including the frozen-991 continuation and both completed analog blocks. It
  covers 186 systems with saved chains and all 191 systems in the routing
  table. No OGGM model was advanced.
- Sixty-one systems have at least one stable chain passing both legacy 10%
  total-length endpoints, but only three systems pass both the formal 20%
  separation endpoints and the total-length endpoints: `991`, `1077`, and
  `1616`. The other 55 total-length-only systems must not be described as
  formal curve matches.
- System `1077` passes with task
  `1077-B_curve_loop_delta_refine_1077_r1_v1-72d31c217d404a04`,
  `delta_t=-0.03125 K`, Stage-2 physical parameters, forward peak `11.600 km`
  versus LIA `11.768 km`, and year-2000 length `10.600 km` versus RGI
  `10.480 km`. Its separation errors are `0.131/0.093`, total-length errors
  `1.432%/1.150%`, amplitude ratio `0.776`, and spin-up stability `0.319%`.
- System `1616` passes with task
  `1616-B_delta_boundary_extension_v1-a3f867943d22c213`,
  `delta_t=+0.75 K`, Stage-2 physical parameters, forward peak `12.350 km`
  versus LIA `12.779 km`, and year-2000 length `9.350 km` versus RGI
  `9.254 km`. Its separation errors are `0.122/0.027`, total-length errors
  `3.360%/1.041%`, amplitude ratio `0.851`, and spin-up stability `1.451%`.
- The mutually exclusive inventory routes are 100 `no_formal_pass`, 55
  `total_length_only`, 28 `small_separation_review`, three `formal_dual_pass`,
  two `blocked_dynamic_review`, two `no_stable_valid_chain`, and one manual
  provisional candidate outside the controller score sources (`1044`).
- Separate same-scale spin-up and forward length plots were generated for
  `991`, `1077`, and `1616`. System 991 retains its already approved frozen
  chain; systems 1077 and 1616 are the next two provisional freeze candidates
  pending curve review. Do not launch a new OGGM parameter batch before this
  review.
- Inventory evidence and plots are in
  `validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/diagnostics/dual_gate_inventory_after_analog_v1/`.
  Reusable code is
  `Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/build_lia_191_dual_gate_inventory_after_analog_v1.py`
  and
  `plot_lia_191_formal_dual_candidates_v1.py`.
"""


def main() -> None:
    path = Path(__file__).resolve().parents[4] / "LIA_PROJECT_FACTS.md"
    original = path.read_bytes()
    section_bytes = SECTION.encode("utf-8")
    if section_bytes in original:
        print("formal dual-gate inventory facts section already present")
        return
    if original.count(OLD_HEADER) != 1:
        raise ValueError("expected exactly one preceding Last updated header")
    updated = original.replace(OLD_HEADER, NEW_HEADER, 1)
    updated = updated.rstrip(b"\r\n") + b"\n\n" + section_bytes + b"\n"
    temporary = path.with_name(path.name + ".tmp_dual_inventory_v1")
    temporary.write_bytes(updated)
    os.replace(temporary, path)
    print(f"updated {path}")


if __name__ == "__main__":
    main()
