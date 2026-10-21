"""Append the verified formal delta-T batch-3 result to the project facts contract."""

from __future__ import annotations

from .build_lia_191_iterative_loop import project_root


MARKER = "## 30. Formal delta-T edge screening batch 3 and continuous-batch stop (2026-08-15)"

TEXT = r"""

## 30. Formal delta-T edge screening batch 3 and continuous-batch stop (2026-08-15)

- The user enabled continuous batched execution with automatic freezing of
  formal passes. Systems `2939` and `4319` from batch 2 were first frozen and
  their package hashes reproduced. A read-only inventory then contained 2443
  deduplicated chains and eight formal systems: `991`, `1077`, `1616`, `2815`,
  `2902`, `2928`, `2939`, and `4319`.
- The four remaining bounded delta-T edge candidates were probed before any
  dynamics. Existing outlet gates pass systems `2266`, `2337`, and `4628` and
  fail system `2748`; no override was applied. Spatial-downstream-v2 geometry
  was built only for the three passing systems. All three preserve the Core-191
  initial geometry and pass the static and 0.1/5-year zero-MB gates.
- Exactly one predicted midpoint was then run for each passing system with only
  delta-T changed and all Stage-2 physics, forcing, B-window, and observation
  definitions frozen. All three chains are technically clean, adaptive-spin-up
  stable, and observation-domain valid.
- System `2337` at `delta_t=-0.36708192 K` passes the full formal dual gate. Its
  forward peak/year-2000 lengths are `14.050/11.250 km`; its separation errors
  are `0.1192/0.0905`, total-length errors are `2.3865%/2.2663%`, and spin-up
  stability is `0.8626%`. It is automatically frozen under the user-approved
  continuous-batch rule.
- System `2266` at the midpoint `delta_t=+0.65404781 K` passes the LIA endpoint
  but misses the RGI separation endpoint narrowly (`0.0997/0.2131`). The one
  informative warmer-side point `+0.66408426 K` was therefore run; its
  year-2000 length remains `11.050 km` and it still misses the RGI separation
  gate (`0.1284/0.2131`). The bounded delta-T branch is closed. System `4628`
  at `delta_t=+0.14017630 K` misses by `0.2570/1.2192` and is also closed.
- The final read-only inventory contains 2447 deduplicated chains and exactly
  nine formal systems: `991`, `1077`, `1616`, `2337`, `2815`, `2902`, `2928`,
  `2939`, and `4319`. Its routes are 99 `no_formal_pass`, 50
  `total_length_only`, 28 `small_separation_review`, nine `formal_dual_pass`,
  two `blocked_dynamic_review`, two `no_stable_valid_chain`, and one manual
  provisional candidate outside controller sources. The frozen registry
  contains the same nine systems and all recorded hashes reproduce.
- The declared bounded Stage-2-physics delta-T interpolation/edge-extrapolation
  pool is now exhausted. Continuous execution stops here because the next OGGM
  block would open a new physical parameter family or change a scientific
  routing decision; do not launch that block automatically.
- Evidence is in
  `validation_data/results/paleo_surface_validation_v0/lia_191_formal_delta_bisection_batch3_v1/`,
  `lia_191_formal_delta_bisection_batch3b_2266_v1/`,
  `lia_downstream_geometry_probe_formal_delta_bisection_batch3_v1/`,
  `lia_elevation_band_spatial_downstream_formal_delta_bisection_batch3_v1/`,
  `lia_191_controller_v1/diagnostics/dual_gate_inventory_after_batch3_v1/`,
  and `lia_191_frozen_curve_registry_v1/`.
"""


def main() -> None:
    path = project_root() / "LIA_PROJECT_FACTS.md"
    current = path.read_text(encoding="utf-8")
    if MARKER in current:
        print("batch-3 facts already present")
        return
    path.write_text(current.rstrip() + "\n" + TEXT.lstrip(), encoding="utf-8")
    print("appended formal delta-T batch-3 facts")


if __name__ == "__main__":
    main()
