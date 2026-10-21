"""Append the verified no-formal-pass N2 batch result to the project facts."""

from __future__ import annotations

from .build_lia_191_iterative_loop import project_root


HEADER_OLD = "Last updated: 2026-08-15 (class round 2 closed; eleven curves remain frozen)"
HEADER_NEW = "Last updated: 2026-08-15 (N2 glen-a promotion closed; eleven curves remain frozen)"
SECTION = r"""

## 38. No-formal-pass N2 glen-a promotion (2026-08-15)

- The read-only router divided the 99 `no_formal_pass` systems by saved
  endpoint reachability: 13 N1 systems can reach both endpoints only on
  different chains, 40 N2 systems can reach only the RGI total-length
  endpoint, 11 N3 systems can reach only the LIA total-length endpoint, and
  35 N4 systems reach neither endpoint under the saved parameter families.
- Four N2 representatives (`1120`, `2580`, `3911`, and `1133`) passed the
  existing outlet and spatial-downstream-v2 geometry gates. Testing Stage-2
  `glen_a x0.75/x1.25` at each best stable RGI-reachable anchor produced eight
  technically clean, stable, observation-valid chains. The x0.75 direction
  was better than x1.25 for all four representatives and improved the original
  anchor objective in three, so it met the declared class-promotion rule.
- The remaining 36 N2 systems were probed before dynamics. System `2898` has
  two outlet components and was stopped without a forced merge. Downstream
  fields were complete for the other 35; the existing outlet gates passed 25
  and failed systems `512`, `1499`, `2473`, `3082`, `3495`, `3808`, `3904`,
  `3910`, `3914`, and `4637`. No gate was overridden.
- Spatial-downstream-v2 geometry was built for the 25 outlet-pass systems.
  All 25 preserve the Core-191 initial state and pass the static and 0.1/5-year
  zero-MB preflight. One new chain per system then retained its best stable
  RGI-reachable anchor `delta_t`, `melt_f`, `prcp_fac`, and `temp_melt` while
  changing only Stage-2 `glen_a` to x0.75. The system-890 preflight passed and
  released the remaining 24 chains with at most 12 workers.
- All 25 promoted chains are technically clean, adaptive-spin-up stable, and
  observation-domain valid. Nineteen improve the formal separation objective
  relative to their saved RGI anchor; the median objective change is
  `-0.15840`. Eight systems newly enter the legacy 10% total-length-only route,
  but no chain passes the formal dual gate. Close this N2 `glen_a x0.75`
  direction without glacier-wise micro-search or another population promotion.
- The updated formal inventory contains 2550 deduplicated chains and remains
  at exactly eleven formal systems: `991`, `1077`, `1546`, `1616`, `2266`,
  `2337`, `2815`, `2902`, `2928`, `2939`, and `4319`. Routes are 91
  `no_formal_pass`, 56 `total_length_only`, 28 `small_separation_review`,
  eleven `formal_dual_pass`, two `blocked_dynamic_review`, two
  `no_stable_valid_chain`, and one manual provisional candidate outside
  controller sources. The frozen registry remains unchanged at eleven systems.
- Evidence is in
  `validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/diagnostics/no_formal_pass_router_v1/`,
  `lia_downstream_geometry_probe_no_formal_N2_reps_v1/`,
  `lia_downstream_geometry_probe_no_formal_N2_promotion_v1/`,
  `lia_elevation_band_spatial_downstream_no_formal_N2_promotion_v1/`,
  `lia_191_no_formal_N2_glen_spatial_pilot_v1/`,
  `lia_191_no_formal_N2_glen_spatial_promotion_v1/`,
  `lia_191_controller_v1/diagnostics/no_formal_N2_glen_outcome_v1/`, and
  `lia_191_controller_v1/diagnostics/dual_gate_inventory_after_no_formal_N2_v1/`.
"""


def main() -> None:
    path = project_root() / "LIA_PROJECT_FACTS.md"
    text = path.read_text(encoding="utf-8-sig")
    if "## 38. No-formal-pass N2 glen-a promotion" in text:
        print("section already present")
        return
    if HEADER_OLD not in text:
        raise RuntimeError("expected facts header not found")
    text = text.replace(HEADER_OLD, HEADER_NEW, 1).rstrip() + SECTION + "\n"
    path.write_text(text, encoding="utf-8")
    print("updated", path)


if __name__ == "__main__":
    main()
