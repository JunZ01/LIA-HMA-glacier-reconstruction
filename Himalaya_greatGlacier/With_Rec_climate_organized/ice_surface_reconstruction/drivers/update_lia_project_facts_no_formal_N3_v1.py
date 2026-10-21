"""Append the verified N3 representative pilot result to project facts."""

from __future__ import annotations

from .build_lia_191_iterative_loop import project_root


OLD = "Last updated: 2026-08-15 (N1 glen-a interpolation closed; eleven curves remain frozen)"
NEW = "Last updated: 2026-08-15 (N3 melt-delta pilot closed; eleven curves remain frozen)"
SECTION = r"""

## 41. No-formal-pass N3 melt-f plus delta-T pilot (2026-08-15)

- The N3 class contains systems whose saved stable chains reach the LIA
  total-length endpoint but not the RGI endpoint. The declared mechanism test
  raised Stage-2 `melt_f` to x1.10/x1.20 and paired each value with the LIA
  anchor `delta_t` and one `-0.125 K` colder recenter. Stage-2 `glen_a/prcp_fac`,
  `temp_melt=-1 degC`, B-window forcing, precipitation anomalies off, adaptive
  spin-up, and formal scoring remained frozen.
- Router representative `1743` and first replacement `4588` have complete DEM
  downstream fields but fail the existing outlet gate and were stopped without
  an override. A single read-only probe of the remaining N3 systems found six
  outlet-pass systems. System `2906`, the passing system with the smallest
  current formal objective, replaced them alongside outlet-pass system `2793`.
  Spatial-downstream-v2 builds for `2793/2906` preserve the Core-191 initial
  states and pass all static and 0.1/5-year zero-MB gates.
- All eight representative chains are technically clean, adaptive-spin-up
  stable, and observation-domain valid. No chain passes the formal dual gate.
  Relative to the previous best stable objective, system `2793` worsens by
  `+0.88794`, while system `2906` improves only `-0.01263`. The declared
  two-representative promotion rule is not met; close this N3 `melt_f +
  delta-T` family without population promotion.
- The updated formal inventory contains 2570 deduplicated chains. It remains
  at eleven formal systems and routes 87 `no_formal_pass`, 60
  `total_length_only`, 28 `small_separation_review`, eleven
  `formal_dual_pass`, two `blocked_dynamic_review`, two
  `no_stable_valid_chain`, and one manual provisional candidate outside
  controller sources. The frozen registry remains unchanged at eleven systems.
- Evidence is in
  `validation_data/results/paleo_surface_validation_v0/lia_downstream_geometry_probe_no_formal_N3_reps_v1/`,
  `lia_downstream_geometry_probe_no_formal_N3_replacement_v1/`,
  `lia_downstream_geometry_probe_no_formal_N3_remaining_v1/`,
  `lia_downstream_geometry_probe_no_formal_N3_selected_v1/`,
  `lia_elevation_band_spatial_downstream_no_formal_N3_reps_v1/`,
  `lia_191_no_formal_N3_melt_delta_spatial_pilot_v1/`,
  `lia_191_controller_v1/diagnostics/no_formal_N3_outcome_v1/`, and
  `lia_191_controller_v1/diagnostics/dual_gate_inventory_after_no_formal_N3_v1/`.
"""


def main() -> None:
    path = project_root() / "LIA_PROJECT_FACTS.md"
    text = path.read_text(encoding="utf-8-sig")
    if "## 41. No-formal-pass N3 melt-f plus delta-T pilot" in text:
        print("section already present")
        return
    if OLD not in text:
        raise RuntimeError("expected facts header not found")
    path.write_text(text.replace(OLD, NEW, 1).rstrip() + SECTION + "\n", encoding="utf-8")
    print("updated", path)


if __name__ == "__main__":
    main()
