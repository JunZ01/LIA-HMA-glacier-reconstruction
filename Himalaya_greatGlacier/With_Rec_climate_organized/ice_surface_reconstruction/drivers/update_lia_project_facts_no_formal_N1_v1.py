"""Append the verified N1 glen interpolation result to project facts."""

from __future__ import annotations

from .build_lia_191_iterative_loop import project_root


OLD = "Last updated: 2026-08-15 (N2 glen-a promotion closed; eleven curves remain frozen)"
NEW = "Last updated: 2026-08-15 (N1 glen-a interpolation closed; eleven curves remain frozen)"
SECTION = r"""

## 40. No-formal-pass N1 glen-a interpolation (2026-08-15)

- The N2 promotion created four new N1 systems (`5146`, `5770`, `2005`, and
  `2827`) whose LIA and RGI total-length endpoints were reached by the same
  saved `delta_t` and Stage-2 physics but different `glen_a` multipliers:
  x0.75 and x1.0. This supplied a direct bounded parameter bracket rather than
  an undirected search.
- The interpolation tested Stage-2 `glen_a x0.8125/x0.875/x0.9375` for every
  system, while freezing each system's `delta_t`, Stage-2 `melt_f/prcp_fac`,
  `temp_melt=-1 degC`, B-window forcing, precipitation anomalies off, spatial-
  downstream-v2 geometry, and formal scoring. The system-5146/x0.875
  preflight passed before the remaining eleven chains were released.
- All 12 chains are technically clean, adaptive-spin-up stable, and
  observation-domain valid. No chain passes the formal dual gate. The best
  formal objectives for systems `5146/5770/2005/2827` are respectively
  `0.87517/0.79158/0.62644/0.62145`; none contains a joint 20% separation
  endpoint solution. Close this bracket without glacier-wise micro-bisection.
- The interpolation gives all four systems a legacy 10% total-length-only
  chain, so the updated formal inventory contains 2562 deduplicated chains and
  routes 87 `no_formal_pass`, 60 `total_length_only`, 28
  `small_separation_review`, eleven `formal_dual_pass`, two
  `blocked_dynamic_review`, two `no_stable_valid_chain`, and one manual
  provisional candidate outside controller sources. The formal system list and
  frozen registry remain unchanged at eleven systems.
- Evidence is in
  `validation_data/results/paleo_surface_validation_v0/lia_191_no_formal_N1_glen_interpolation_v1/`,
  `lia_191_controller_v1/diagnostics/no_formal_N1_glen_outcome_v1/`, and
  `lia_191_controller_v1/diagnostics/dual_gate_inventory_after_no_formal_N1_v1/`.
"""


def main() -> None:
    path = project_root() / "LIA_PROJECT_FACTS.md"
    text = path.read_text(encoding="utf-8-sig")
    if "## 40. No-formal-pass N1 glen-a interpolation" in text:
        print("section already present")
        return
    if OLD not in text:
        raise RuntimeError("expected facts header not found")
    path.write_text(text.replace(OLD, NEW, 1).rstrip() + SECTION + "\n", encoding="utf-8")
    print("updated", path)


if __name__ == "__main__":
    main()
