"""Append the consolidated continuous class-loop boundary to project facts."""

from __future__ import annotations

from .build_lia_191_iterative_loop import project_root


OLD = "Last updated: 2026-08-15 (N4 branch diagnosis closed; eleven curves remain frozen)"
NEW = "Last updated: 2026-08-15 (continuous class loop at declared-family boundary)"
SECTION = r"""

## 43. Continuous class-loop consolidated boundary (2026-08-15)

- The continuous loop from class round 2 through the N2, N1, and N3 blocks
  executed 67 new real OGGM chains: 14 in class round 2, 33 in the N2
  representative/promotion block, 12 in the N1 interpolation, and eight in the
  N3 representative pilot. There were zero process/technical failures and zero
  new formal dual-gate passes.
- The final formal inventory contains 2570 deduplicated chains. Exactly eleven
  systems remain formal and frozen: `991`, `1077`, `1546`, `1616`, `2266`,
  `2337`, `2815`, `2902`, `2928`, `2939`, and `4319`. Routes are 87
  `no_formal_pass`, 60 `total_length_only`, 28 `small_separation_review`,
  eleven `formal_dual_pass`, two `blocked_dynamic_review`, two
  `no_stable_valid_chain`, and one manual provisional candidate outside
  controller sources.
- A fresh read-only registry validation covers all eleven frozen systems and
  98 recorded files. Membership, file existence, byte counts, and SHA-256
  hashes all pass, with zero mismatch.
- The updated 60-system `total_length_only` router contains 20 C1 peak-short /
  RGI-long, 14 C2 peak-short / RGI-near, 19 C3 both-short, four C4 peak-near /
  RGI-short, and three C5 peak-near / RGI-long systems. It reports 20 nominal
  per-system eligible flags, but their proposed C1/C3/C4 `glen_a` or
  `prcp_fac` families were already closed at class level in rounds 1-2. Do not
  reopen the same mechanisms simply because a later chain moved a new system
  into the route.
- Continuous automatic OGGM execution therefore stops at the declared-family
  boundary. The next action requires either a new scientific family outside
  the closed C1-C5 and N1-N4 mechanisms, or consolidated review/validation of
  the eleven frozen curves. This is a scientific routing boundary, not a
  request for glacier-by-glacier approval.
- Evidence is in
  `validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/diagnostics/continuous_class_loop_outcome_v1/`,
  `lia_191_controller_v1/diagnostics/total_length_only_batch_router_after_no_formal_v1/`,
  `lia_191_controller_v1/diagnostics/dual_gate_inventory_after_no_formal_N3_v1/`,
  and `lia_191_frozen_curve_registry_v1/registry_validation_v1/`.
"""


def main() -> None:
    path = project_root() / "LIA_PROJECT_FACTS.md"
    text = path.read_text(encoding="utf-8-sig")
    if "## 43. Continuous class-loop consolidated boundary" in text:
        print("section already present")
        return
    if OLD not in text:
        raise RuntimeError("expected facts header not found")
    path.write_text(text.replace(OLD, NEW, 1).rstrip() + SECTION + "\n", encoding="utf-8")
    print("updated", path)


if __name__ == "__main__":
    main()
