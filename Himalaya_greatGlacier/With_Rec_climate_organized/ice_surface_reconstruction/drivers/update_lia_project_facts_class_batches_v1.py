"""Record the verified class-batch outcome in the canonical facts file."""

from __future__ import annotations

from pathlib import Path

from .build_lia_191_iterative_loop import project_root


FACTS = project_root() / "LIA_PROJECT_FACTS.md"
OLD_HEADER = "Last updated: 2026-08-15 (system 2266 Layer-3 pass; ten curves frozen)"
NEW_HEADER = (
    "Last updated: 2026-08-15 (class-based batches completed; "
    "system 1546 pass; eleven curves frozen)"
)
SECTION_MARKER = "## 36. Class-based continuous parameter batches and system 1546 freeze"
SECTION = r"""


## 36. Class-based continuous parameter batches and system 1546 freeze (2026-08-15)

- The user explicitly replaced glacier-by-glacier approval with class-based
  continuous execution. Batches may run and formally passing systems may be
  frozen automatically; the user reviews consolidated class outcomes after a
  batch boundary. This does not authorize changing the formal score, forcing,
  geometry gates, or opening undeclared parameter families.
- The 49 `total_length_only` systems in the post-2266 inventory were routed by
  the signed formal separation errors of their best stable,
  observation-valid total-length-pass chain. Counts were 15 C1 peak-short / RGI-long,
  13 C2 peak-short / RGI-near, 16 C3 both-short, four C4 peak-near / RGI-short,
  and one C5 peak-near / RGI-long. Forty-three systems were eligible for a new
  block; six were already closed by completed Layer-3 evidence. Eighteen
  representative systems were selected before any class promotion.
- Eleven named experiments added exactly 45 real OGGM chains. Thirty-four are
  technically clean, adaptive-spin-up stable, and observation-domain valid;
  three have technical failures. The class mechanisms and decisions are:
  C1 tested `glen_a` on representatives and promoted the improving x1.25
  direction, but produced no formal pass and is closed; C2 tested `glen_a`,
  promoted x0.75, and allowed one delta-T recenter for the near-gate cases;
  C3 tested `prcp_fac`, exceeded the 30% technical/domain failure stop rule,
  produced no formal pass, and is closed without promotion; C4 and C5 tested
  bounded `glen_a` directions, produced no formal pass, and are closed without
  promotion. Existing outlet-gate failures were never overridden.
- System `1546` is the single new formal pass. Task
  `1546-B_cls_C2_dt_rec_v1-614218f49edc6746` uses
  `delta_t=+0.09375 K`, Stage-2 `melt_f x1.0`, and Stage-2 `glen_a x0.75`.
  Its forward peak/year-2000 lengths are `11.550/10.000 km`; its LIA/RGI
  separation errors are `0.101384/0.017629`, total-length errors are
  `1.5209%/0.3111%`, and preceding-200-year spin-up volume change is
  `0.6460%`. The chain is technically clean, stable, and observation-domain
  valid, so it is automatically frozen under the approved continuous-batch
  rule.
- The updated formal inventory contains 2503 deduplicated chains and exactly
  eleven formal systems: `991`, `1077`, `1546`, `1616`, `2266`, `2337`,
  `2815`, `2902`, `2928`, `2939`, and `4319`. Its routes are 99
  `no_formal_pass`, 48 `total_length_only`, 28 `small_separation_review`,
  eleven `formal_dual_pass`, two `blocked_dynamic_review`, two
  `no_stable_valid_chain`, and one manual provisional candidate outside
  controller sources.
- The frozen registry contains the same eleven systems. Ninety-eight recorded
  package files reproduce both their stored SHA-256 hashes and byte counts
  with zero mismatch.
- Evidence is in
  `validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/diagnostics/total_length_only_batch_router_v1/`,
  `validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/diagnostics/class_batch_outcome_v1/`,
  `validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/diagnostics/dual_gate_inventory_after_class_batches_v1/`,
  `validation_data/results/paleo_surface_validation_v0/lia_191_class_C2_delta_recenter_v1/`,
  and `validation_data/results/paleo_surface_validation_v0/lia_191_frozen_curve_registry_v1/system_1546/`.
"""


def main() -> None:
    text = FACTS.read_text(encoding="utf-8")
    if OLD_HEADER in text:
        text = text.replace(OLD_HEADER, NEW_HEADER, 1)
    elif NEW_HEADER not in text:
        raise RuntimeError("Unexpected facts header; refusing to rewrite it")

    if SECTION_MARKER not in text:
        text = text.rstrip() + SECTION + "\n"

    FACTS.write_text(text, encoding="utf-8")
    print(FACTS)
    print(NEW_HEADER)
    print("section_present=", SECTION_MARKER in text)


if __name__ == "__main__":
    main()
