"""Record the verified second-round class batches in project facts."""

from __future__ import annotations

from .build_lia_191_iterative_loop import project_root


FACTS = project_root() / "LIA_PROJECT_FACTS.md"
HEADER = (
    "Last updated: 2026-08-15 (class round 2 closed; "
    "eleven curves remain frozen)"
)
MARKER = "## 37. Class-based parameter batches round 2"
TEXT = r"""


## 37. Class-based parameter batches round 2 (2026-08-15)

- After the user directed the continuous class loop to continue, three bounded
  class questions were opened. They added 14 real OGGM chains, with zero
  process/technical failures. Ten chains overflow the declared observation
  domain and only four are both adaptive-spin-up stable and observation-valid.
  No chain passes the formal dual gate.
- C1 first tested Stage-2 `melt_f x0.90` at system `843` without a delta-T
  recenter. The preflight was nonconverged with `23.32%` preceding-200-year
  volume change and observation-domain overflow, so the remaining V1 manifest
  was not released. A documented V2 retained only the already declared warmer
  `+0.125/+0.250 K` recenter offsets for systems `843`, `5231`, and `5460`.
  Of six V2 chains, only `843/+0.250 K` is stable and observation-valid; it has
  LIA/RGI separation errors `0.00835/0.38232`. The other five overflow, three
  are also nonconverged, and the 83.3% invalid rate exceeds the class stop
  threshold. Close this C1 `melt_f x0.90 + delta-T` direction without
  promotion or micro-search.
- The closed C2 delta-T recenter was not reopened. A separate `temp_melt`
  direction pilot used the three closest unresolved C2 anchors, systems
  `1524`, `2569`, and `2622`, at `temp_melt=-1.25/-0.75 degC`. All three
  colder-threshold branches are stable and observation-valid but shorten the
  historical peak strongly; their formal objectives are `0.555`, `1.443`,
  and `1.499`. All three warmer-threshold branches are nonconverged and
  observation-domain overflow. The 50% invalid rate exceeds the stop threshold.
  Close this C2 `temp_melt` family without promotion.
- The C4 spatial pilot asked whether raising `temp_melt` could reduce excessive
  retreat for systems `520` and `2955`. The mandatory system-`2955`,
  `temp_melt=-0.875 degC` preflight converged with `1.155%` stability but
  overflowed the observation domain. The remaining three chains were not
  released. Close this C4 direction at preflight.
- The updated read-only formal inventory contains 2517 deduplicated chains.
  It remains at exactly eleven formal systems: `991`, `1077`, `1546`, `1616`,
  `2266`, `2337`, `2815`, `2902`, `2928`, `2939`, and `4319`. Route counts
  remain 99 `no_formal_pass`, 48 `total_length_only`, 28
  `small_separation_review`, eleven `formal_dual_pass`, two
  `blocked_dynamic_review`, two `no_stable_valid_chain`, and one manual
  provisional candidate outside controller sources. The frozen registry is
  unchanged at eleven systems.
- The repeated long-branch overflow/nonconvergence response is a class-level
  stop result, not a reason to override the outlet or observation-domain gates.
  Do not reopen these three closed directions glacier by glacier. The next
  safe automatic action is a read-only response/reachability router for the 99
  `no_formal_pass` systems; a new OGGM family must follow a declared class
  question from that router.
- Evidence is in
  `validation_data/results/paleo_surface_validation_v0/lia_191_class_C1_melt_delta_spatial_pilot_v1/`,
  `lia_191_class_C1_melt_delta_spatial_pilot_v2/`,
  `lia_191_class_C2_temp_melt_pilot_v1/`,
  `lia_191_class_C4_temp_melt_spatial_pilot_v1/`,
  `lia_191_controller_v1/diagnostics/class_batch_round2_outcome_v1/`, and
  `lia_191_controller_v1/diagnostics/dual_gate_inventory_after_class_round2_v1/`.
"""


def main() -> None:
    text = FACTS.read_text(encoding="utf-8")
    if MARKER not in text:
        text = text.rstrip() + TEXT + "\n"
    lines = text.splitlines()
    for index, line in enumerate(lines[:20]):
        if line.startswith("Last updated:"):
            lines[index] = HEADER
            break
    else:
        raise RuntimeError("Facts header not found")
    FACTS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(FACTS)
    print(HEADER)
    print("section_present=", MARKER in "\n".join(lines))


if __name__ == "__main__":
    main()
