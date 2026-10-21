"""Append the verified read-only N4 diagnosis to project facts."""

from __future__ import annotations

from .build_lia_191_iterative_loop import project_root


OLD = "Last updated: 2026-08-15 (N3 melt-delta pilot closed; eleven curves remain frozen)"
NEW = "Last updated: 2026-08-15 (N4 branch diagnosis closed; eleven curves remain frozen)"
SECTION = r"""

## 42. No-formal-pass N4 branch-position diagnosis (2026-08-15)

- A read-only diagnosis used all stable, observation-valid saved chains from
  the 2570-chain inventory for the 35 N4 systems that reach neither 10%
  total-length endpoint. No OGGM model was advanced and no saved result was
  changed.
- Thirty-four systems have both the forward LIA peak and year-2000 endpoint
  more than 10% short in every stable saved chain. Their median best formal
  separation objective is `4.76655`. They do not supply a bounded endpoint
  interpolation bracket, so another automatic local physical-parameter batch
  is not justified.
- System `2677` is the only other response: its LIA peak remains 14.4-49.3%
  short in all three stable chains, while the RGI endpoint straddles the 10%
  band without a near stable chain. Its best formal objective is `0.81390` and
  amplitude ratio `0.18610`. This single case does not justify a population
  promotion or glacier-wise micro-search.
- Decision: close N4 at the read-only routing boundary. The next automatic
  action should inspect the updated `total_length_only` population, where new
  class batches may have produced bounded formal-separation near-misses.
- Evidence is in
  `validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/diagnostics/no_formal_N4_branch_diagnosis_v1/`.
"""


def main() -> None:
    path = project_root() / "LIA_PROJECT_FACTS.md"
    text = path.read_text(encoding="utf-8-sig")
    if "## 42. No-formal-pass N4 branch-position diagnosis" in text:
        print("section already present")
        return
    if OLD not in text:
        raise RuntimeError("expected facts header not found")
    path.write_text(text.replace(OLD, NEW, 1).rstrip() + SECTION + "\n", encoding="utf-8")
    print("updated", path)


if __name__ == "__main__":
    main()
