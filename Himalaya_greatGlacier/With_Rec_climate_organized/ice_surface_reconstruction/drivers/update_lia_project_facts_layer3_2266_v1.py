"""Record the verified system-2266 Layer-2/Layer-3 result in project facts."""

from __future__ import annotations

from .build_lia_191_iterative_loop import project_root


MARKER = "## 31. System 2266 Layer-2/Layer-3 completion and frozen curve (2026-08-15)"

TEXT = r"""

## 31. System 2266 Layer-2/Layer-3 completion and frozen curve (2026-08-15)

- After the bounded Stage-2 delta-T pool was exhausted, a read-only endpoint-
  reachability audit found that system `2266` was the only unresolved system
  whose stable saved chains reached the LIA and RGI separation gates on
  different delta-T branches and that had not already received a `melt_f`
  family block. This explicitly justified one Layer-2 block; no population-wide
  physical-parameter sweep was opened.
- The first Layer-2 block tested Stage-2 `melt_f` multipliers `1.05/1.10` with
  six colder/recentered delta-T values. All six chains are technically clean,
  stable, and observation-domain valid, but none passes the formal dual gate.
  Increasing `melt_f` reduced peak-to-2000 amplitude from the approximately
  `1.20 km` Stage-2 anchor to `0.80-0.85 km` at x1.05 and `0.55-0.65 km` at
  x1.10, moving the response in the wrong direction.
- The one permitted Layer-2 refinement therefore tested the opposite
  `melt_f x0.95` direction at delta-T `+0.800/+0.825/+0.850 K`. All three
  chains are clean, stable, and observation-valid, but none passes both formal
  endpoints. The best new point is `x0.95/+0.850 K`, with LIA/RGI separation
  errors `0.1284/0.2131`; it returns to the same discrete year-2000 length
  plateau. The `melt_f + delta-T` family is closed for this named configuration.
- The diagnosed remaining problem was response amplitude, so Layer 3 opened
  exactly one family, `glen_a`, at the best Stage-2 anchor
  (`delta_t=+0.65404781 K`, Stage-2 `melt_f/prcp_fac`, `temp_melt=-1 degC`).
  The two direction points were Stage-2 `glen_a x0.75/x1.25`; both are clean,
  stable, and observation-valid.
- The `glen_a x1.25` chain passes every formal gate without a delta-T recenter.
  Task `2266-B_l3_ga_2266_v1-833096a33742933d` has forward peak/year-2000
  lengths `12.100/11.000 km`, LIA/RGI separation errors `0.1856/0.1844`,
  total-length errors `2.609%/3.016%`, and preceding-200-year spin-up volume
  change `0.6822%`. The `glen_a x0.75` direction misses the RGI separation gate
  and is retained only as response evidence.
- System `2266` is automatically frozen under the approved continuous-batch
  rule. The frozen registry now contains exactly ten systems: `991`, `1077`,
  `1616`, `2266`, `2337`, `2815`, `2902`, `2928`, `2939`, and `4319`; all
  recorded package hashes reproduce.
- The updated formal inventory contains 2458 deduplicated chains and ten formal
  systems. Its routes are 99 `no_formal_pass`, 49 `total_length_only`, 28
  `small_separation_review`, ten `formal_dual_pass`, two
  `blocked_dynamic_review`, two `no_stable_valid_chain`, and one manual
  provisional candidate outside controller sources.
- No additional unresolved system satisfies the same unused Layer-2 entry
  condition. Continuous execution stops at this routing boundary; continuing
  would require a new declared question for a different route class rather
  than applying the 2266 mechanism indiscriminately.
- Evidence is in
  `validation_data/results/paleo_surface_validation_v0/lia_191_layer2_melt_delta_2266_v1/`,
  `lia_191_layer2_melt_delta_2266_refine_v1/`,
  `lia_191_layer2_melt_delta_2266_consolidated_v1/`,
  `lia_191_layer3_glen_2266_v1/`,
  `lia_191_controller_v1/diagnostics/dual_gate_inventory_after_layer3_2266_v1/`,
  and `lia_191_frozen_curve_registry_v1/system_2266/`.
"""


def main() -> None:
    path = project_root() / "LIA_PROJECT_FACTS.md"
    text = path.read_text(encoding="utf-8")
    if MARKER not in text:
        text = text.rstrip() + "\n" + TEXT.lstrip()
    lines = text.splitlines()
    for index, line in enumerate(lines[:20]):
        if line.startswith("Last updated:"):
            lines[index] = (
                "Last updated: 2026-08-15 (system 2266 Layer-3 pass; "
                "ten curves frozen)"
            )
            break
    else:
        raise ValueError("project facts last-updated header not found")
    suffix = "\n" if text.endswith("\n") else ""
    path.write_text("\n".join(lines) + suffix, encoding="utf-8")
    print(lines[index])


if __name__ == "__main__":
    main()
