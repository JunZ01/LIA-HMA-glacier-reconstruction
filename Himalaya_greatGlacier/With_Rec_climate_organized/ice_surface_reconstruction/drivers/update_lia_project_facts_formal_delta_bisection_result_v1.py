"""Record the completed formal delta-T bisection batch-1 result."""

from __future__ import annotations

from .build_lia_191_iterative_loop import project_root


SECTION = r"""

## 26. Formal delta-T bisection batch 1 result (2026-08-15)

- Spatial-downstream-v2 geometry was built only for systems `2815` and `2902`
  after both passed the existing outlet gate. Their original Core-191 LIA
  surface, bed, thickness, width, area, and volume are unchanged to the
  declared numerical tolerances. Both pass the common static and 0.1/5-year
  zero-MB gates. The one-chain `2815/delta_t=+0.359375 K` dynamics preflight
  then converged at year 500 with `0.1173%` preceding-200-year volume change
  and no observation-domain overflow, releasing the remaining five chains.
- All six spatial chains for systems `2815` and `2902` are technically clean,
  stable, observation-valid, and pass both the 10% total-length and 20%
  separation-normalized endpoint gates. System `2815` passes at all three
  tested delta-T values `+0.34375/+0.359375/+0.3671875 K`; system `2902`
  passes at `-0.296875/-0.28125/-0.265625 K`.
- Best system `2815` chain by formal separation objective is
  `2815-B_formal_delta_bisection_spatial_rerun_v1-11230d542cf5171c` at
  `delta_t=+0.34375 K`. Its forward peak is `8.850 km` versus LIA
  `8.809 km`, year-2000 length is `7.300 km` versus RGI `7.411 km`,
  separation errors are `0.030/0.080`, total-length errors are
  `0.468%/1.502%`, and stability is `0.120%`.
- Best system `2902` chain is
  `2902-B_formal_delta_bisection_spatial_rerun_v1-ec24fcef6bc9d348` at
  `delta_t=-0.28125 K`. Its forward peak is `9.600 km` versus LIA
  `9.484 km`, year-2000 length is `7.850 km` versus RGI `7.696 km`,
  separation errors are `0.065/0.086`, total-length errors are
  `1.226%/1.996%`, and stability is `0.382%`.
- System `2928` has one formal pass at `delta_t=+0.390625 K`, task
  `2928-B_formal_delta_bisection_2928_v1-87fa80a56981cec1`. Its forward peak
  is `14.650 km` versus LIA `14.261 km`, year-2000 length is `10.400 km`
  versus RGI `10.986 km`, separation errors are `0.119/0.179`, total-length
  errors are `2.727%/5.337%`, and stability is `1.220%`.
- Across the executed work there are nine technically clean, stable-valid
  spatial chains and one deliberately retained invalid Core-191 preflight
  showing the `2815` observation-domain overflow. Seven chains from three
  systems pass the formal dual gate. Systems `1546` and `2806` were not run
  because their existing outlet geometry gates failed; they remain explicit
  geometry-review items rather than silent failures.
- Systems `2815`, `2902`, and `2928` are curve candidates pending user review;
  they are not yet copied into the frozen registry. Formal tables and separate
  plots are in
  `validation_data/results/paleo_surface_validation_v0/lia_191_formal_delta_bisection_batch1_v1/`.
  Reusable code includes
  `lia_191_spatial_glen_tradeoff_pilot.py`,
  `build_lia_191_formal_delta_bisection_batch1_v1.py`,
  `build_lia_191_formal_delta_bisection_spatial_rerun_v1.py`, and
  `summarize_plot_lia_191_formal_delta_bisection_batch1_v2.py`.
"""


def main() -> None:
    path = project_root() / "LIA_PROJECT_FACTS.md"
    text = path.read_text(encoding="utf-8")
    if "## 26. Formal delta-T bisection batch 1 result" not in text:
        text = text.rstrip() + SECTION + "\n"
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Last updated:"):
            lines[index] = "Last updated: 2026-08-15 (formal delta-T bisection batch 1 completed)"
            break
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
