"""Record batch-1 preflight results and the bounded spatial reroute."""

from __future__ import annotations

from .build_lia_191_iterative_loop import project_root


SECTION = r"""

## 25. Formal delta-T bisection batch 1 preflights (2026-08-15)

- The regular Core-191 preflight for system `2815` at
  `delta_t=+0.359375 K` is numerically clean and converged at spin-up year 500
  with `0.1173%` preceding-200-year volume change, but it first crosses the
  original LIA observation-domain boundary in forward year `1830`. The first
  overflow node is index `177` and the maximum saved overflow thickness is
  `32.30 m`. The regular four-system block is therefore stopped and its other
  eleven chains are not released.
- The independent spatial-downstream-v2 preflight for system `2928` at
  `delta_t=+0.421875 K` is technically clean, converged at spin-up year 500
  with `1.233%` volume change, and has no observation-domain overflow. Its two
  remaining declared delta-T points were released. The colder
  `delta_t=+0.390625 K` chain passes the full dual gate: LIA/RGI separation
  errors `0.119/0.179`, total-length errors `2.727%/5.337%`, stability
  `1.220%`, and no overflow. It is a curve candidate pending review, not yet a
  frozen curve.
- A geometry-only DEM downstream probe was then run for systems `1546`,
  `2806`, `2815`, and `2902`. Downstream fields are finite and complete for all
  four. Existing outlet gates pass only for `2815` and `2902`; they fail for
  `1546` and `2806`, so those two remain stopped and no downstream-domain
  override is allowed.
- The next bounded action is to build spatial-downstream-v2 geometry only for
  `2815` and `2902`, prove unchanged initial Core-191 geometry plus static and
  zero-MB gates, and then rerun only their already declared fine delta-T
  points. No physical parameter or forcing choice changes.
- Evidence is stored under
  `validation_data/results/paleo_surface_validation_v0/lia_191_formal_delta_bisection_batch1_v1/`
  and
  `validation_data/results/paleo_surface_validation_v0/lia_downstream_geometry_probe_formal_delta_bisection_batch1_v1/`.
"""


def main() -> None:
    path = project_root() / "LIA_PROJECT_FACTS.md"
    text = path.read_text(encoding="utf-8")
    if "## 25. Formal delta-T bisection batch 1 preflights" not in text:
        text = text.rstrip() + SECTION + "\n"
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Last updated:"):
            lines[index] = "Last updated: 2026-08-15 (formal delta-T bisection batch 1 preflights completed)"
            break
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
