"""Record the user-approved freeze of systems 2815, 2902, and 2928."""

from __future__ import annotations

from .build_lia_191_iterative_loop import project_root


SECTION = r"""

## 27. Systems 2815, 2902, and 2928 frozen curves (2026-08-15)

- After reviewing the separate complete spin-up and 1470-2000 length plots,
  the user replied `ok` and approved continuing. Systems `2815`, `2902`, and
  `2928` are therefore frozen as provisional historical curves. No OGGM model
  was advanced during freezing and these curves are not promoted to final
  parameter estimates.
- The selected tasks are
  `2815-B_formal_delta_bisection_spatial_rerun_v1-11230d542cf5171c`,
  `2902-B_formal_delta_bisection_spatial_rerun_v1-ec24fcef6bc9d348`, and
  `2928-B_formal_delta_bisection_2928_v1-87fa80a56981cec1`, at delta-T values
  `+0.34375`, `-0.28125`, and `+0.390625 K`, respectively. All retain
  Stage-2 physical parameters, `temp_melt=-1 degC`, precipitation anomalies
  off, the B-window climate, and spatial-downstream-v2 geometry.
- Their LIA/RGI separation errors are `0.030/0.080`, `0.065/0.086`, and
  `0.119/0.179`; their spin-up stability fractions are `0.120%`, `0.382%`,
  and `1.220%`. Each passes the technical, adaptive-spin-up, observation-domain,
  10% total-length, and 20% separation gates.
- The versioned frozen registry now contains exactly six systems: `991`,
  `1077`, `1616`, `2815`, `2902`, and `2928`. Every package contains the
  complete timeseries, compact length/area/volume trajectory, spin-up and
  year-2000 states, source task and summary, displayed plot, candidate record,
  and hashes. All saved hashes reproduce without mismatch.
- Registry:
  `validation_data/results/paleo_surface_validation_v0/lia_191_frozen_curve_registry_v1/`.
  Reusable freeze code:
  `Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/freeze_lia_191_candidates_2815_2902_2928_v1.py`.
"""


def main() -> None:
    path = project_root() / "LIA_PROJECT_FACTS.md"
    text = path.read_text(encoding="utf-8")
    if "## 27. Systems 2815, 2902, and 2928 frozen curves" not in text:
        text = text.rstrip() + SECTION + "\n"
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Last updated:"):
            lines[index] = "Last updated: 2026-08-15 (six formal dual-gate curves frozen)"
            break
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
