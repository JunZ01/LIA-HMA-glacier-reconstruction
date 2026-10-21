"""Append the verified frozen-curve modern validation result to project facts."""

from __future__ import annotations

from pathlib import Path

from .build_lia_191_iterative_loop import project_root


OLD_HEADER = "Last updated: 2026-08-15 (continuous class loop at declared-family boundary)"
NEW_HEADER = "Last updated: 2026-08-15 (frozen-curve 2000-2020 validation completed)"
SECTION_HEADER = "## 46. Eleven frozen-curve 2000-2020 validation (2026-08-15)"
SECTION = """
## 46. Eleven frozen-curve 2000-2020 validation (2026-08-15)

- All eleven formal frozen systems were continued from their saved year-2000
  states to 2020 with historical GSWP3-W5E5 2000-2019 climate. Each run kept
  its frozen candidate `delta_t`, `melt_f`, `prcp_fac`, `glen_a`, and
  `temp_melt`. This was a diagnostic continuation; it did not change the
  historical formal gate or refit any parameter.
- All eleven continuations completed without a process or technical failure.
  Every modeled mean 2000-2020 surface mass balance is more negative than the
  Stage-2 Hugonnet system reference. Residuals span `-699.13` to
  `-208.23 kg m-2 yr-1`; zero systems are within one reference sigma, three
  (`1077`, `1546`, and `2337`) are within two sigma, and the median absolute
  mismatch is `2.774 sigma`.
- All eleven curves retreat after 2000. Length changes span `-3.00` to
  `-0.40 km`, and volume changes span `-30.97%` to `-14.82%` by 2020. The
  largest mass-balance mismatch is system `991` at `5.567 sigma`.
- This systematic modern-negative result is a class-level forcing/parameter-
  consistency diagnostic. It does not independently reject the eleven
  historical curves because Hugonnet already informed Stage 2 and direct
  comparability between the Core-191 pseudo-flowline systems and the Stage-2
  modern observation aggregation remains limited. Do not retune `melt_f` from
  this diagnostic alone and do not unfreeze a historical curve silently.
- The next bounded historical-search family, if execution continues beyond
  the closed C1-C5 and N1-N4 mechanisms, should separate initial-state control
  from forward-forcing control on a small representative class before any
  population promotion. Reuse saved converged spin-up states where possible;
  do not launch another undirected five-parameter search.
- Evidence and the reusable batch driver are in
  `validation_data/results/paleo_surface_validation_v0/lia_191_frozen_modern_validation_v1/`
  and
  `Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/run_lia_191_frozen_modern_validation_v1.py`.
"""


def main() -> None:
    path = project_root() / "LIA_PROJECT_FACTS.md"
    text = path.read_text(encoding="utf-8")
    if SECTION_HEADER in text:
        print("facts-section-already-present")
        return
    if text.count(OLD_HEADER) != 1:
        raise RuntimeError("facts-header-marker-mismatch")
    text = text.replace(OLD_HEADER, NEW_HEADER, 1)
    path.write_text(text.rstrip() + "\n\n" + SECTION.strip() + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
