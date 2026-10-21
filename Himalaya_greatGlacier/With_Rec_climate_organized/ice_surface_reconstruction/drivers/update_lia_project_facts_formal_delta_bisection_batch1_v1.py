"""Record the declared formal delta-T bisection batch in the project contract."""

from __future__ import annotations

from .build_lia_191_iterative_loop import project_root


SECTION = r"""

## 24. Formal delta-T bisection batch 1 declared (2026-08-15)

- After freezing systems `991`, `1077`, and `1616`, the next bounded block is
  a five-system fine delta-T interpolation for systems `1546`, `2815`, `2902`,
  `2806`, and `2928`. These are stable, observation-valid saved near-misses
  whose endpoint directions can plausibly enter the formal 20% separation
  gate under a small colder delta-T perturbation. This selection replaces the
  discarded system `584`, whose saved response does not provide an overlapping
  delta-T-only interval for both endpoints.
- The block changes delta-T only. Stage-2 `melt_f`, `prcp_fac`, and `glen_a`,
  `temp_melt=-1 degC`, precipitation anomalies off, the B-window `1441-1470`,
  forward period `1470-2000`, W5E5 climatology, and `ensemble_rec_ref`
  temperature anomalies are frozen.
- Planned delta-T values are: system `1546` at
  `0.09375/0.1015625/0.109375 K`; system `2815` at
  `0.34375/0.359375/0.3671875 K`; system `2902` at
  `-0.296875/-0.28125/-0.265625 K`; system `2806` at
  `0.21875/0.2265625/0.234375 K`; and system `2928` at
  `0.390625/0.40625/0.421875 K`. There are 15 planned chains.
- System `2928` must retain the verified
  `core-lia-plus-spatial-dem-downstream-v2` geometry and spatial observation
  runner. The other four systems use the corrected Core-191 elevation-band
  geometry. The two geometry groups have separate one-chain preflights before
  batch release: `2815/delta_t=0.359375 K` and
  `2928/delta_t=0.421875 K`.
- A curve may be frozen only if it is technically clean, converges within the
  2% preceding-200-year volume-stability gate, has a valid observation-domain
  length, passes both 10% total-length endpoints, and passes both 20%
  separation-normalized endpoints. No other physical parameter family opens
  in this block.
- Design, selection diagnostics, and configurations are versioned under
  `validation_data/results/paleo_surface_validation_v0/lia_191_formal_delta_bisection_batch1_v1/`
  and
  `Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/lia_191_formal_delta_bisection_batch1_v1/`.
"""


def main() -> None:
    facts_path = project_root() / "LIA_PROJECT_FACTS.md"
    text = facts_path.read_text(encoding="utf-8")
    if "## 24. Formal delta-T bisection batch 1 declared" not in text:
        text = text.rstrip() + SECTION + "\n"
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Last updated:"):
            lines[index] = "Last updated: 2026-08-15 (formal delta-T bisection batch 1 declared)"
            break
    facts_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(facts_path)


if __name__ == "__main__":
    main()
