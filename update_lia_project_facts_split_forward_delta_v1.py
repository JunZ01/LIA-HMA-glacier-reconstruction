"""Append the verified split-forward-delta pilot result to the project contract."""

from pathlib import Path


path = Path("LIA_PROJECT_FACTS.md")
text = path.read_text(encoding="utf-8")
text = text.replace(
    "Last updated: 2026-08-15 (frozen-curve 2000-2020 validation completed)",
    "Last updated: 2026-08-15 (split forward-delta representative pilot completed)",
    1,
)
text = text.replace(
    "## 48. GPCC calibration-target and gauge-support closure (2026-08-15)",
    "## 49. GPCC calibration-target and gauge-support closure (2026-08-15)",
    1,
)

section = r'''

## 50. Saved-state split forward-delta representative pilot (2026-08-15)

- The first new historical-search family beyond the closed C1-C5 and N1-N4
  mechanisms separated the temperature offset used to create the stable LIA
  state from the constant offset used during the 1470-2000 forward run. It
  reused each source `spinup_state.pkl.gz`; no spin-up, geometry, or physical
  parameter was recomputed. W5E5 1961-1990 climatology,
  `ensemble_rec_ref`, precipitation anomalies off, `melt_f`, `prcp_fac`,
  `glen_a`, and `temp_melt` remained frozen.
- Representatives were systems `1524` (opposite-sign endpoint near-miss),
  `2569/2622` (both endpoints short), and `843` (both endpoints long). The
  original forward delta-T branch was rerun for every representative before
  release. All saved decadal lengths reproduced exactly and maximum relative
  area/volume differences were below floating-point precision, passing the
  declared 50 m and 0.05% reproduction tolerances.
- The coarse and one bounded 0.005 K refinement contain 32 forward-only OGGM
  chains. All 32 completed with zero process or technical failure. Eleven
  colder branches overflow the declared observation domain, giving a 34.375%
  invalid/domain-overflow rate. This exceeds the predeclared 30% class
  promotion ceiling, so the family is not released across the unresolved
  population from this pilot.
- System `2569` formally passes with the source spin-up offset `-0.500 K` and
  forward offset `-0.515 K`. Its forward peak/year-2000 lengths are
  `10.350/8.600 km`; separation errors are `0.02388/0.01373`, total-length
  errors are `0.3906%/0.2686%`, and the reused source spin-up stability is
  `0.9554%`.
- System `2622` formally passes with spin-up offset `0.000 K` and forward
  offset `-0.010 K`. Its forward peak/year-2000 lengths are
  `15.050/12.150 km`; separation errors are `0.01921/0.10856`, total-length
  errors are `0.3386%/2.3234%`, and source spin-up stability is `0.9169%`.
- Systems `1524` and `843` do not pass. Warming their forward offsets reduces
  the year-2000 error but moves the LIA peak out of tolerance faster; retain
  this as endpoint-tradeoff evidence and do not micro-search those branches.
- Systems `2569` and `2622` were automatically frozen under the approved
  continuous-batch rule. The frozen registry now contains exactly 13 systems:
  `991`, `1077`, `1546`, `1616`, `2266`, `2337`, `2569`, `2622`, `2815`,
  `2902`, `2928`, `2939`, and `4319`. All recorded file hashes and byte counts
  reproduce with zero mismatch. These are provisional historical curves, not
  final parameter posteriors.
- The next bounded action is a second representative confirmation inside the
  same-sign, both-short class using only `-0.005/-0.010 K` forward offsets and
  the existing outlet/observation-domain gates. It is not a population
  promotion. Population release requires the new confirmation to remain below
  the invalid-rate ceiling and reproduce formal improvement before expansion.
- Evidence and reusable code are in
  `validation_data/results/paleo_surface_validation_v0/lia_191_split_forward_delta_pilot_v1/`,
  `Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/lia_forward_delta_from_saved_spinup.py`,
  `run_lia_191_split_forward_delta_pilot_v1.py`,
  `run_lia_191_split_forward_delta_pilot_refine_v1.py`, and
  `freeze_lia_191_split_forward_delta_passes_v1.py`.
'''

if "## 50. Saved-state split forward-delta representative pilot" not in text:
    text = text.rstrip() + section + "\n"
path.write_text(text, encoding="utf-8")
