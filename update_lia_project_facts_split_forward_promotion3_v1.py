from pathlib import Path


path = Path(__file__).resolve().parent / "LIA_PROJECT_FACTS.md"
text = path.read_text(encoding="utf-8")
heading = "## 55. Split forward-delta promotion batch 3 closure (2026-08-15)"
if heading not in text:
    section = r'''

## 55. Split forward-delta promotion batch 3 closure (2026-08-15)

- Promotion batch 3 tested the next five previously untested, stable,
  observation-valid both-short systems: `2226`, `2328`, `2378`, `660`, and
  spatial-downstream-v2 system `4565`. The first round used each source
  forward offset and `-0.010/-0.020 K` colder offsets. All 15 chains were
  technically clean; two `660` directed branches overflowed the observation
  domain. Three systems improved and none passed in the first round.
- A single response-directed nine-chain refinement continued only `2226`,
  `2328`, and `4565`. All nine chains were technically clean, but all six
  colder `2226/2328` branches overflowed the observation domain and are
  rejected. System `2378` was closed because its first-round lengths did not
  respond to `-0.020 K`; system `660` was closed at overflow.
- System `4565` supplied an explicit endpoint bracket between forward offsets
  `+0.470` and `+0.480 K`, so one final three-point overlap check tested
  `+0.4725/+0.4750/+0.4775 K`. All three chains were technically clean and
  observation-valid. The `+0.4725 K` branch passes every formal gate: its
  source spin-up offset is `+0.500 K`, forward peak/year-2000 lengths are
  `11.000/10.100 km`, separation errors are `0.03974/0.12920`, and
  total-length errors are `0.3584%/1.2806%`.
- System `4565` was automatically frozen. The frozen registry now contains
  exactly 20 systems and all recorded file hashes and byte counts reproduce
  with zero mismatch. Across batch 3 and its bounded follow-ups, 27 new
  forward-only chains were run with zero technical failure; no spin-up state,
  physical parameter, precipitation setting, geometry, or scoring rule was
  changed.
- The five batch-3 branches are closed. Continuing the same confirmed family
  may proceed only to the next previously untested both-short systems as a new
  bounded batch with mandatory source-forward reproduction, the existing
  observation-domain gate, one response-directed refinement at most, and
  automatic freezing only under the formal dual gate.
- Evidence is in
  `validation_data/results/paleo_surface_validation_v0/lia_191_split_forward_delta_promotion3_v1/`,
  `lia_191_split_forward_delta_promotion3_refine_v1/`,
  `lia_191_split_forward_delta_promotion3_finish_v1/`, and
  `lia_191_frozen_curve_registry_v1/system_4565/`.
'''
    text = text.rstrip() + section + "\n"
text = text.replace(
    "Last updated: 2026-08-15 (split forward-delta promotion batch 2 completed)",
    "Last updated: 2026-08-15 (split forward-delta promotion batch 3 completed)",
)
path.write_text(text, encoding="utf-8")
