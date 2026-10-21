"""Append the verified split-forward confirmation and declared promotion batch."""

from pathlib import Path


path = Path("LIA_PROJECT_FACTS.md")
text = path.read_text(encoding="utf-8")
text = text.replace(
    "Last updated: 2026-08-15 (split forward-delta representative pilot completed)",
    "Last updated: 2026-08-15 (split forward-delta confirmation completed; promotion batch 1 declared)",
    1,
)
section = r'''

## 52. Split forward-delta confirmation and promotion batch 1 (2026-08-15)

- A second representative confirmation tested systems `214`, `2259`, `4631`,
  and spatial-downstream-v2 system `2827`. Each reused its stable saved spin-up
  state and ran only the original forward offset plus `-0.005/-0.010 K` colder
  forward offsets. All physical parameters, B-window climate, precipitation
  anomalies off, geometry, and formal scoring remained frozen.
- All four original forward branches reproduced their saved decadal curves.
  The complete block contains 12 technically clean chains, two observation-
  domain overflows, and no process failure. Two of four representatives improve
  their formal objective; the invalid/domain-overflow rate is 16.67%.
- System `2827` formally passes at spin-up offset `+1.375 K` and forward offset
  `+1.370 K`. Its peak/year-2000 lengths are `14.050/12.600 km`, separation
  errors are `0.18907/0.07509`, total-length errors are `2.5832%/1.1884%`, and
  source spin-up stability is `1.4876%`. It is automatically frozen. The
  registry now contains 14 systems and all recorded hashes reproduce.
- The predeclared confirmation rule is met: at least two representatives
  improve, at least one formal pass occurs, and the invalid rate is below 30%.
  This authorizes one bounded class-promotion batch; it does not authorize all
  unresolved systems or another physical-parameter family.
- Promotion batch 1 contains systems `1301`, `2806`, `3915`, `5143`, and
  `2645`. They are the remaining nearest stable, observation-valid,
  total-length-pass chains with both signed separation endpoints short and
  formal objective below `0.65`, excluding systems already tested by this
  family. Each receives only its source forward offset and offsets
  `-0.005/-0.010 K`, for 15 forward-only chains. Baseline reproduction gates
  each system; formal passes freeze automatically; at most 12 workers are
  allowed.
- Evidence is in
  `validation_data/results/paleo_surface_validation_v0/lia_191_split_forward_delta_confirmation_v1/`.
  Reusable confirmation and freeze drivers are
  `run_lia_191_split_forward_delta_confirmation_v1.py` and
  `freeze_lia_191_split_forward_delta_2827_v1.py`.
'''
if "## 52. Split forward-delta confirmation and promotion batch 1" not in text:
    text = text.rstrip() + section + "\n"
path.write_text(text, encoding="utf-8")
