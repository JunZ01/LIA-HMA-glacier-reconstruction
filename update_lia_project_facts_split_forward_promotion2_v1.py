"""Append split-forward promotion batch 2 closure."""

from pathlib import Path


path = Path("LIA_PROJECT_FACTS.md")
text = path.read_text(encoding="utf-8")
text = text.replace(
    "Last updated: 2026-08-15 (split forward-delta promotion batch 1 completed; batch 2 declared)",
    "Last updated: 2026-08-15 (split forward-delta promotion batch 2 completed)",
    1,
)
section = r'''

## 54. Split forward-delta promotion batch 2 closure (2026-08-15)

- Promotion batch 2 tested systems `130`, `252`, `518`, `1012`, and
  spatial-downstream-v2 system `3899`. The first 15 chains used the source
  forward offset and `-0.010/-0.020 K` colder offsets. All chains were
  technically clean and observation-valid; all five formal objectives
  improved, but no system passed at those first offsets.
- One response-scaled 10-chain refinement remained technically clean with no
  observation overflow. Systems `130` and `252` passed the formal dual gate.
  System `130` uses spin-up/forward offsets `-0.250/-0.350 K`, peak/year-2000
  lengths `11.500/10.100 km`, and separation errors `0.18309/0.19553`.
  System `252` uses `+0.375/+0.295 K`, lengths `8.450/7.000 km`, and errors
  `0.15527/0.16253`.
- A final four-chain check was allowed only for `518` and `1012`. System `518`
  passes at spin-up/forward offsets `+1.000/+0.800 K`, with peak/year-2000
  lengths `19.750/16.400 km` and separation errors `0.08360/0.02160`.
  System `1012` reaches a good LIA peak at forward `+0.455 K` but the RGI
  separation error remains `0.24218`; its colder `+0.450 K` branch overflows
  the observation domain. It is closed. System `3899` is also closed because
  its endpoint tradeoff remains far from joint tolerance.
- Systems `130`, `252`, and `518` were automatically frozen. Together with
  promotion batch 1 and the representative blocks, the frozen registry now
  contains exactly 19 systems. All recorded package hashes and byte counts
  reproduce with zero mismatch.
- Across the new split-forward family in this execution, eight systems were
  newly frozen: `130`, `252`, `518`, `1301`, `2569`, `2622`, `2645`, and
  `2827`. The historical total increased from eleven to nineteen without
  changing a spin-up state or a physical parameter.
- Evidence is in
  `validation_data/results/paleo_surface_validation_v0/lia_191_split_forward_delta_promotion2_v1/`,
  `lia_191_split_forward_delta_promotion2_refine_v1/`, and
  `lia_191_split_forward_delta_promotion2_finish_v1/`.
'''
if "## 54. Split forward-delta promotion batch 2 closure" not in text:
    text = text.rstrip() + section + "\n"
path.write_text(text, encoding="utf-8")
