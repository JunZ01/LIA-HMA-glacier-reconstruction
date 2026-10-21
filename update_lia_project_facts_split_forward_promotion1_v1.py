"""Append split-forward promotion batch 1 closure and batch 2 declaration."""

from pathlib import Path


path = Path("LIA_PROJECT_FACTS.md")
text = path.read_text(encoding="utf-8")
text = text.replace(
    "Last updated: 2026-08-15 (split forward-delta confirmation completed; promotion batch 1 declared)",
    "Last updated: 2026-08-15 (split forward-delta promotion batch 1 completed; batch 2 declared)",
    1,
)
section = r'''

## 53. Split forward-delta promotion batch 1 closure (2026-08-15)

- Promotion batch 1 tested systems `1301`, `2806`, `3915`, `5143`, and
  spatial-downstream-v2 system `2645`. The 15-chain first round used each
  original forward offset and `-0.005/-0.010 K` offsets. All 15 chains were
  technically clean and observation-valid; all five systems improved their
  formal objective, but none passed in that first step.
- A response-directed refinement continued only `1301`, `2806`, and `2645`.
  Nine chains at three colder offsets per system were clean and valid.
  `3915` was closed because the RGI endpoint crossed tolerance before the LIA
  peak, and `5143` was closed because the peak response was nearly flat.
- The declared final six-chain overlap check produced formal passes for
  `1301` and `2645`. System `1301` uses spin-up offset `+0.750 K` and forward
  offset `+0.715 K`; its peak/year-2000 lengths are `8.350/6.500 km` and its
  separation errors are `0.03025/0.02376`. System `2645` uses spin-up offset
  `+0.375 K` and forward offset `+0.340 K`; its peak/year-2000 lengths are
  `15.350/13.000 km` and separation errors are `0.02780/0.00945`.
- The two `2806` final branches overflow the declared observation domain and
  are rejected. Per the predeclared stop rule, all five promotion-1 branches
  are now closed without further micro-search.
- Systems `1301` and `2645` were automatically frozen. The registry now
  contains 16 systems and every recorded file hash and byte count reproduces.
- Promotion batch 2 is declared for the next five both-short systems by saved
  formal objective: `130`, `252`, `518`, `1012`, and spatial-downstream-v2
  system `3899`. Each receives only its original forward offset and
  `-0.010/-0.020 K` offsets, for 15 forward-only chains. Original forward
  reproduction is mandatory; observation overflow remains a hard rejection;
  no physical parameter, spin-up state, or precipitation setting changes.
- Evidence is in
  `validation_data/results/paleo_surface_validation_v0/lia_191_split_forward_delta_promotion1_v1/`,
  `lia_191_split_forward_delta_promotion1_refine_v1/`, and
  `lia_191_split_forward_delta_promotion1_finish_v1/`.
'''
if "## 53. Split forward-delta promotion batch 1 closure" not in text:
    text = text.rstrip() + section + "\n"
path.write_text(text, encoding="utf-8")
