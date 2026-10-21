from pathlib import Path


path = Path(__file__).resolve().parent / "LIA_PROJECT_FACTS.md"
text = path.read_text(encoding="utf-8")
heading = "## 57. Split forward-delta promotion batch 5 and both-short tail closure (2026-08-15)"
if heading not in text:
    section = r'''

## 57. Split forward-delta promotion batch 5 and both-short tail closure (2026-08-15)

- Promotion batch 5 tested the next five farther both-short systems: `705`,
  `2649`, `4369`, `1931`, and `2342`. It reused the saved stable spin-up
  states and ran the source forward offset plus `-0.05/-0.10 K` colder
  offsets. All 15 chains were technically clean; three directed chains
  overflowed the observation domain, four systems improved, and none passed
  the formal dual gate.
- One response-directed six-chain refinement continued only `2649` and `4369`.
  Five chains overflowed. System `2649` remained valid at forward
  `delta_t=+0.390 K`, with LIA/RGI separation errors `0.09545/0.35558`, but
  its next colder `+0.380 K` branch already overflowed. No endpoint-overlap
  finish is justified. System `4369` overflowed at every refinement point.
- Promotion 5 therefore added 21 forward-only chains with zero process or
  technical failure, eight observation-domain overflows, and zero formal
  pass. The frozen registry remains at 20 systems; no spin-up state, physical
  parameter, precipitation setting, geometry, or formal score changed.
- The remaining previously untested both-short candidates begin at formal
  objective above `4.0`, while promotion-4 and promotion-5 refinements show
  repeated overflow before the missing endpoint distance closes. Close the
  untested both-short tail for automatic split-forward cooling. This is a
  class-level diminishing-return and domain-gate decision, not a statement
  that any glacier is abnormal.
- Continuing the split-forward architecture requires a separately declared
  signed-endpoint class question rather than additional blind cooling of the
  both-short tail.
- Evidence is in
  `validation_data/results/paleo_surface_validation_v0/lia_191_split_forward_delta_promotion5_v1/`
  and
  `validation_data/results/paleo_surface_validation_v0/lia_191_split_forward_delta_promotion5_refine_v1/`.
'''
    text = text.rstrip() + section + "\n"
text = text.replace(
    "Last updated: 2026-08-15 (split forward-delta promotion batch 4 completed)",
    "Last updated: 2026-08-15 (split forward-delta both-short tail closed)",
)
path.write_text(text, encoding="utf-8")
