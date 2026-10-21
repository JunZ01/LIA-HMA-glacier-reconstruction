from pathlib import Path


path = Path(__file__).resolve().parent / "LIA_PROJECT_FACTS.md"
text = path.read_text(encoding="utf-8")
heading = "## 60. Split forward-delta signed-endpoint class closure (2026-08-16)"
if heading not in text:
    section = r'''

## 60. Split forward-delta signed-endpoint class closure (2026-08-16)

- The complete three-system peak-long/RGI-short class (`2955`, `520`, and
  `4491`) reused stable saved spin-up states and tested only forward
  `delta_t` at source and `+/-0.005 K`. All nine chains are technically clean,
  observation-valid, and reproduce the source branches. No chain passes the
  formal dual gate. System `2955` remains closest at errors
  `0.14332/0.51135`; systems `520` and `4491` retain much larger RGI-short
  errors. Close the class without refinement.
- The consolidated signed-endpoint continuation since promotion batch 3
  contains 93 new forward-only scientific chains across 23 systems. There are
  zero process or technical failures, 22 hard observation-domain overflows,
  and one formal pass: system `4565`. No spin-up state or physical parameter
  was recomputed or changed.
- Together with the preceding 103 representative/promotion chains, the saved-
  state split-forward family now contains 196 forward-only chains and has
  added nine frozen systems in total. The frozen registry contains exactly 20
  systems and the latest freeze validation reproduces every recorded hash and
  byte count.
- The automatic split-forward signed classes are now closed: the useful
  both-short region was exhausted before the high-objective overflow tail;
  both-long is an endpoint tradeoff; peak-short/RGI-long failed its promotion
  confirmation; and the complete peak-long/RGI-short class produced no pass.
  Do not continue delta-T micro-search glacier by glacier.
- Any next OGGM batch must declare a new mechanism that changes more than the
  forward offset while preserving the formal score and geometry/domain gates.
  This is a scientific family boundary, not a request for individual glacier
  approval.
- Consolidated evidence is in
  `validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/diagnostics/split_forward_delta_signed_classes_v1/`.
'''
    text = text.rstrip() + section + "\n"
text = text.replace(
    "Last updated: 2026-08-16 (split forward-delta peak-short/RGI-long class closed)",
    "Last updated: 2026-08-16 (split forward-delta signed-endpoint classes closed)",
)
path.write_text(text, encoding="utf-8")
