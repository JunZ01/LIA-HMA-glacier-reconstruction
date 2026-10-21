from pathlib import Path


path = Path(__file__).resolve().parent / "LIA_PROJECT_FACTS.md"
text = path.read_text(encoding="utf-8")
heading = "## 59. Split forward-delta peak-short/RGI-long confirmation closure (2026-08-16)"
if heading not in text:
    section = r'''

## 59. Split forward-delta peak-short/RGI-long confirmation closure (2026-08-16)

- A four-system confirmation tested unresolved peak-short/RGI-long systems
  `584`, `2697`, `2255`, and `3850`. Each stable saved spin-up state was
  reused and only the forward offset changed by `-0.005/0/+0.005 K` around
  its source value. All 12 branches are technically clean,
  observation-valid, and reproduce their source forward curves.
- No system passes both formal endpoints. Two representatives improve, but
  the predeclared promotion rule requires at least one formal pass. System
  `584` is unchanged across the perturbation. Systems `2255` and `3850`
  retain endpoint tradeoffs.
- System `2697` gives the closest diagnostic bracket: at forward
  `delta_t=+0.185 K` its LIA/RGI separation errors are `0.14698/0.26557`,
  while at `+0.190 K` they are `0.27753/0.20030`. The two endpoints cross
  without a shared 20% interval under the discrete length response, so no
  micro-bisection is justified.
- Close this class without population promotion. The frozen registry remains
  at 20 systems; no spin-up state, physical parameter, precipitation setting,
  geometry, or score changed.
- Evidence is in
  `validation_data/results/paleo_surface_validation_v0/lia_191_split_forward_delta_peak_short_rgi_long_confirmation_v1/`.
'''
    text = text.rstrip() + section + "\n"
text = text.replace(
    "Last updated: 2026-08-15 (split forward-delta both-long class closed)",
    "Last updated: 2026-08-16 (split forward-delta peak-short/RGI-long class closed)",
)
path.write_text(text, encoding="utf-8")
