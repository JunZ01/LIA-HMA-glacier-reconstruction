from pathlib import Path


path = Path(__file__).resolve().parent / "LIA_PROJECT_FACTS.md"
text = path.read_text(encoding="utf-8")
heading = "## 58. Split forward-delta both-long class closure (2026-08-15)"
if heading not in text:
    section = r'''

## 58. Split forward-delta both-long class closure (2026-08-15)

- After closing the both-short tail, the untested both-long signed-endpoint
  class contained only spatial-downstream-v2 system `2570`; system `843` had
  already been closed by the representative split-forward pilot. System 2570
  reused its stable source state at spin-up/forward `delta_t=-0.250 K` and
  tested forward-only warming to `-0.230/-0.210 K`.
- The first controller attempt stopped before OGGM because historical spatial
  reruns store `source_task.json` rather than `task.json`. The generic pilot
  and saved-state driver now accept either filename, preferring `task.json`.
  Both files still require the same source summary, spin-up stability,
  precipitation-off, geometry, and observation-domain contracts. Syntax and
  source-forward reproduction passed after the compatibility fix.
- All three scientific branches are technically clean and observation-valid.
  The source branch has LIA/RGI separation errors `0.03943/0.79803`.
  Forward `delta_t=-0.230 K` shortens the RGI endpoint but also collapses the
  LIA peak, giving errors `0.89598/0.40895`; `-0.210 K` worsens both further.
  No formal pass occurs and no refinement is justified.
- The both-long class is closed as a split-forward endpoint tradeoff. The
  frozen registry remains at 20 systems and no physical parameter, spin-up
  state, geometry, precipitation setting, or formal score changed.
- Evidence and reusable compatibility code are in
  `validation_data/results/paleo_surface_validation_v0/lia_191_split_forward_delta_both_long_2570_v1/`,
  `run_lia_191_split_forward_delta_pilot_v1.py`, and
  `lia_forward_delta_from_saved_spinup.py`.
'''
    text = text.rstrip() + section + "\n"
text = text.replace(
    "Last updated: 2026-08-15 (split forward-delta both-short tail closed)",
    "Last updated: 2026-08-15 (split forward-delta both-long class closed)",
)
path.write_text(text, encoding="utf-8")
