from pathlib import Path


path = Path(__file__).resolve().parent / "LIA_PROJECT_FACTS.md"
text = path.read_text(encoding="utf-8")
heading = "## 56. Split forward-delta promotion batch 4 closure (2026-08-15)"
if heading not in text:
    section = r'''

## 56. Split forward-delta promotion batch 4 closure (2026-08-15)

- Promotion batch 4 tested the next five previously untested both-short
  systems: `2748`, spatial-downstream-v2 system `3523`, `79`, `801`, and
  `3632`. It reused each stable saved spin-up state and ran only the original
  forward offset plus `-0.020/-0.040 K` colder forward offsets. All 15 chains
  were technically clean, observation-valid, and improved their system's
  formal objective; no system passed the formal dual gate in the first round.
- Response geometry closed `3523`, `801`, and `3632` as endpoint tradeoffs.
  A single six-chain refinement continued only systems `79` and `2748`, the
  two branches with remaining endpoint-overlap potential. Every refinement
  chain was technically clean but overflowed the declared observation domain,
  so all six are hard-rejected and no further finish is permitted.
- The batch therefore contains 21 new forward-only chains, zero process or
  technical failures, six observation-domain overflows, and zero formal pass.
  The frozen registry remains at exactly 20 systems; no spin-up state,
  physical parameter, precipitation setting, geometry, or scoring rule was
  changed.
- All five promotion-4 branches are closed. The same split-forward family may
  continue only as another bounded batch on previously untested both-short
  systems, with mandatory source reproduction and the unchanged domain and
  formal acceptance gates.
- Evidence is in
  `validation_data/results/paleo_surface_validation_v0/lia_191_split_forward_delta_promotion4_v1/`
  and
  `validation_data/results/paleo_surface_validation_v0/lia_191_split_forward_delta_promotion4_refine_v1/`.
'''
    text = text.rstrip() + section + "\n"
text = text.replace(
    "Last updated: 2026-08-15 (split forward-delta promotion batch 3 completed)",
    "Last updated: 2026-08-15 (split forward-delta promotion batch 4 completed)",
)
path.write_text(text, encoding="utf-8")
