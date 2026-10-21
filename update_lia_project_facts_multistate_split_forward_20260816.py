"""Append the verified multistate split-forward result to LIA_PROJECT_FACTS.md."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
MARKER = "## 137. Alternate-state split-forward family closure (2026-08-16)"
OLD_HEADER = "Last updated: 2026-08-16 (split forward-delta signed-endpoint classes closed)"
NEW_HEADER = "Last updated: 2026-08-16 (alternate-state split-forward family closed; system 1524 frozen)"

SECTION = r'''

## 137. Alternate-state split-forward family closure (2026-08-16)

- A read-only router inspected 861 reusable stable, observation-valid saved
  states outside the then-20-system frozen registry. Sixty unresolved systems
  contain at least two distinct saved `glen_a` states. The declared mechanism
  reused an alternate already-converged state and then varied only the
  independent 1470-2000 forward `delta_t`; no spin-up or geometry was
  recomputed, precipitation anomalies remained off, and every source-forward
  reproduction gate was enforced.
- The representative pilot, one response-directed refinement, one explicit
  endpoint-overlap finish, and two five-system promotions contain 48 real
  forward-only OGGM chains across 13 systems. All 48 are technically clean;
  three branches overflow the declared observation domain and are hard
  rejected.
- System `1524` is the single new formal pass. It reuses the Stage-2
  `glen_a x1.0` stable state at spin-up `delta_t=0.000 K` and uses forward
  `delta_t=-0.0375 K`. Its forward peak/year-2000 lengths are
  `11.850/9.550 km`; LIA/RGI separation errors are
  `0.145889/0.186455`, total-length errors are `4.0685%/7.2108%`, and
  source spin-up stability is `1.1528%`. The chain is technically clean,
  stable, and observation-domain valid, so it was automatically frozen.
- The first promotion produced no pass: systems `584` and `2259` were flat,
  while `313`, `843`, and `2255` retained endpoint tradeoffs. The final
  promotion (`5265`, `5524`, `904`, `2005`, and `2547`) also produced no
  pass or endpoint-overlap bracket. Close the alternate-saved-`glen_a`-state
  plus forward-`delta_t` family; do not continue it glacier by glacier.
- The frozen registry now contains exactly 21 systems:
  `130`, `252`, `518`, `991`, `1077`, `1301`, `1524`, `1546`, `1616`,
  `2266`, `2337`, `2569`, `2622`, `2645`, `2815`, `2827`, `2902`,
  `2928`, `2939`, `4319`, and `4565`. Every recorded file hash and byte
  count reproduces with zero mismatch.
- Any next OGGM batch must declare a mechanism outside both the closed
  forward-offset-only family and this closed alternate-`glen_a`-state family,
  while preserving the formal length score, stable-state requirement, and
  geometry/observation-domain gates.
- Consolidated evidence is in
  `validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/diagnostics/multistate_split_forward_family_v1/`.
'''


def main() -> None:
    text = FACTS.read_text(encoding="utf-8")
    if MARKER in text:
        raise SystemExit("facts-section-already-present")
    if OLD_HEADER not in text:
        raise RuntimeError("unexpected-facts-header")
    text = text.replace(OLD_HEADER, NEW_HEADER, 1).rstrip() + SECTION + "\n"
    FACTS.write_text(text, encoding="utf-8")
    print("updated", FACTS)


if __name__ == "__main__":
    main()
