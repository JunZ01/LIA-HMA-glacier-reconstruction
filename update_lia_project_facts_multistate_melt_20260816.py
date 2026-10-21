"""Append the verified alternate-melt-state pilot closure to project facts."""

from pathlib import Path


FACTS = Path(__file__).resolve().parent / "LIA_PROJECT_FACTS.md"
MARKER = "## 142. Alternate-melt-state split-forward pilot closure (2026-08-16)"
OLD_HEADER = "Last updated: 2026-08-16 (post-final-test climate boundary; alternate-state split-forward family closed; system 1524 frozen)"
NEW_HEADER = "Last updated: 2026-08-16 (alternate-melt-state split-forward pilot closed; 21 systems frozen)"

SECTION = r'''

## 142. Alternate-melt-state split-forward pilot closure (2026-08-16)

- A read-only router found 23 unresolved systems with at least two distinct
  reusable, stable saved `melt_f` states after excluding the 21 frozen systems
  and every state already used by a split-forward experiment. The declared
  mechanism reused one untested converged `melt_f` state and varied only the
  independent forward `delta_t`; no spin-up, geometry, physical parameter,
  precipitation setting, or formal score was recomputed.
- Representatives were systems `2259`, `2697`, and `5168`. The pilot contains
  nine real forward-only OGGM chains. All nine are technically clean,
  observation-domain valid, and reproduce their source-forward baselines.
- Only system `2697` improves its formal objective, and no system passes the
  formal dual gate. System `2697` crosses from LIA/RGI separation errors
  `0.17962/0.29821` at forward `delta_t=+0.245 K` to
  `0.37543/0.13503` at `+0.255 K`, without a shared 20% interval. System
  `2259` retains a short peak near `0.303` separation error, and system `5168`
  shortens its peak before the long RGI endpoint approaches tolerance.
- The predeclared promotion rule fails: fewer than two representatives improve
  and there is no formal pass. Close the alternate-saved-`melt_f`-state plus
  forward-`delta_t` family at the pilot boundary without refinement or
  population promotion. The frozen registry remains at exactly 21 systems.
- Evidence is in
  `validation_data/results/paleo_surface_validation_v0/lia_191_multistate_melt_split_forward_pilot_v1/`
  and
  `validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/diagnostics/multistate_melt_split_forward_router_v1/`.
'''


def main() -> None:
    text = FACTS.read_text(encoding="utf-8")
    if MARKER in text:
        raise SystemExit("facts-section-already-present")
    if text.count("## 141. Alternate-state split-forward family closure (2026-08-16)") != 1:
        raise RuntimeError("unexpected-section-141-state")
    if text.count(OLD_HEADER) != 1:
        raise RuntimeError("unexpected-facts-header")
    text = text.replace(OLD_HEADER, NEW_HEADER, 1).rstrip() + SECTION + "\n"
    FACTS.write_text(text, encoding="utf-8")
    print("updated", FACTS)


if __name__ == "__main__":
    main()
