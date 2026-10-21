"""Append the completed analog Layer-3 glen_a result to project facts."""

from __future__ import annotations

import os
from pathlib import Path


OLD_HEADER = b"Last updated: 2026-08-14 (analog melt_f-delta-T batch completed; glen_a block declared)"
NEW_HEADER = b"Last updated: 2026-08-14 (analog glen_a block completed and closed)"

SECTION = """## 21. Four-system analog Layer-3 glen_a result (2026-08-14)

- The declared Layer-3 block completed all 16 chains for systems `214`,
  `2569`, `2622`, and `4631`, using four Stage-2 `glen_a` multipliers per
  system (`1.5/2.0/2.5/3.0`) and each system's frozen best Layer-2
  `delta_t` and `melt_f`. All 16 chains are technically clean, converged within
  the 2% preceding-200-year volume-stability gate, and retain valid assessed
  observation-domain lengths with no overflow or unresolved case.
- No chain passes both 20% separation endpoints. One chain passes both 10%
  total-length endpoints, and no chain passes the required dual gate. No new
  curve is frozen from this block.
- Increasing `glen_a` moved all four systems in the wrong direction: both the
  LIA peak and year-2000 length shortened, and the best tested Layer-3 point
  was always the lower `glen_a x1.5` boundary. Relative to the corresponding
  Layer-2 `glen_a x1.0` anchor, the formal separation objective changed as:

  | system | objective at x1.0 | best objective at x1.5 | result |
  |---:|---:|---:|---|
  | 214 | 0.357 | 0.701 | worsened |
  | 2569 | 0.257 | 0.761 | worsened |
  | 2622 | 0.372 | 1.011 | worsened |
  | 4631 | 0.405 | 1.738 | worsened |
- The tested `glen_a` direction contains no endpoint bracket, so the declared
  one-family block is closed without another delta-T recenter or a lower-
  `glen_a` extension. This is a named-configuration response result, not a
  statement that any glacier is abnormal.
- Before declaring another OGGM block, perform a read-only dual-gate inventory
  of all saved Core-191 chains. This separates already usable curves from
  systems that genuinely require another parameter-search layer and prevents
  repeating the failed 991-analog mechanism indiscriminately.
- Reusable scores and summaries are in
  `validation_data/results/paleo_surface_validation_v0/lia_191_analog_glen_after_991_v1/`;
  the summarizer is
  `Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/summarize_lia_191_analog_glen_after_991_v1.py`.
"""


def main() -> None:
    path = Path(__file__).resolve().parents[4] / "LIA_PROJECT_FACTS.md"
    original = path.read_bytes()
    section_bytes = SECTION.encode("utf-8")
    if section_bytes in original:
        print("analog glen_a facts section already present")
        return
    if original.count(OLD_HEADER) != 1:
        raise ValueError("expected exactly one preceding Last updated header")
    updated = original.replace(OLD_HEADER, NEW_HEADER, 1)
    updated = updated.rstrip(b"\r\n") + b"\n\n" + section_bytes + b"\n"
    temporary = path.with_name(path.name + ".tmp_analog_glen_v1")
    temporary.write_bytes(updated)
    os.replace(temporary, path)
    print(f"updated {path}")


if __name__ == "__main__":
    main()
