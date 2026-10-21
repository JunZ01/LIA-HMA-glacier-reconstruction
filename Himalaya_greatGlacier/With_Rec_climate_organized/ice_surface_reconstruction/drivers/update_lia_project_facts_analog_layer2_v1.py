"""Append the completed analog Layer-2 result and declared glen_a block."""

from __future__ import annotations

import os
from pathlib import Path


OLD_HEADER = b"Last updated: 2026-08-14 (system 991 curve frozen; five-system analog batch declared)"
NEW_HEADER = b"Last updated: 2026-08-14 (analog melt_f-delta-T batch completed; glen_a block declared)"

SECTION = """## 20. Five-system analog Layer-2 result and glen_a block (2026-08-14)

- The system-991 analog `melt_f`-`delta_t` batch completed all 30 chains for
  systems `2569`, `2645`, `214`, `4631`, and `2622`. There were zero technical
  failures, 23 chains passed the adaptive spin-up and observation-domain gates,
  and every saved observation length was assessed. The mandatory
  `2569/delta_t=-0.250 K/melt_f x1.10` preflight converged with `1.029%`
  preceding-200-year volume change and released the remaining 29 tasks.
- No chain passed both 20% separation endpoints. Five chains passed both 10%
  total-length endpoints, so no new curve is frozen from Layer 2. Best stable
  formal-separation results by system are:
  | system | delta-T K | melt_f xStage-2 | E_LIA_sep | E_RGI_sep | amplitude ratio |
  |---:|---:|---:|---:|---:|---:|
  | 214 | -0.625 | 1.20 | 0.336 | 0.022 | 0.686 |
  | 2569 | -0.500 | 1.20 | 0.213 | 0.043 | 0.830 |
  | 2622 | 0.000 | 1.10 | 0.245 | 0.127 | 0.883 |
  | 2645 | +0.125 | 1.10 | 0.811 | 0.296 | 0.486 |
  | 4631 | 0.000 | 1.10 | 0.300 | 0.105 | 0.805 |
- System `2645` does not enter the next block because even its best point misses
  the LIA total-length gate and remains far outside both separation targets.
  Systems `214`, `2569`, `2622`, and `4631` pass both total-length endpoints
  at their best Layer-2 points but retain a bounded short-LIA/amplitude deficit.
  They therefore route to exactly one diagnosis-driven `glen_a` family block.
- The declared Layer-3 block freezes each system's best Layer-2 `delta_t` and
  `melt_f`, freezes Stage-2 `prcp_fac`, `temp_melt=-1 degC`, precipitation
  anomalies off, climate, geometry, and observation definitions, and tests
  Stage-2 `glen_a` multipliers `1.5/2.0/2.5/3.0`. This is 16 tasks total.
  It asks whether increased dynamic response closes the remaining LIA
  separation error without losing the RGI endpoint; it is not a global
  multi-parameter search.
- The mandatory glen_a preflight is system `2569`, `delta_t=-0.500 K`,
  Stage-2 `melt_f x1.20`, and Stage-2 `glen_a x2.0`, with one worker. The
  remaining block may use at most 12 workers only if the numerical, adaptive
  spin-up, and observation-domain gates pass. Results and reusable summaries
  for Layer 2 are in
  `validation_data/results/paleo_surface_validation_v0/lia_191_analog_melt_delta_after_991_v1/`;
  the Layer-3 design is in
  `validation_data/results/paleo_surface_validation_v0/lia_191_analog_glen_after_991_v1/`.
"""


def main() -> None:
    path = Path(__file__).resolve().parents[4] / "LIA_PROJECT_FACTS.md"
    original = path.read_bytes()
    section_bytes = SECTION.encode("utf-8")
    if section_bytes in original:
        print("analog Layer-2 facts section already present")
        return
    if original.count(OLD_HEADER) != 1:
        raise ValueError("expected exactly one preceding Last updated header")
    updated = original.replace(OLD_HEADER, NEW_HEADER, 1)
    updated = updated.rstrip(b"\r\n") + b"\n\n" + section_bytes + b"\n"
    temporary = path.with_name(path.name + ".tmp_analog_layer2_v1")
    temporary.write_bytes(updated)
    os.replace(temporary, path)
    print(f"updated {path}")


if __name__ == "__main__":
    main()
