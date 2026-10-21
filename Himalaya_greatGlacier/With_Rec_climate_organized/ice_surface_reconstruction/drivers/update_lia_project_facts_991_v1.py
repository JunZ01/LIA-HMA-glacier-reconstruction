"""Apply the verified system-991 facts update without normalizing old line endings.

The canonical facts file currently contains mixed historical line endings, so
ordinary patch tools cannot match it reliably.  This one-time updater replaces
one exact ASCII header and appends one exact UTF-8 section atomically.
"""

from __future__ import annotations

import os
from pathlib import Path


OLD_HEADER = (
    b"Last updated: 2026-08-14 (post-observation-domain-gate Layer-1 "
    b"delta-T extension completed)"
)
NEW_HEADER = (
    b"Last updated: 2026-08-14 (system 991 curve frozen; five-system analog "
    b"batch declared)"
)

SECTION = """## 19. System 991 frozen curve and analog-batch transition (2026-08-14)

- The user-approved local Layer-2/Layer-3 continuation for system `991`
  supersedes its `hold_after_layer1` status in Section 18. The selected chain is
  `991-B_finish_refine_991_v2-703e4cca9afaba39`, with `delta_t=-0.125 K`,
  Stage-2 `melt_f x1.10`, Stage-2 `glen_a x2.90`, Stage-2 `prcp_fac`,
  `temp_melt=-1 degC`, and precipitation anomalies off. The B-window
  `1441-1470` spin-up and `1470-2000` forward climate, geometry, and
  observation-domain definition are unchanged.
- The chain is technically clean, converged at spin-up year 500, and has
  `0.6815%` preceding-200-year volume change. It has no observation-domain
  overflow or unresolved observation length. Its forward peak is `11.550 km`
  in 1830 versus the `12.070 km` LIA target, and its year-2000 length is
  `9.900 km` versus the `9.361 km` RGI target. Total-length errors are
  `4.309%/5.758%`; separation-normalized errors are `0.1920/0.1990`.
  Therefore it passes both the active 10% total-length rule and the 20%
  separation diagnostic.
- The complete saved trajectory remains inside the 20%-expanded LIA-RGI
  corridor. Its forward minimum is `9.500 km` in 1970 rather than 2000, and
  its peak-to-2000 amplitude is `1.650 km`, or `60.91%` of the target
  separation. `minimum_at_2000` remains an auxiliary diagnostic, not a hard
  rejection. The user reviewed the complete spin-up and 1470-2000 length
  plot and explicitly approved retaining this curve.
- The candidate is frozen as a provisional historical curve, not promoted to
  a final parameter estimate. Its complete length/area/volume checkpoints,
  spin-up and year-2000 states, source metadata, and hashes are stored in
  `validation_data/results/paleo_surface_validation_v0/lia_191_frozen_curve_registry_v1/system_991/`.
  Reusable freeze code is
  `Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/freeze_lia_191_candidate_991_v1.py`.
- The read-only router now merges the three final 991 chains as source
  `finish_refine_991_v2`. It contains 2382 score rows, identifies 57 systems
  with a direct 10% total-length candidate, and routes system 991 to
  `freeze_provisional_candidate` using the selected task above. Mutually
  exclusive route counts are 55 `freeze_provisional_candidate`, 81
  `hold_after_layer1`, 13 `layer3_closed`, 23 `small_separation_review`, 15
  `target_direction_review`, two `blocked_dynamic_review`, and two
  `named_configuration_closed`. Outputs are in
  `validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/diagnostics/iterative_loop_after_991_v1/`.
- The next declared OGGM block is a five-system analog Layer-2 test for
  systems `2569`, `2645`, `214`, `4631`, and `2622`. They are the five nearest
  unresolved Layer-1 responses to pre-refinement 991 in scaled LIA error, RGI
  error, and amplitude ratio, under the explicit filter: retreat target,
  year-2000 total-length error at most 10%, LIA-peak total-length error between
  10% and 25%, and amplitude ratio between 0.30 and 0.85. Each receives three
  delta-T values `0.125/0.250/0.375 K` colder than its best stable Layer-1
  anchor and Stage-2 `melt_f` multipliers `1.10/1.20`, for 30 tasks total.
  `glen_a`, `prcp_fac`, `temp_melt`, precipitation anomalies, climate,
  geometry, and observation definitions remain frozen. This asks whether the
  colder-delta-T plus higher-melt_f mechanism that first improved 991
  generalizes before any glen_a layer is opened; it is not a blind population
  glen_a sweep.
- The mandatory preflight is system `2569`, `delta_t=-0.250 K`, Stage-2
  `melt_f x1.10`, one worker. The remaining batch may use at most 12 workers
  only if that chain is technically clean, converged within the 2% stability
  gate, and has a valid observation-domain length. Configuration and declared
  design are in
  `validation_data/results/paleo_surface_validation_v0/lia_191_analog_melt_delta_after_991_v1/`
  and
  `Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/lia_191_analog_melt_delta_after_991_v1/`.
"""


def main() -> None:
    path = Path(__file__).resolve().parents[4] / "LIA_PROJECT_FACTS.md"
    original = path.read_bytes()
    section_bytes = SECTION.encode("utf-8")
    if section_bytes in original:
        print("system-991 facts section already present")
        return
    if original.count(OLD_HEADER) != 1:
        raise ValueError("expected exactly one old Last updated header")
    updated = original.replace(OLD_HEADER, NEW_HEADER, 1)
    updated = updated.rstrip(b"\r\n") + b"\n\n" + section_bytes + b"\n"
    temporary = path.with_name(path.name + ".tmp_991_v1")
    temporary.write_bytes(updated)
    os.replace(temporary, path)
    print(f"updated {path}")


if __name__ == "__main__":
    main()
