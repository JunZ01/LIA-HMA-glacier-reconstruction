"""Schema-corrected v2 entrypoint for the Cycle-4 3850 timing chain.

V1 was not run: its pre-run schema inspection found that it used shortened
reference-column aliases.  This wrapper preserves the V1 implementation while
adapting the authoritative registered-length column names in memory only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import run_cycle4_state_shooting_3850_timing_v1 as core


REPO_ROOT = Path(__file__).resolve().parents[4]
DRIVER_VERSION = "cycle4-state-shooting-3850-timing-v2"


def _schema_corrected_select_inputs(config):
    ref, stage2, pixel = _ORIGINAL_SELECT_INPUTS(config)
    ref = ref.copy()
    ref["lia_length_m"] = ref["lia_registered_length_m"]
    ref["rgi_length_m"] = ref["rgi_registered_length_m"]
    return ref, stage2, pixel


_ORIGINAL_SELECT_INPUTS = core._select_inputs


def run(config_path: Path):
    core.DRIVER_VERSION = DRIVER_VERSION
    core._select_inputs = _schema_corrected_select_inputs
    return core.run(config_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "Himalaya_greatGlacier"
        / "With_Rec_climate_organized"
        / "ice_surface_reconstruction"
        / "config"
        / "cycle4_state_shooting_3850_timing_v2.json",
    )
    args = parser.parse_args()
    decision = run(args.config.resolve())
    print(json.dumps(core._json_safe(decision), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
