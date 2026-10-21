"""Schema-only invocation shim for the system-1026 EB50 gate.

The registered per-system audit predates the aggregate
``all_geometry_gates_pass`` field expected by v1.  This wrapper derives that
boolean from the frozen component metrics, creates a byte-preserving input
overlay, and calls the unchanged v1 scientific runner with a fresh output.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .run_negative_flux_eb50_state_preserving_1026_v1 import PROJECT, run


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    source_config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    source_root = resolve(source_config["elevation_band_root"])
    source_system = source_root / str(source_config["system_id"])
    audit = json.loads((source_system / "elevation_band_audit.json").read_text(encoding="utf-8"))
    checkpoint = audit["zero_mb_checkpoints"][-1]
    audit["all_geometry_gates_pass"] = bool(
        abs(float(audit["area_error_fraction"])) <= 1e-10
        and abs(float(audit["volume_error_fraction"])) <= 1e-10
        and float(audit["hypsometry_maximum_cumulative_area_error_fraction"]) <= 0.02
        and abs(float(audit["initial_length_error_m"]))
        / float(audit["target_main_flowline_length_m"]) <= 0.10
        and abs(float(checkpoint["volume_change_fraction_from_initial"])) <= 0.001
        and float(checkpoint["downstream_free_distance_m"]) >= 5000.0
        and float(checkpoint["terminal_thickness_m"]) <= 10.0
    )
    if not audit["all_geometry_gates_pass"]:
        raise RuntimeError("derived-registered-static-gate-failed")

    overlay = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux_eb50_state_preserving_v1/audit_overlay_1026_v2"
    if overlay.exists():
        raise FileExistsError(f"versioned-overlay-already-exists:{overlay}")
    target_system = overlay / str(source_config["system_id"])
    target_system.mkdir(parents=True)
    shutil.copy2(source_system / "elevation_band_flowline_lia.pkl.gz", target_system / "elevation_band_flowline_lia.pkl.gz")
    (target_system / "elevation_band_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    invocation = dict(source_config)
    invocation["elevation_band_root"] = str(overlay)
    invocation["schema"] = "negative-flux-eb50-state-preserving-1026-v2"
    invocation_path = overlay / "invocation_config.json"
    invocation_path.write_text(json.dumps(invocation, indent=2), encoding="utf-8")
    print(json.dumps(run(invocation_path), indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
