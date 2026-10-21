"""Pre-run parent-physics corrected entrypoint for Cycle-4 Gen0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import run_cycle4_state_shooting_gen0_v1 as core


REPO_ROOT = Path(__file__).resolve().parents[4]


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def run(overlay_path: Path):
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    master = json.loads(_resolve(overlay["base_config"]).read_text(encoding="utf-8"))
    master["experiment_id"] = overlay["experiment_id"]
    for representative in master["representatives"]:
        override = overlay["representative_overrides"].get(str(representative["system_id"]), {})
        if "parent_physics" in override:
            representative["parent_physics"].update(override["parent_physics"])
    master["paths"].update(overlay["path_overrides"])
    master["pre_run_schema_and_parent_physics_audit"] = {
        "base_v1_executed": False,
        "correction": overlay["pre_run_correction"],
        "overlay_config": str(overlay_path),
    }
    work_root = _resolve(master["paths"]["workdir_root"])
    work_root.mkdir(parents=True, exist_ok=True)
    resolved = work_root / "resolved_master_config_v2.json"
    resolved.write_text(json.dumps(master, ensure_ascii=False, indent=2), encoding="utf-8")
    return core.run(resolved)


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
        / "cycle4_state_shooting_gen0_three_representatives_v2.json",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
