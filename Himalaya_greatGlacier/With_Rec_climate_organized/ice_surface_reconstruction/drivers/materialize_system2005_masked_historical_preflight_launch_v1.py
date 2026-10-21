"""Materialize route-bound launch configs for the v10.154 2D preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
ROUTE = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_154/summary.json"
ROUTE_SHA256 = "33F85B1C629A18698BE3D54FCD609DD7D99C907E9FCC483ED416C4FD4BCB0E5D"
PREPARATION = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_system2005_masked_historical_preparation_v1/summary.json"
PREPARATION_SHA256 = "1A9126AFA032CA7C2C5ECEC478A1213AC87D3488786CC384B82BCF31A849E41F"
CONFIG_DIR = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config"
PAIRS = (
    (
        "system2005_masked_historical_preflight_endp0p834281_v1.json",
        "system2005_masked_historical_preflight_endp0p834281_v2_launch.json",
    ),
    (
        "system2005_masked_historical_preflight_endp1p334281_v1.json",
        "system2005_masked_historical_preflight_endp1p334281_v2_launch.json",
    ),
)
OUTPUT = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_system2005_masked_historical_preflight_prelaunch_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def bytes_for(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def main() -> None:
    if sha256(ROUTE) != ROUTE_SHA256 or sha256(PREPARATION) != PREPARATION_SHA256:
        raise RuntimeError("prelaunch-authority-hash-mismatch")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    state = json.loads((ROOT / "LIA_CURRENT_STATE.json").read_text(encoding="utf-8"))
    if state["state_id"] != "2026-08-29-v10.154-system2005-historical-response-preflight-queued":
        raise RuntimeError("unexpected-controller-state")
    if not state["next_action"]["launch_authorized"] or state["counts"]["queued"] != 2:
        raise RuntimeError("controller-does-not-authorize-two-branch-launch")
    OUTPUT.mkdir(parents=True)
    rows = []
    for source_name, target_name in PAIRS:
        source = CONFIG_DIR / source_name
        target = CONFIG_DIR / target_name
        if target.exists():
            raise FileExistsError(target)
        config = json.loads(source.read_text(encoding="utf-8"))
        if (ROOT / config["output_root"]).exists():
            raise FileExistsError(ROOT / config["output_root"])
        config.update(
            {
                "launch_route": str(ROUTE.relative_to(ROOT)).replace("\\", "/"),
                "launch_route_sha256": ROUTE_SHA256,
                "launch_authorized": True,
                "prelaunch_evidence": str((OUTPUT / "summary.json").relative_to(ROOT)).replace(
                    "\\", "/"
                ),
            }
        )
        target.write_bytes(bytes_for(config))
        rows.append(
            {
                "source_config": str(source.relative_to(ROOT)).replace("\\", "/"),
                "source_config_sha256": sha256(source),
                "launch_config": str(target.relative_to(ROOT)).replace("\\", "/"),
                "launch_config_sha256": sha256(target),
                "output_root": config["output_root"],
                "temperature_offset_2000_k": config["temperature_offset_2000_k"],
            }
        )
    summary = {
        "schema": "lia-191-system2005-masked-historical-preflight-prelaunch-v1",
        "status": "prelaunch-pass",
        "system_id": 2005,
        "route": str(ROUTE.relative_to(ROOT)).replace("\\", "/"),
        "route_sha256": ROUTE_SHA256,
        "preparation": str(PREPARATION.relative_to(ROOT)).replace("\\", "/"),
        "preparation_sha256": PREPARATION_SHA256,
        "branches": rows,
        "launch_policy": "sequential on the single registered GPU; colder branch first",
        "model_run": False,
        "initial_thickness_modified": False,
        "production_temperature_modified": False,
    }
    (OUTPUT / "summary.json").write_bytes(bytes_for(summary))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
