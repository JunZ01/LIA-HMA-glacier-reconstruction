"""Materialize v10.155 launch configs using the calendar-window v2 driver."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
ROUTE = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_155/summary.json"
ROUTE_SHA256 = "161C7199F21F0143B437DBDCA9BCB870E4FD35F693DE89F463B42D0A81FB4318"
DRIVER = (
    "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/"
    "run_system2005_masked_historical_branch_v2.py"
)
DRIVER_SHA256 = "63597902DD88DD7F5958A2A922414622CACD1E3600BB2512D8C14F4B92CFF95B"
CONFIG_DIR = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config"
PAIRS = (
    (
        "system2005_masked_historical_preflight_endp0p834281_v1.json",
        "system2005_masked_historical_preflight_endp0p834281_v3_launch.json",
    ),
    (
        "system2005_masked_historical_preflight_endp1p334281_v1.json",
        "system2005_masked_historical_preflight_endp1p334281_v3_launch.json",
    ),
)
OUTPUT = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_system2005_masked_historical_preflight_prelaunch_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def encode(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def main() -> None:
    if sha256(ROUTE) != ROUTE_SHA256 or sha256(ROOT / DRIVER) != DRIVER_SHA256:
        raise RuntimeError("route-or-driver-hash-mismatch")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    state = json.loads((ROOT / "LIA_CURRENT_STATE.json").read_text(encoding="utf-8"))
    if state["state_id"] != "2026-08-29-v10.155-system2005-historical-response-preflight-queued-window-corrected":
        raise RuntimeError("unexpected-controller-state")
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
                "driver": DRIVER,
                "driver_sha256": DRIVER_SHA256,
                "launch_route": str(ROUTE.relative_to(ROOT)).replace("\\", "/"),
                "launch_route_sha256": ROUTE_SHA256,
                "launch_authorized": True,
                "prelaunch_evidence": str((OUTPUT / "summary.json").relative_to(ROOT)).replace(
                    "\\", "/"
                ),
            }
        )
        target.write_bytes(encode(config))
        rows.append(
            {
                "launch_config": str(target.relative_to(ROOT)).replace("\\", "/"),
                "launch_config_sha256": sha256(target),
                "output_root": config["output_root"],
                "temperature_offset_2000_k": config["temperature_offset_2000_k"],
            }
        )
    summary = {
        "schema": "lia-191-system2005-masked-historical-preflight-prelaunch-v2",
        "status": "prelaunch-pass",
        "system_id": 2005,
        "route": str(ROUTE.relative_to(ROOT)).replace("\\", "/"),
        "route_sha256": ROUTE_SHA256,
        "driver": DRIVER,
        "driver_sha256": DRIVER_SHA256,
        "branches": rows,
        "launch_policy": "sequential on one GPU; colder branch first",
        "model_run": False,
    }
    (OUTPUT / "summary.json").write_bytes(encode(summary))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
