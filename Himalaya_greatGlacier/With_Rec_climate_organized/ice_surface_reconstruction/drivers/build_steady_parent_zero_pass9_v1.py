"""Build nine versioned parent-temperature steady-state configs.

This builder is deliberately preparation-only: it performs provenance and
contract checks, writes new configs plus one manifest with exclusive-create
semantics, and never imports or runs OGGM.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


MODULE = (
    "Himalaya_greatGlacier.With_Rec_climate_organized."
    "ice_surface_reconstruction.drivers.synthetic_lia_system_steady_50y_v1"
)
PYTHON = Path(r"D:\Anaconda\envs\oggm_env2\python.exe")
CONFIG_DIR = (
    "Himalaya_greatGlacier/With_Rec_climate_organized/"
    "ice_surface_reconstruction/config/"
    "synthetic_lia_system_standard_inversion_v1"
)
MANIFEST_NAME = "w704steady_parent_zero_pass9_manifest_v1.json"
STAGE = "remaining-route-v3-zero-pass9-parent-steady-50y-v1"

# Fixed by the audited remaining-route parent evidence.  These aliases are
# intentionally explicit so this builder cannot silently select a newer or
# merely similarly named parent chain.
CASES = (
    (1645, "w643zero_conservative_batch35_39_1645_v1", "w581inv1645v1"),
    (1880, "w643zero_conservative_batch35_39_1880_v1", "w580inv1880v1"),
    (3523, "w643zero_conservative_batch35_39_3523_v1", "w514inv3523v1"),
    (3654, "w643zero_conservative_batch35_39_3654_v1", "w577inv3654v1"),
    (3909, "w643zero_conservative_batch35_39_3909_v1", "w529inv3909v1"),
    (4464, "w643zero_conservative_batch35_39_4464_v1", "w576inv4464v1"),
    (4607, "w643zero_conservative_batch35_39_4607_v1", "w535inv4607v1"),
    (5127, "w643zero_conservative_batch35_39_5127_v1", "w552inv5127v1"),
    (5536, "w643zero_conservative_batch35_39_5536_v1", "w513inv5536v1"),
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    cfg_root = root / CONFIG_DIR
    manifest_path = root / MANIFEST_NAME

    destinations: list[Path] = [manifest_path]
    output_roots: list[Path] = []
    for system_id, _, _ in CASES:
        destinations.append(
            cfg_root / f"steady_parent_zero_pass9_{system_id}_v1.json"
        )
        output_roots.append(root / f"w704steady_parent_zero_pass9_{system_id}_v1")

    collisions = [str(path) for path in destinations + output_roots if path.exists()]
    _require(not collisions, "refusing-to-overwrite-existing-paths:" + repr(collisions))
    _require(PYTHON.exists(), f"python-missing:{PYTHON}")

    prepared: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for system_id, zero_name, expected_inv_name in CASES:
        zero_root = root / zero_name
        inv_root = root / expected_inv_name
        zero_formal_path = zero_root / "zero_mb_summary_formal.json"
        zero_cfg_path = zero_root / "config_snapshot.json"
        inv_formal_path = inv_root / "inversion_summary_formal.json"
        inv_cfg_path = inv_root / "config_snapshot.json"
        parents = (
            zero_formal_path,
            zero_cfg_path,
            inv_formal_path,
            inv_cfg_path,
        )
        missing = [str(path) for path in parents if not path.is_file()]
        _require(not missing, f"system-{system_id}-missing-parent-files:{missing}")

        zero_formal = _read_json(zero_formal_path)
        zero_cfg = _read_json(zero_cfg_path)
        inv_formal = _read_json(inv_formal_path)
        inv_cfg = _read_json(inv_cfg_path)

        _require(
            zero_formal.get("status") == "zero-mb-pass",
            f"system-{system_id}-zero-mb-not-pass",
        )
        _require(
            not zero_formal.get("review_flags", []),
            f"system-{system_id}-zero-mb-review-flags",
        )
        _require(
            inv_formal.get("status") == "inversion-pass",
            f"system-{system_id}-inversion-not-pass",
        )
        _require(
            not inv_formal.get("review_flags", []),
            f"system-{system_id}-inversion-review-flags",
        )
        for label, data in (
            ("zero-formal", zero_formal),
            ("zero-config", zero_cfg),
            ("inversion-formal", inv_formal),
            ("inversion-config", inv_cfg),
        ):
            _require(
                int(data.get("system_id", -1)) == system_id,
                f"system-{system_id}-{label}-id-mismatch",
            )

        actual_inv_name = str(zero_cfg["inversion_root"]["sub"])
        _require(
            actual_inv_name == expected_inv_name,
            f"system-{system_id}-inversion-root-mismatch:{actual_inv_name}",
        )
        _require(
            str(zero_cfg["synthetic_rgi_id"]) == str(inv_cfg["synthetic_rgi_id"]),
            f"system-{system_id}-synthetic-rgi-mismatch",
        )
        _require(
            list(inv_cfg["spinup_window"]) == [1441, 1470],
            f"system-{system_id}-spinup-window-not-B",
        )
        stage2 = inv_cfg["stage2"]
        _require(
            stage2.get("prcp_anomaly_mode") == "off",
            f"system-{system_id}-precipitation-anomaly-not-off",
        )
        _require(
            float(stage2["fs"]) == 0.0,
            f"system-{system_id}-fs-not-frozen-zero",
        )
        _require(
            float(zero_cfg["glen_a"]) == float(stage2["glen_a"])
            and float(zero_cfg["fs"]) == float(stage2["fs"]),
            f"system-{system_id}-zero-mb-physics-mismatch",
        )

        output_name = f"w704steady_parent_zero_pass9_{system_id}_v1"
        config_path = (
            cfg_root / f"steady_parent_zero_pass9_{system_id}_v1.json"
        )
        payload = {
            "experiment": "conservative-one-way-tributary-fluxbased-v1",
            "stage": STAGE,
            "system_id": system_id,
            "synthetic_rgi_id": inv_cfg["synthetic_rgi_id"],
            "inversion_root": {"anchor": "project", "sub": expected_inv_name},
            "zero_mb_root": {"anchor": "project", "sub": zero_name},
            "output_root": {"anchor": "project", "sub": output_name},
            "climate_filesuffix": "_batch1_1441_1470_prcp_off",
            "spinup_window": [1441, 1470],
            "registered_lia_length_m": float(inv_cfg["registered_lia_length_m"]),
            "stage2": stage2,
            "gates": {
                "volume_change_fraction_max": 0.02,
                "lia_length_relative_error_max": 0.1,
                "minimum_free_downstream_m": 5000.0,
            },
            "authorization": {"historical": False},
        }
        command = (
            f"& {_ps_quote(str(PYTHON))} -m {MODULE} "
            f"--config {_ps_quote(str(config_path))}"
        )
        evidence = {
            "zero_mb_formal": {
                "path": str(zero_formal_path.relative_to(root)),
                "sha256": _sha256(zero_formal_path),
            },
            "zero_mb_config_snapshot": {
                "path": str(zero_cfg_path.relative_to(root)),
                "sha256": _sha256(zero_cfg_path),
            },
            "inversion_formal": {
                "path": str(inv_formal_path.relative_to(root)),
                "sha256": _sha256(inv_formal_path),
            },
            "inversion_config_snapshot": {
                "path": str(inv_cfg_path.relative_to(root)),
                "sha256": _sha256(inv_cfg_path),
            },
        }
        record = {
            "system_id": system_id,
            "config": str(config_path.relative_to(root)),
            "output_root": output_name,
            "parent_delta_t_c": float(stage2["delta_t_c"]),
            "stage2": stage2,
            "registered_lia_length_m": float(inv_cfg["registered_lia_length_m"]),
            "parent_evidence": evidence,
            "run_command_powershell": command,
        }
        prepared.append((config_path, payload, record))

    cfg_root.mkdir(parents=True, exist_ok=True)
    for config_path, payload, _ in prepared:
        with config_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    # Validate the serialized configs before committing the manifest.
    for config_path, payload, _ in prepared:
        _require(
            _read_json(config_path) == payload,
            f"generated-config-roundtrip-mismatch:{config_path}",
        )
    _require(
        all(not path.exists() for path in output_roots),
        "steady-output-root-created-during-preparation",
    )

    manifest = {
        "status": "configs-built-not-run",
        "stage": STAGE,
        "count": len(prepared),
        "runner_module": MODULE,
        "python": str(PYTHON),
        "dispatch": {
            "maximum_available_workers": 12,
            "policy": "dispatch-gradually-as-workers-become-free",
            "oggm_started_by_builder": False,
        },
        "contract": {
            "spinup_window": [1441, 1470],
            "climate_filesuffix": "_batch1_1441_1470_prcp_off",
            "precipitation_anomaly_mode": "off",
            "volume_change_fraction_max": 0.02,
            "lia_length_relative_error_max": 0.1,
            "minimum_free_downstream_m": 5000.0,
            "historical_authorized": False,
        },
        "static_validation": "pass",
        "cases": [record for _, _, record in prepared],
    }
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    _require(_read_json(manifest_path) == manifest, "manifest-roundtrip-mismatch")

    print(
        json.dumps(
            {
                "status": manifest["status"],
                "count": manifest["count"],
                "manifest": str(manifest_path),
                "static_validation": manifest["static_validation"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
