"""Build zero-MB configs for newly formal-passing batch-35--39 inversions."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOTS = [
    "w199inv2546dtp025v1", "w200inv7dt0v1",
    "w507inv3897v1", "w508inv3632v1", "w509inv3297v1", "w510inv3495v1",
    "w511inv2378v1", "w512inv5143v1", "w513inv5536v1", "w514inv3523v1",
    "w515inv3082v1", "w529inv3909v1", "w530inv5146v1", "w531inv3808v1",
    "w532inv660v1", "w533inv3010v1", "w535inv4607v1", "w536inv5062v1",
    "w537inv4492v1", "w552inv5127v1", "w553inv4369v1", "w554inv3912v1",
    "w555inv2793v1", "w556inv801v1", "w557inv2619v1", "w558inv1326v1",
    "w559inv709v1", "w560inv2541v1", "w561inv520v1", "w576inv4464v1",
    "w577inv3654v1", "w578inv3244v1", "w579inv902v1", "w580inv1880v1",
    "w581inv1645v1", "w582inv2611v1", "w583inv1743v1", "w586inv2689v1",
    "w587inv553v1", "w612inv1871v1", "w613inv3906v2", "w614inv3907v2",
    "w615inv2727v2", "w616inv558v2", "w617inv2538v2", "w618inv2141v2",
    "w619inv3185v2", "w620inv2603v2", "w621inv2177v2", "w622inv1642v2",
    "w623inv1871v2", "w626inv551_recovery_v1",
]


def order_key(name: str) -> int:
    match = re.match(r"w(\d+)", name)
    return int(match.group(1)) if match else -1


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    accepted = pd.read_csv(
        root / "validation_data/results/paleo_surface_validation_v0/lia_191_accepted_35_curve_gallery_v1/accepted_35_curve_metrics.csv"
    )
    accepted_ids = set(pd.to_numeric(accepted["system_id"]).astype(int))
    chosen: dict[int, tuple[str, dict]] = {}
    for name in ROOTS:
        formal_path = root / name / "inversion_summary_formal.json"
        if not formal_path.exists():
            continue
        formal = json.loads(formal_path.read_text(encoding="utf-8"))
        if formal.get("status") != "inversion-pass":
            continue
        system_id = int(formal["system_id"])
        if system_id in accepted_ids:
            continue
        current = chosen.get(system_id)
        if current is None or order_key(name) > order_key(current[0]):
            chosen[system_id] = (name, formal)

    config_root = root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/synthetic_lia_system_standard_inversion_v1"
    written = []
    cases = []
    for system_id, (name, formal) in sorted(chosen.items()):
        snapshot = json.loads((root / name / "config_snapshot.json").read_text(encoding="utf-8"))
        stage2 = snapshot["stage2"]
        output_name = f"w627zero_batch35_39_{system_id}_v1"
        payload = {
            "experiment": "synthetic-lia-system-zero-mb-v2",
            "stage": "batch-35-39-formal-zero-mb",
            "system_id": system_id,
            "synthetic_rgi_id": snapshot["synthetic_rgi_id"],
            "inversion_root": {"anchor": "project", "sub": name},
            "output_root": {"anchor": "project", "sub": output_name},
            "glen_a": float(stage2["glen_a"]),
            "fs": float(stage2["fs"]),
            "gates": {
                "volume_drift_fraction_max": 0.001,
                "root_terminal_thickness_m_max": 10.0,
                "mass_creation_relative_fraction": 1e-12,
                "mass_creation_absolute_m3": 1e-6,
            },
        }
        path = config_root / f"zero_mb_batch35_39_{system_id}_v1.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(str(path))
        cases.append({"system_id": system_id, "inversion_root": name, "output_root": output_name, "config": str(path)})
    manifest = root / "w627zero_batch35_39_manifest_v1.json"
    manifest.write_text(json.dumps({"count": len(cases), "cases": cases}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(cases), "manifest": str(manifest), "written": written}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
