"""Per-case hole-count gate wrapper for deterministic topology repair v2."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from ..paths import resolve
from . import repair_invalid_lia_polygon_geometry_v2 as v2


def run(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = resolve(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in config["cases"]:
        local = deepcopy(config)
        local["topology_repair_gates"]["required_explicit_hole_count_change"] = int(
            case["required_explicit_hole_count_change"]
        )
        rows.append(v2._repair_one(local, case, output_root))
    frame = pd.DataFrame(rows).sort_values("system_id").reset_index(drop=True)
    frame.to_csv(output_root / "system_summary.csv", index=False)
    payload = {
        "schema_version": 3,
        "experiment": "deterministic-invalid-lia-polygon-topology-repair-v3-per-case-hole-gate",
        "status": "static-pilot-pass" if frame["status"].eq("geometry-pass").all() else "static-pilot-partial-or-fail",
        "system_count": int(len(frame)),
        "topology_repair_pass_count": int(frame["topology_repair_pass"].sum()),
        "geometry_pass_count": int(frame["status"].eq("geometry-pass").sum()),
        "inversion_run": False,
        "zero_mb_run": False,
        "source_products_mutated": False,
        "systems": rows,
        "config": str(config_path.resolve()),
        "scientific_contract_changed": False,
        "implementation_note": "Only the predeclared per-system explicit-hole delta differs; make_valid and every static gate are unchanged from v2.",
    }
    v2.common._write_json(output_root / "summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
