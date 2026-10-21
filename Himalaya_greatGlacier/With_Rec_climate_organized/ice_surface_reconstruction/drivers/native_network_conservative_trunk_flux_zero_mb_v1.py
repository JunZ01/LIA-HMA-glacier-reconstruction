"""One authorized FluxBasedModel zero-MB test of the unchanged trunk state."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import pickle
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


def hash_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    def safe(item):
        if isinstance(item, dict):
            return {str(k): safe(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(v) for v in item]
        if isinstance(item, np.generic):
            return safe(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe(value), indent=2, allow_nan=False), encoding="utf-8")


def run(config_path: Path, project_root: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = Path(config["trunk_flowline"])
    output = Path(config["output_root"])
    if not source.is_absolute():
        source = project_root / source
    if not output.is_absolute():
        output = project_root / output
    output.mkdir(parents=True, exist_ok=True)
    before_hash = hash_file(source)
    with gzip.open(source, "rb") as stream:
        flowlines = pickle.load(stream)

    from oggm import cfg
    from oggm.core.flowline import FluxBasedModel
    from oggm.core.massbalance import ScalarMassBalance

    cfg.initialize(logging_level="WARNING")
    model = FluxBasedModel(
        copy.deepcopy(flowlines), mb_model=ScalarMassBalance(0), y0=0,
        glen_a=float(config["stage2"]["glen_a"]), fs=float(config["stage2"]["fs"]),
    )
    initial_volume = float(model.volume_m3)
    rows = []
    error = None
    try:
        for year in (0.1, 5.0):
            model.run_until(year)
            fl = model.fls[0]
            section = np.asarray(fl.section, float)
            thick = np.asarray(fl.thick, float)
            rows.append({
                "year": year,
                "volume_m3": float(model.volume_m3),
                "volume_relative_change": float((model.volume_m3 - initial_volume) / initial_volume),
                "area_m2": float(model.area_m2),
                "length_m": float(model.length_m),
                "terminal_thickness_m": float(thick[-1]),
                "minimum_section_m2": float(section.min()),
                "minimum_thickness_m": float(thick.min()),
                "maximum_thickness_m": float(thick.max()),
                "finite": bool(np.isfinite(section).all() and np.isfinite(thick).all()),
            })
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    pd.DataFrame(rows).to_csv(output / "zero_mb_checkpoints.csv", index=False)
    if rows:
        maximum_drift = max(abs(row["volume_relative_change"]) for row in rows)
        maximum_terminal = max(row["terminal_thickness_m"] for row in rows)
        finite_nonnegative = all(
            row["finite"] and row["minimum_section_m2"] >= -1e-10
            and row["minimum_thickness_m"] >= -1e-10 for row in rows
        )
    else:
        maximum_drift = float("inf")
        maximum_terminal = float("inf")
        finite_nonnegative = False
    gates = config["gates"]
    flags = {
        "completed": len(rows) == 2 and rows[-1]["year"] == 5.0,
        "volume_conservation": maximum_drift <= float(gates["volume_drift_max"]),
        "finite_nonnegative": finite_nonnegative,
        "ice_free_boundary": maximum_terminal <= float(gates["terminal_thickness_m_max"]),
        "no_runtime_error": error is None,
        "source_immutable": before_hash == hash_file(source),
    }
    passed = bool(all(flags.values()))
    result = {
        "schema_version": 1,
        "status": "zero-mb-pass" if passed else "zero-mb-fail",
        "experiment": "native-network-conservative-trunk-condensation-v1",
        "variant": "unchanged-static-fluxbased-dynamics",
        "system_id": int(config["system_id"]),
        "solver": "FluxBasedModel",
        "initial_volume_m3": initial_volume,
        "maximum_absolute_volume_drift": maximum_drift,
        "maximum_terminal_thickness_m": maximum_terminal,
        "gate_flags": flags,
        "error": error,
        "source_hash_before": before_hash,
        "source_hash_after": hash_file(source),
        "source_unchanged": before_hash == hash_file(source),
        "steady_run": false,
        "historical_run": false,
        "next_authorized_stage": "steady-50y" if passed else None
    }
    write_json(output / "zero_mb_summary.json", result)
    write_json(output / "config_snapshot.json", config)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.project_root.resolve()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
