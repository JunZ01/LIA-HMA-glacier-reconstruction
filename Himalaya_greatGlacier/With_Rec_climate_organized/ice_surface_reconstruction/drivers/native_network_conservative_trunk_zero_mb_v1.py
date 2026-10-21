"""0.1/5-year zero-MB gate for the passing conservative trunk static state."""

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


def _write_json(path: Path, payload: dict) -> None:
    def safe(value):
        if isinstance(value, dict):
            return {str(k): safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(v) for v in value]
        if isinstance(value, np.generic):
            return safe(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe(payload), indent=2, allow_nan=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def run(config_path: Path, project_root: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = _resolve(project_root, config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    static_summary_path = _resolve(project_root, config["static_summary"])
    trunk_path = _resolve(project_root, config["trunk_flowline"])
    static_summary = json.loads(static_summary_path.read_text(encoding="utf-8"))
    if static_summary.get("status") != "static-pass":
        raise RuntimeError("zero-mb-not-authorized-static-did-not-pass")
    before = {_path: _sha256(_path) for _path in (static_summary_path, trunk_path)}
    with gzip.open(trunk_path, "rb") as stream:
        flowlines = pickle.load(stream)

    from oggm import cfg
    from oggm.core.flowline import SemiImplicitModel
    from oggm.core.massbalance import ScalarMassBalance

    cfg.initialize(logging_level="WARNING")
    model = SemiImplicitModel(
        copy.deepcopy(flowlines),
        mb_model=ScalarMassBalance(0.0),
        y0=0.0,
        glen_a=float(config["stage2"]["glen_a"]),
        fs=float(config["stage2"]["fs"]),
    )
    initial_volume = float(model.volume_m3)
    rows = []
    status = "zero-mb-pass"
    error = None
    try:
        for year in (0.1, 5.0):
            model.run_until(year)
            fl = model.fls[0]
            thick = np.asarray(fl.thick, float)
            rows.append({
                "year": year,
                "volume_m3": float(model.volume_m3),
                "volume_relative_change": float((model.volume_m3 - initial_volume) / initial_volume),
                "area_m2": float(model.area_m2),
                "length_m": float(model.length_m),
                "terminal_thickness_m": float(thick[-1]),
                "maximum_thickness_m": float(np.max(thick)),
                "finite": bool(np.isfinite(thick).all()),
                "negative_thickness_count": int(np.sum(thick < -1e-10)),
            })
    except Exception as exc:
        status = "zero-mb-fail"
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    table = pd.DataFrame(rows)
    table.to_csv(output_root / "zero_mb_checkpoints.csv", index=False)
    if rows:
        max_drift = float(max(abs(row["volume_relative_change"]) for row in rows))
        max_terminal = float(max(row["terminal_thickness_m"] for row in rows))
        numerical_clean = bool(all(row["finite"] and row["negative_thickness_count"] == 0 for row in rows))
    else:
        max_drift = float("inf")
        max_terminal = float("inf")
        numerical_clean = False
    gates = config["gates"]
    gate_flags = {
        "completed_0p1_and_5y": len(rows) == 2 and rows[-1]["year"] == 5.0,
        "volume_conservation": max_drift <= float(gates["volume_drift_max"]),
        "finite_nonnegative": numerical_clean,
        "ice_free_boundary": max_terminal <= float(gates["terminal_thickness_m_max"]),
        "no_runtime_error": error is None,
    }
    after = {_path: _sha256(_path) for _path in (static_summary_path, trunk_path)}
    gate_flags["source_immutable"] = before == after
    passed = bool(all(gate_flags.values()))
    if not passed:
        status = "zero-mb-fail"
    summary = {
        "schema_version": 1,
        "status": status,
        "experiment": "native-network-conservative-trunk-condensation-v1",
        "stage": "zero-mb",
        "system_id": int(config["system_id"]),
        "solver": "SemiImplicitModel",
        "mass_balance": "ScalarMassBalance(0.0)",
        "initial_volume_m3": initial_volume,
        "maximum_absolute_volume_drift": max_drift,
        "maximum_terminal_thickness_m": max_terminal,
        "gate_flags": gate_flags,
        "error": error,
        "source_hashes_before": {str(k): v for k, v in before.items()},
        "source_hashes_after": {str(k): v for k, v in after.items()},
        "source_unchanged": before == after,
        "steady_50y_run": False,
        "historical_run": False,
        "next_authorized_stage": "steady-50y" if passed else None,
    }
    _write_json(output_root / "zero_mb_summary.json", summary)
    _write_json(output_root / "config_snapshot.json", config)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run(args.config.resolve(), args.project_root.resolve())
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
