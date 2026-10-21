"""Run a bounded tradeoff pilot with an optional explicit glen_a multiplier.

This is an additive runner for user-approved follow-up blocks. Historical
tradeoff batches remain on ``lia_191_tradeoff_pilot`` unchanged. The new
runner reuses the verified controller execution and scoring path but allows
each explicit parameter point to carry ``glen_a_stage2_multiplier``.
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import pandas as pd

from .lia_191_controller import (
    _atomic_csv,
    _atomic_json,
    _path_signature,
    _resolve,
    _utc_now,
    controller_root,
    load_config,
    parameter_fingerprint,
)
from .lia_191_tradeoff_pilot import (
    _one_stage2_row,
    _selected_preflight,
    run as run_tasks,
    sync as sync_tasks,
    print_status,
)


def _parameter_specs(stage2: pd.Series, design: dict, system_design: dict) -> list[dict]:
    stage2_melt_f = float(stage2["melt_f"])
    stage2_glen_a = float(stage2["glen_a_pa_3_s_1"])
    default_prcp_multiplier = float(design["prcp_fac_multiplier"])
    default_temp_melt = float(design["temp_melt"])
    explicit = system_design.get("parameter_pairs")
    if explicit is not None:
        specs = []
        for index, item in enumerate(explicit):
            missing = {"delta_t", "melt_f_stage2_multiplier"}.difference(item)
            if missing:
                raise ValueError(
                    f"parameter_pairs[{index}] missing fields: {sorted(missing)}"
                )
            melt_multiplier = float(item["melt_f_stage2_multiplier"])
            glen_multiplier = float(item.get("glen_a_stage2_multiplier", 1.0))
            prcp_multiplier = float(
                item.get("prcp_fac_multiplier", default_prcp_multiplier)
            )
            temp_melt = float(item.get("temp_melt", default_temp_melt))
            specs.append(
                {
                    "delta_t": float(item["delta_t"]),
                    "melt_f": stage2_melt_f * melt_multiplier,
                    "melt_f_stage2_multiplier": melt_multiplier,
                    "prcp_fac_multiplier": prcp_multiplier,
                    "glen_a": stage2_glen_a * glen_multiplier,
                    "glen_a_stage2_multiplier": glen_multiplier,
                    "temp_melt": temp_melt,
                }
            )
        return specs

    delta_t_values = [float(value) for value in system_design["delta_t"]]
    multipliers = [
        float(value) for value in system_design["melt_f_stage2_multipliers"]
    ]
    return [
        {
            "delta_t": delta_t,
            "melt_f": stage2_melt_f * multiplier,
            "melt_f_stage2_multiplier": multiplier,
            "prcp_fac_multiplier": default_prcp_multiplier,
            "glen_a": stage2_glen_a,
            "glen_a_stage2_multiplier": 1.0,
            "temp_melt": default_temp_melt,
        }
        for delta_t, multiplier in itertools.product(delta_t_values, multipliers)
    ]


def build_manifest(config: dict) -> pd.DataFrame:
    root = controller_root(config)
    root.mkdir(parents=True, exist_ok=True)
    preflight = _selected_preflight(config)
    _atomic_csv(root / "preflight_manifest.csv", preflight)
    design = config["pilot_design"]
    block = str(design["block"])
    work_root = _resolve(config, config["execution"]["work_root"])
    existing_path = root / "task_manifest.csv"
    existing = pd.read_csv(existing_path) if existing_path.exists() else pd.DataFrame()
    existing_by_fp = (
        {str(row.fingerprint): row._asdict() for row in existing.itertuples(index=False)}
        if not existing.empty
        else {}
    )

    rows = []
    for item in preflight.itertuples(index=False):
        system_id = int(item.lia_unique_id)
        rgi_id = str(item.rgi_id)
        stage2 = _one_stage2_row(config, system_id, rgi_id)
        for spec in _parameter_specs(stage2, design, design["systems"][str(system_id)]):
            parameters = {
                key: spec[key]
                for key in ("delta_t", "melt_f", "prcp_fac_multiplier", "glen_a", "temp_melt")
            }
            payload = {
                "schema_version": 1,
                "lia_unique_id": system_id,
                "rgi_id": rgi_id,
                "block": block,
                "parameters": parameters,
                "forcing": config["forcing"],
                "workflow": config["workflow"],
                "inputs": {
                    key: _path_signature(_resolve(config, value))
                    for key, value in config["inputs"].items()
                    if key != "source_preflight"
                },
            }
            fingerprint = parameter_fingerprint(payload)
            output_dir = root / "runs" / f"system_{system_id}" / block / fingerprint[:16]
            task_work = work_root / fingerprint[:16]
            if fingerprint in existing_by_fp:
                row = dict(existing_by_fp[fingerprint])
                row["output_dir"] = str(output_dir.resolve())
                row["workdir"] = str(task_work.resolve())
                rows.append(row)
                continue
            rows.append(
                {
                    "task_id": f"{system_id}-{block}-{fingerprint[:16]}",
                    "fingerprint": fingerprint,
                    "lia_unique_id": system_id,
                    "rgi_id": rgi_id,
                    "block": block,
                    **parameters,
                    "melt_f_stage2_multiplier": spec["melt_f_stage2_multiplier"],
                    "glen_a_stage2_multiplier": spec["glen_a_stage2_multiplier"],
                    "task_state": "planned",
                    "output_dir": str(output_dir.resolve()),
                    "workdir": str(task_work.resolve()),
                    "created_at_utc": _utc_now(),
                    "updated_at_utc": _utc_now(),
                    "returncode": math.nan,
                    "error": "",
                }
            )
    manifest = pd.DataFrame(rows).sort_values(
        ["lia_unique_id", "melt_f", "glen_a", "delta_t"]
    )
    _atomic_csv(existing_path, manifest)
    _atomic_json(root / "task_manifest.json", manifest.to_dict(orient="records"))
    _atomic_json(root / "dry_run_summary.json", {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "controller_config": str(config["_config_path"]),
        "project_facts": _path_signature(_resolve(config, config["project_facts"])),
        "purpose": design["purpose"],
        "systems": sorted(int(value) for value in design["systems"]),
        "task_count": int(len(manifest)),
        "task_states": manifest["task_state"].value_counts().to_dict(),
    })
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("sync")
    subparsers.add_parser("status")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "prepare":
        manifest = build_manifest(config)
        print(manifest.groupby("lia_unique_id").size().to_string())
    elif args.command == "run":
        scores = run_tasks(config, args.limit)
        print(f"scored={len(scores)}")
    elif args.command == "sync":
        scores = sync_tasks(config)
        print(f"scored={len(scores)}")
    else:
        print_status(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
