"""Run the bounded two-system melt_f-delta_t tradeoff pilot.

The pilot reuses the verified Core-191 preflight and the controller's task
execution/scoring path.  It is deliberately separate from the completed
186-system manifest so the initial 1860-chain provenance cannot be replaced.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .lia_191_controller import (
    _atomic_csv,
    _atomic_json,
    _path_signature,
    _resolve,
    _run_one,
    _utc_now,
    controller_root,
    load_config,
    parameter_fingerprint,
    score_saved_task,
)


def _one_stage2_row(config: dict, system_id: int, rgi_id: str) -> pd.Series:
    stage2 = pd.read_csv(_resolve(config, config["inputs"]["stage2_members"]))
    selected = stage2[
        stage2["lia_unique_id"].astype(int).eq(system_id)
        & stage2["rgi_id"].astype(str).eq(rgi_id)
        & stage2["ensemble_role"].astype(str).eq(config["forcing"]["ensemble_role"])
    ]
    if len(selected) != 1:
        raise ValueError(
            f"system {system_id} expected one Stage-2 row, found {len(selected)}"
        )
    return selected.iloc[0]


def _selected_preflight(config: dict) -> pd.DataFrame:
    path = _resolve(config, config["inputs"]["source_preflight"])
    preflight = pd.read_csv(path)
    system_ids = {int(value) for value in config["pilot_design"]["systems"]}
    selected = preflight[preflight["lia_unique_id"].astype(int).isin(system_ids)].copy()
    if set(selected["lia_unique_id"].astype(int)) != system_ids:
        missing = sorted(system_ids.difference(selected["lia_unique_id"].astype(int)))
        raise ValueError(f"pilot systems missing from source preflight: {missing}")
    failed = selected[~selected["preflight_state"].eq("ready")]
    if not failed.empty:
        details = failed[["lia_unique_id", "preflight_state", "preflight_reasons"]]
        raise RuntimeError(f"pilot preflight is not ready:\n{details.to_string(index=False)}")
    return selected.sort_values("lia_unique_id")


def _parameter_pairs(stage2: pd.Series, system_design: dict) -> list[tuple[float, float, float]]:
    """Return delta_t, absolute melt_f, and Stage-2 melt_f multiplier tuples."""
    stage2_melt_f = float(stage2["melt_f"])
    explicit = system_design.get("parameter_pairs")
    if explicit is not None:
        pairs: list[tuple[float, float, float]] = []
        for index, item in enumerate(explicit):
            missing = {"delta_t", "melt_f_stage2_multiplier"}.difference(item)
            if missing:
                raise ValueError(
                    f"parameter_pairs[{index}] missing fields: {sorted(missing)}"
                )
            multiplier = float(item["melt_f_stage2_multiplier"])
            pairs.append(
                (float(item["delta_t"]), stage2_melt_f * multiplier, multiplier)
            )
        return pairs

    delta_t_values = [float(value) for value in system_design["delta_t"]]
    multipliers = [
        float(value) for value in system_design["melt_f_stage2_multipliers"]
    ]
    return [
        (delta_t, stage2_melt_f * multiplier, multiplier)
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
    existing_by_fingerprint = (
        {str(row.fingerprint): row._asdict() for row in existing.itertuples(index=False)}
        if not existing.empty
        else {}
    )

    rows: list[dict] = []
    for item in preflight.itertuples(index=False):
        system_id = int(item.lia_unique_id)
        rgi_id = str(item.rgi_id)
        stage2 = _one_stage2_row(config, system_id, rgi_id)
        system_design = design["systems"][str(system_id)]
        for delta_t, melt_f, melt_f_multiplier in _parameter_pairs(
            stage2, system_design
        ):
            parameters = {
                "delta_t": delta_t,
                "melt_f": melt_f,
                "prcp_fac_multiplier": float(design["prcp_fac_multiplier"]),
                "glen_a": float(stage2["glen_a_pa_3_s_1"]),
                "temp_melt": float(design["temp_melt"]),
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
            if fingerprint in existing_by_fingerprint:
                row = dict(existing_by_fingerprint[fingerprint])
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
                    "melt_f_stage2_multiplier": melt_f_multiplier,
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
        ["lia_unique_id", "melt_f", "delta_t"]
    )
    _atomic_csv(existing_path, manifest)
    _atomic_json(root / "task_manifest.json", manifest.to_dict(orient="records"))
    _atomic_json(
        root / "dry_run_summary.json",
        {
            "schema_version": 1,
            "created_at_utc": _utc_now(),
            "controller_config": str(config["_config_path"]),
            "project_facts": _path_signature(
                _resolve(config, config["project_facts"])
            ),
            "purpose": design["purpose"],
            "systems": sorted(int(value) for value in design["systems"]),
            "task_count": int(len(manifest)),
            "task_states": manifest["task_state"].value_counts().to_dict(),
        },
    )
    return manifest


def sync(config: dict) -> pd.DataFrame:
    root = controller_root(config)
    manifest = pd.read_csv(root / "task_manifest.csv")
    updated_rows: list[dict] = []
    score_rows: list[dict] = []
    for task in manifest.to_dict(orient="records"):
        output_dir = Path(task["output_dir"])
        summary_path = output_dir / "summary.json"
        timeseries_path = output_dir / "timeseries.csv"
        result_path = output_dir / "task_result.json"
        if summary_path.exists() and timeseries_path.exists():
            task["task_state"] = "completed"
            task["returncode"] = 0
            task["error"] = ""
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            timeseries = pd.read_csv(timeseries_path)
            score_rows.append(score_saved_task(config, task, summary, timeseries))
        elif result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            task.update(result)
            score_rows.append(
                {
                    **task,
                    "technical_failure": True,
                    "accepted": False,
                    "acceptance_state": "technical-failure",
                    "objective_sep": math.nan,
                    "e_lia_sep": math.nan,
                    "e_rgi_sep": math.nan,
                    "spinup_stability_fraction": math.nan,
                }
            )
        updated_rows.append(task)
    updated = pd.DataFrame(updated_rows)
    scores = pd.DataFrame(score_rows)
    _atomic_csv(root / "task_manifest.csv", updated)
    _atomic_csv(root / "scores.csv", scores)
    return scores


def run(config: dict, limit: int | None) -> pd.DataFrame:
    if not config["execution"].get("allow_runs", False):
        raise RuntimeError("execution-disabled-in-controller-config")
    workers = int(config["execution"]["max_workers"])
    if workers > 12:
        raise ValueError("max-workers-exceeds-verified-limit-12")
    root = controller_root(config)
    manifest = pd.read_csv(root / "task_manifest.csv")
    planned = manifest[manifest["task_state"].eq("planned")]
    if limit is not None:
        planned = planned.head(limit)
    if planned.empty:
        return sync(config)
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(planned))) as pool:
        futures = {
            pool.submit(_run_one, config, row._asdict()): row.task_id
            for row in planned.itertuples(index=False)
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"completed {index}/{len(futures)} task={result['task_id']} "
                f"state={result['task_state']}",
                flush=True,
            )
    by_task = {row["task_id"]: row for row in results}
    updated = pd.DataFrame(
        [by_task.get(row["task_id"], row) for row in manifest.to_dict(orient="records")]
    )
    _atomic_csv(root / "task_manifest.csv", updated)
    return sync(config)


def print_status(config: dict) -> None:
    root = controller_root(config)
    manifest = pd.read_csv(root / "task_manifest.csv")
    print(manifest["task_state"].value_counts(dropna=False).to_string())
    score_path = root / "scores.csv"
    if score_path.exists():
        scores = pd.read_csv(score_path)
        if not scores.empty and "accepted" in scores:
            print("\nAccepted")
            print(scores["accepted"].value_counts(dropna=False).to_string())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--limit", type=int, default=None)
    subparsers.add_parser("sync")
    subparsers.add_parser("status")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "prepare":
        manifest = build_manifest(config)
        print(manifest.groupby("lia_unique_id").size().to_string())
    elif args.command == "run":
        scores = run(config, args.limit)
        print(f"scored={len(scores)}")
    elif args.command == "sync":
        scores = sync(config)
        print(f"scored={len(scores)}")
    else:
        print_status(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
