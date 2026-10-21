"""Run a bounded Layer-3 single-parameter-family pilot.

This reuses the verified Layer-2 controller execution and scoring path while
varying exactly one family (glen_a or prcp_fac) together with a small delta-T
recenter.  It does not enable precipitation anomalies.
"""

from __future__ import annotations

import argparse
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
        raise ValueError(f"system {system_id} expected one Stage-2 row, found {len(selected)}")
    return selected.iloc[0]


def _selected_preflight(config: dict) -> pd.DataFrame:
    path = _resolve(config, config["inputs"]["source_preflight"])
    preflight = pd.read_csv(path)
    system_ids = {int(value) for value in config["pilot_design"]["systems"]}
    selected = preflight[preflight["lia_unique_id"].astype(int).isin(system_ids)].copy()
    if set(selected["lia_unique_id"].astype(int)) != system_ids:
        raise ValueError("Layer-3 systems missing from source preflight")
    failed = selected[~selected["preflight_state"].eq("ready")]
    if not failed.empty:
        raise RuntimeError(f"Layer-3 preflight not ready:\n{failed.to_string(index=False)}")
    return selected.sort_values("lia_unique_id")


def _parameter_specs(stage2: pd.Series, design: dict, system_design: dict) -> list[dict]:
    family = str(design["parameter_family"])
    base_melt = float(stage2["melt_f"])
    base_glen = float(stage2["glen_a_pa_3_s_1"])
    base_prcp = float(stage2["prcp_fac"])
    specs = []
    for dt in system_design["delta_t"]:
        for value in system_design["family_values"]:
            params = {
                "delta_t": float(dt),
                "melt_f": base_melt,
                "prcp_fac_multiplier": 1.0,
                "glen_a": base_glen,
                "temp_melt": float(design.get("temp_melt", -1.0)),
            }
            if family == "glen_a":
                params["glen_a"] = base_glen * float(value)
            elif family == "prcp_fac":
                params["prcp_fac_multiplier"] = float(value)
            else:
                raise ValueError(f"unsupported Layer-3 family: {family}")
            specs.append({**params, "family_value": float(value)})
    return specs


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
    existing_by_fp = {str(row.fingerprint): row._asdict() for row in existing.itertuples(index=False)} if not existing.empty else {}
    rows = []
    for item in preflight.itertuples(index=False):
        sid, rgi = int(item.lia_unique_id), str(item.rgi_id)
        stage2 = _one_stage2_row(config, sid, rgi)
        for spec in _parameter_specs(stage2, design, design["systems"][str(sid)]):
            payload = {
                "schema_version": 1,
                "lia_unique_id": sid,
                "rgi_id": rgi,
                "block": block,
                "parameters": {k: spec[k] for k in ("delta_t", "melt_f", "prcp_fac_multiplier", "glen_a", "temp_melt")},
                "forcing": config["forcing"],
                "workflow": config["workflow"],
                "inputs": {k: _path_signature(_resolve(config, v)) for k, v in config["inputs"].items() if k != "source_preflight"},
            }
            fp = parameter_fingerprint(payload)
            output_dir = root / "runs" / f"system_{sid}" / block / fp[:16]
            task_work = work_root / fp[:16]
            if fp in existing_by_fp:
                row = dict(existing_by_fp[fp]); row["output_dir"] = str(output_dir.resolve()); row["workdir"] = str(task_work.resolve()); rows.append(row); continue
            rows.append({
                "task_id": f"{sid}-{block}-{fp[:16]}",
                "fingerprint": fp,
                "lia_unique_id": sid,
                "rgi_id": rgi,
                "block": block,
                **{k: spec[k] for k in ("delta_t", "melt_f", "prcp_fac_multiplier", "glen_a", "temp_melt")},
                "family_value": spec["family_value"],
                "task_state": "planned",
                "output_dir": str(output_dir.resolve()),
                "workdir": str(task_work.resolve()),
                "created_at_utc": _utc_now(),
                "updated_at_utc": _utc_now(),
                "returncode": math.nan,
                "error": "",
            })
    manifest = pd.DataFrame(rows).sort_values(["lia_unique_id", "family_value", "delta_t"])
    _atomic_csv(existing_path, manifest)
    _atomic_json(root / "task_manifest.json", manifest.to_dict(orient="records"))
    _atomic_json(root / "experiment_contract.json", {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "parameter_family": design["parameter_family"],
        "purpose": design["purpose"],
        "task_count": int(len(manifest)),
        "systems": sorted(int(v) for v in design["systems"]),
        "project_facts": _path_signature(_resolve(config, config["project_facts"])),
        "precipitation_anomaly_mode": config["forcing"]["precipitation_anomaly_mode"],
        "max_workers": int(config["execution"]["max_workers"]),
    })
    return manifest


def sync(config: dict) -> pd.DataFrame:
    root = controller_root(config)
    manifest = pd.read_csv(root / "task_manifest.csv")
    updated, scores = [], []
    for task in manifest.to_dict(orient="records"):
        out = Path(task["output_dir"])
        summary_path, ts_path, result_path = out / "summary.json", out / "timeseries.csv", out / "task_result.json"
        if summary_path.exists() and ts_path.exists():
            task.update({"task_state": "completed", "returncode": 0, "error": ""})
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            ts = pd.read_csv(ts_path)
            scores.append(score_saved_task(config, task, summary, ts))
        elif result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8")); task.update(result)
            scores.append({**task, "technical_failure": True, "accepted": False, "acceptance_state": "technical-failure", "objective_sep": math.nan, "e_lia_sep": math.nan, "e_rgi_sep": math.nan, "spinup_stability_fraction": math.nan})
        updated.append(task)
    updated_df, scores_df = pd.DataFrame(updated), pd.DataFrame(scores)
    _atomic_csv(root / "task_manifest.csv", updated_df); _atomic_csv(root / "scores.csv", scores_df)
    return scores_df


def run(config: dict, limit: int | None) -> pd.DataFrame:
    if not config["execution"].get("allow_runs", False): raise RuntimeError("execution-disabled-in-controller-config")
    workers = int(config["execution"]["max_workers"])
    if workers > 12: raise ValueError("max-workers-exceeds-verified-limit-12")
    root = controller_root(config); manifest = pd.read_csv(root / "task_manifest.csv")
    planned = manifest[manifest["task_state"].eq("planned")]
    if limit is not None: planned = planned.head(limit)
    if planned.empty: return sync(config)
    results = []
    with ThreadPoolExecutor(max_workers=min(workers, len(planned))) as pool:
        futures = {pool.submit(_run_one, config, row._asdict()): row.task_id for row in planned.itertuples(index=False)}
        for idx, fut in enumerate(as_completed(futures), start=1):
            result = fut.result(); results.append(result); print(f"completed {idx}/{len(futures)} task={result['task_id']} state={result['task_state']}", flush=True)
    by_task = {r["task_id"]: r for r in results}
    updated = pd.DataFrame([by_task.get(row["task_id"], row) for row in manifest.to_dict(orient="records")])
    _atomic_csv(root / "task_manifest.csv", updated)
    return sync(config)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True, type=Path)
    sub = parser.add_subparsers(dest="command", required=True); sub.add_parser("prepare"); rp = sub.add_parser("run"); rp.add_argument("--limit", type=int, default=None); sub.add_parser("sync")
    args = parser.parse_args(); config = load_config(args.config)
    if args.command == "prepare": print(build_manifest(config).groupby("lia_unique_id").size().to_string())
    elif args.command == "run": print(f"scored={len(run(config, args.limit))}")
    else: print(f"scored={len(sync(config))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
