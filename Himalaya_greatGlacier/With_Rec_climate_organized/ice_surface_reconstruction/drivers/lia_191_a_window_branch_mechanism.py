"""Run the bounded A-window state-branch mechanism experiment.

This experiment asks one question: under frozen Stage-2 physics, does moving
the spin-up/forward window from B (1441-1470/1470-2000) to A
(1171-1200/1200-2000) change the stable state branch reached by selected
delta-T anchors?

The driver deliberately requires a named single-chain preflight before the
remaining tasks can be released.  It is a state/window mechanism diagnostic,
not a new parameter search and not authorization for a Core-191 batch.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .build_lia_191_iterative_loop import load_scores
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
from .lia_191_tradeoff_pilot import _one_stage2_row, _selected_preflight


def _close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12)


def _source_score(anchor: dict, system_id: int) -> pd.Series:
    scores = load_scores()
    selected = scores[
        scores["lia_unique_id"].eq(system_id)
        & scores["task_id"].astype(str).eq(str(anchor["source_task_id"]))
    ]
    if len(selected) != 1:
        raise ValueError(
            f"source-anchor-score-count:{system_id}:{anchor['source_task_id']}:{len(selected)}"
        )
    return selected.iloc[0]


def _validate_source_anchor(
    config: dict, system_id: int, rgi_id: str, stage2: pd.Series, anchor: dict
) -> dict:
    score = _source_score(anchor, system_id)
    source_dir = _resolve(config, anchor["source_output_dir"])
    score_dir = Path(str(score["output_dir"])).resolve()
    if source_dir.resolve() != score_dir:
        raise ValueError(
            f"source-anchor-output-conflict:{system_id}:{source_dir}:{score_dir}"
        )
    summary_path = source_dir / "summary.json"
    timeseries_path = source_dir / "timeseries.csv"
    if not summary_path.is_file() or not timeseries_path.is_file():
        raise FileNotFoundError(f"source-anchor-files-missing:{source_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary["lia_unique_id"]) != system_id or str(summary["rgi_id"]) != rgi_id:
        raise ValueError(f"source-anchor-identity-conflict:{system_id}")
    if summary.get("spinup_window") != [1441, 1470] or int(summary["start_year"]) != 1470:
        raise ValueError(f"source-anchor-not-B-window:{system_id}:{anchor['source_task_id']}")
    parameters = summary["parameters"]
    checks = {
        "delta_t": (parameters["delta_t_c"], anchor["delta_t"]),
        "melt_f": (parameters["melt_f"], stage2["melt_f"]),
        "prcp_fac_multiplier": (parameters["prcp_fac_multiplier"], 1.0),
        "glen_a": (parameters["glen_a_pa_3_s_1"], stage2["glen_a_pa_3_s_1"]),
        "temp_melt": (parameters["temp_melt_c"], -1.0),
    }
    conflicts = [name for name, values in checks.items() if not _close(*values)]
    if conflicts:
        raise ValueError(
            f"source-anchor-frozen-parameter-conflict:{system_id}:{','.join(conflicts)}"
        )
    stable = bool(score["stable_chain"])
    expected_stable = bool(anchor["source_expected_stable"])
    if stable != expected_stable:
        raise ValueError(
            f"source-anchor-stability-conflict:{system_id}:{anchor['source_task_id']}:"
            f"expected={expected_stable}:actual={stable}"
        )
    return {
        "source_summary": summary,
        "source_score": score,
        "source_dir": source_dir,
    }


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
    bom_rows: list[dict] = []
    named_preflight_count = 0
    for item in preflight.itertuples(index=False):
        system_id = int(item.lia_unique_id)
        rgi_id = str(item.rgi_id)
        stage2 = _one_stage2_row(config, system_id, rgi_id)
        anchors = design["systems"][str(system_id)]["anchors"]
        if len(anchors) != 2:
            raise ValueError(f"expected-two-anchors:{system_id}:{len(anchors)}")
        for anchor in anchors:
            checked = _validate_source_anchor(config, system_id, rgi_id, stage2, anchor)
            source_score = checked["source_score"]
            named_preflight = bool(anchor.get("named_preflight", False))
            named_preflight_count += int(named_preflight)
            parameters = {
                "delta_t": float(anchor["delta_t"]),
                "melt_f": float(stage2["melt_f"]),
                "prcp_fac_multiplier": 1.0,
                "glen_a": float(stage2["glen_a_pa_3_s_1"]),
                "temp_melt": -1.0,
            }
            payload = {
                "schema_version": 1,
                "lia_unique_id": system_id,
                "rgi_id": rgi_id,
                "block": block,
                "anchor_role": anchor["anchor_role"],
                "source_task_id": anchor["source_task_id"],
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
            base_row = {
                "task_id": f"{system_id}-{block}-{fingerprint[:16]}",
                "fingerprint": fingerprint,
                "lia_unique_id": system_id,
                "rgi_id": rgi_id,
                "block": block,
                "anchor_role": str(anchor["anchor_role"]),
                "named_preflight": named_preflight,
                "source_task_id": str(anchor["source_task_id"]),
                "source_output_dir": str(checked["source_dir"].resolve()),
                "source_expected_stable": bool(anchor["source_expected_stable"]),
                **parameters,
                "melt_f_stage2_multiplier": 1.0,
                "output_dir": str(output_dir.resolve()),
                "workdir": str(task_work.resolve()),
            }
            if fingerprint in existing_by_fingerprint:
                row = {**base_row, **dict(existing_by_fingerprint[fingerprint])}
                row["output_dir"] = str(output_dir.resolve())
                row["workdir"] = str(task_work.resolve())
            else:
                row = {
                    **base_row,
                    "task_state": "planned",
                    "created_at_utc": _utc_now(),
                    "updated_at_utc": _utc_now(),
                    "returncode": math.nan,
                    "error": "",
                }
            rows.append(row)
            bom_rows.append(
                {
                    "lia_unique_id": system_id,
                    "rgi_id": rgi_id,
                    "anchor_role": anchor["anchor_role"],
                    "named_preflight": named_preflight,
                    "delta_t": float(anchor["delta_t"]),
                    "melt_f": parameters["melt_f"],
                    "prcp_fac_multiplier": 1.0,
                    "glen_a": parameters["glen_a"],
                    "temp_melt": -1.0,
                    "source_task_id": anchor["source_task_id"],
                    "source_score_source": source_score["score_source"],
                    "source_summary_status": source_score["summary_status"],
                    "source_stable_chain": bool(source_score["stable_chain"]),
                    "source_spinup_stability_fraction": source_score[
                        "spinup_stability_fraction"
                    ],
                    "source_forward_peak_m": source_score["forward_peak_m"],
                    "source_forward_2000_m": source_score["forward_2000_m"],
                    "lia_target_m": source_score["lia_target_m"],
                    "rgi_target_m": source_score["rgi_target_m"],
                    "A_spinup_window": "1171-1200",
                    "A_forward_window": "1200-2000",
                }
            )
    if named_preflight_count != 1:
        raise ValueError(f"expected-one-named-preflight:{named_preflight_count}")

    manifest = pd.DataFrame(rows).sort_values(
        ["named_preflight", "lia_unique_id", "delta_t"], ascending=[False, True, True]
    )
    _atomic_csv(existing_path, manifest)
    _atomic_json(root / "task_manifest.json", manifest.to_dict(orient="records"))
    _atomic_csv(root / "anchor_bom.csv", pd.DataFrame(bom_rows))
    _atomic_json(
        root / "experiment_contract.json",
        {
            "schema_version": 1,
            "created_at_utc": _utc_now(),
            "controller_config": str(config["_config_path"]),
            "project_facts": _path_signature(_resolve(config, config["project_facts"])),
            "purpose": design["purpose"],
            "interpretation_limit": design["interpretation_limit"],
            "task_count": int(len(manifest)),
            "named_preflight_task_id": str(
                manifest.loc[manifest["named_preflight"].astype(bool), "task_id"].iloc[0]
            ),
            "release_gate": {
                "technical_clean": True,
                "spinup_converged": True,
                "spinup_stability_fraction_max": 0.02,
                "spinup_window": [1171, 1200],
                "forward_window": [1200, 2000],
                "precipitation_anomalies": "off",
            },
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
        updated_rows.append(task)
    updated = pd.DataFrame(updated_rows)
    scores = pd.DataFrame(score_rows)
    _atomic_csv(root / "task_manifest.csv", updated)
    _atomic_csv(root / "scores.csv", scores)
    return scores


def _preflight_gate(config: dict) -> dict:
    root = controller_root(config)
    manifest = pd.read_csv(root / "task_manifest.csv")
    selected = manifest[manifest["named_preflight"].astype(str).str.lower().eq("true")]
    if len(selected) != 1:
        raise ValueError(f"named-preflight-task-count:{len(selected)}")
    task = selected.iloc[0]
    output_dir = Path(task["output_dir"])
    summary_path = output_dir / "summary.json"
    timeseries_path = output_dir / "timeseries.csv"
    reasons: list[str] = []
    if not summary_path.is_file() or not timeseries_path.is_file():
        reasons.append("missing-summary-or-timeseries")
        summary = {}
        series = pd.DataFrame()
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        series = pd.read_csv(timeseries_path)
        if summary.get("status") != "ok":
            reasons.append(f"summary-status:{summary.get('status')}")
        if summary.get("spinup_window") != [1171, 1200]:
            reasons.append("wrong-spinup-window")
        if int(summary.get("start_year", -1)) != 1200:
            reasons.append("wrong-forward-start")
        control = summary.get("spinup_control", {})
        if control.get("mode") != "adaptive":
            reasons.append("spinup-not-adaptive")
        if not bool(control.get("converged", False)):
            reasons.append("spinup-not-converged")
        stability = float(control.get("final_stability_fraction", math.nan))
        if not math.isfinite(stability) or stability > 0.02:
            reasons.append(f"spinup-stability:{stability}")
        if not series["phase"].astype(str).eq("forward").any():
            reasons.append("missing-forward-phase")
        elif not series.loc[series["phase"].astype(str).eq("forward"), "year"].astype(int).eq(2000).any():
            reasons.append("missing-year-2000")
        parameters = summary.get("parameters", {})
        for key, expected in (
            ("delta_t_c", task["delta_t"]),
            ("melt_f", task["melt_f"]),
            ("prcp_fac_multiplier", 1.0),
            ("glen_a_pa_3_s_1", task["glen_a"]),
            ("temp_melt_c", -1.0),
        ):
            if key not in parameters or not _close(parameters[key], expected):
                reasons.append(f"parameter-conflict:{key}")
        climate = summary.get("climate", {})
        for phase in ("spinup", "forward"):
            if "prcp_off" not in str(climate.get(phase, {}).get("source", "")):
                reasons.append(f"precipitation-mode-conflict:{phase}")
    result = {
        "checked_at_utc": _utc_now(),
        "task_id": str(task["task_id"]),
        "passed": not reasons,
        "reasons": reasons,
        "summary_status": summary.get("status"),
        "spinup_actual_years": summary.get("spinup_control", {}).get("actual_years"),
        "spinup_stability_fraction": summary.get("spinup_control", {}).get(
            "final_stability_fraction"
        ),
        "release_remaining_five": not reasons,
    }
    _atomic_json(root / "preflight_gate.json", result)
    return result


def run_selected(config: dict, mode: str) -> pd.DataFrame:
    if not config["execution"].get("allow_runs", False):
        raise RuntimeError("execution-disabled-in-controller-config")
    workers = int(config["execution"]["max_workers"])
    if workers > 12:
        raise ValueError("max-workers-exceeds-verified-limit-12")
    root = controller_root(config)
    manifest = pd.read_csv(root / "task_manifest.csv")
    planned = manifest[manifest["task_state"].eq("planned")].copy()
    named = planned["named_preflight"].astype(str).str.lower().eq("true")
    if mode == "preflight":
        planned = planned[named]
    elif mode == "remaining":
        gate_path = root / "preflight_gate.json"
        if not gate_path.is_file():
            raise RuntimeError("preflight-gate-missing")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if not bool(gate.get("passed", False)):
            raise RuntimeError(f"preflight-gate-failed:{gate.get('reasons')}")
        planned = planned[~named]
    else:
        raise ValueError(f"unsupported-run-mode:{mode}")
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
    updates = {row["task_id"]: row for row in results}
    updated = pd.DataFrame(
        [updates.get(row["task_id"], row) for row in manifest.to_dict(orient="records")]
    )
    _atomic_csv(root / "task_manifest.csv", updated)
    scores = sync(config)
    if mode == "preflight":
        print(json.dumps(_preflight_gate(config), indent=2, ensure_ascii=False))
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--mode", choices=("preflight", "remaining"), required=True)
    subparsers.add_parser("sync")
    subparsers.add_parser("gate")
    subparsers.add_parser("status")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "prepare":
        manifest = build_manifest(config)
        print(manifest[["lia_unique_id", "anchor_role", "delta_t", "named_preflight", "task_state"]].to_string(index=False))
    elif args.command == "run":
        scores = run_selected(config, args.mode)
        print(f"scored={len(scores)}")
    elif args.command == "sync":
        scores = sync(config)
        print(f"scored={len(scores)}")
    elif args.command == "gate":
        print(json.dumps(_preflight_gate(config), indent=2, ensure_ascii=False))
    else:
        root = controller_root(config)
        manifest = pd.read_csv(root / "task_manifest.csv")
        print(manifest["task_state"].value_counts(dropna=False).to_string())
        gate_path = root / "preflight_gate.json"
        if gate_path.exists():
            print("\n" + gate_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
