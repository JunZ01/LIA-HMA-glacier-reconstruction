"""Run the representative saved-state split-delta-T class experiment.

The batch first reproduces each source forward curve at its original delta-T.
Only when all four baseline reproductions pass are the directed forward-only
temperature offsets released.  Spin-up states and all physical parameters stay
frozen.  The script is resumable and uses at most 12 isolated subprocesses.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


MODULE = (
    "Himalaya_greatGlacier.With_Rec_climate_organized."
    "ice_surface_reconstruction.drivers.lia_forward_delta_from_saved_spinup"
)


REPRESENTATIVES = [
    {
        "lia_unique_id": 1524,
        "class": "opposite_sign_near_gate",
        "source": "validation_data/results/paleo_surface_validation_v0/lia_191_class_C2_glen_pilot_v1/runs/system_1524/B_cls_C2_ga_v1/557fe30a47b0fe33",
        "forward_delta_t": [0.0, 0.025, 0.05, 0.075, 0.10],
    },
    {
        "lia_unique_id": 2569,
        "class": "both_short_near_gate",
        "source": "validation_data/results/paleo_surface_validation_v0/lia_191_analog_melt_delta_after_991_v1/runs/system_2569/B_analog_melt_delta_after_991_v1/fe736989cb98953b",
        "forward_delta_t": [-0.5, -0.525, -0.55, -0.575, -0.60],
    },
    {
        "lia_unique_id": 2622,
        "class": "both_short_near_gate",
        "source": "validation_data/results/paleo_surface_validation_v0/lia_191_analog_melt_delta_after_991_v1/runs/system_2622/B_analog_melt_delta_after_991_v1/6ea2db622269ea83",
        "forward_delta_t": [0.0, -0.025, -0.05, -0.075, -0.10],
    },
    {
        "lia_unique_id": 843,
        "class": "both_long_near_gate",
        "source": "validation_data/results/paleo_surface_validation_v0/lia_191_class_C1_melt_delta_spatial_pilot_v2/runs/system_843/B_cls_C1_mfdt_sp_v2/01ee1be7330d1aa0",
        "forward_delta_t": [0.25, 0.275, 0.30, 0.325, 0.35],
    },
]


def _tag(value: float) -> str:
    return f"{value:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")


def _task_option(task: dict, name: str) -> str | None:
    command = [str(value) for value in task.get("command", [])]
    try:
        index = command.index(name)
    except ValueError:
        return None
    return command[index + 1] if index + 1 < len(command) else None


def _source_task_path(source: Path) -> Path:
    for name in ("task.json", "source_task.json"):
        candidate = source / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing-source-task-metadata:{source}")


def _audit_path(source: Path, summary: dict, task: dict) -> Path:
    domain = summary.get("observation_domain") or {}
    if domain.get("source"):
        return Path(domain["source"])
    root = _task_option(task, "--elevation-band-root")
    if root is None:
        raise ValueError(f"missing-elevation-band-root:{source}")
    return Path(root) / str(summary["lia_unique_id"]) / "elevation_band_audit.json"


def _write_geometry_contract(project_root: Path, output_root: Path) -> pd.DataFrame:
    references = pd.read_csv(
        project_root
        / "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/core191_length_references.csv"
    )
    branches = pd.read_csv(
        project_root
        / "validation_data/results/paleo_surface_validation_v0/lia_design_b_targets_and_inputs_v1/lia_design_b_branch_targets.csv"
    )
    rows = []
    for item in REPRESENTATIVES:
        source = project_root / item["source"]
        summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
        task = json.loads(_source_task_path(source).read_text(encoding="utf-8"))
        system_id = int(item["lia_unique_id"])
        reference = references[references["lia_unique_id"].astype(int).eq(system_id)]
        if len(reference) != 1:
            raise ValueError(f"reference-row-count:{system_id}:{len(reference)}")
        reference = reference.iloc[0]
        audit_path = _audit_path(source, summary, task)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        target = float(audit["target_main_flowline_length_m"])
        initial = float(audit["pseudo_flowline_length_m"])
        mismatch = abs(initial - target) / target
        main = branches[
            branches["lia_unique_id"].astype(int).eq(system_id)
            & branches["is_main_branch"].astype(str).str.lower().eq("true")
        ]
        main_indices = ";".join(
            f"component_{int(row.component_index)}:branch_{int(row.branch_index)}"
            for row in main.itertuples(index=False)
        )
        rows.append(
            {
                "lia_unique_id": system_id,
                "rgi_id": str(reference["rgi_id"]),
                "representative_class": item["class"],
                "lia_main_branch_indices": main_indices,
                "lia_spatial_target_m": float(reference["lia_registered_length_m"]),
                "rgi_root_index": int(reference["modern_root_index"]),
                "rgi_root_count": int(reference["modern_root_count"]),
                "rgi_spatial_target_m": float(reference["rgi_registered_length_m"]),
                "elevation_band_target_m": target,
                "elevation_band_initial_ice_length_m": initial,
                "node_spacing_m": float(audit["node_dx_m"]),
                "initial_target_mismatch_fraction": mismatch,
                "observation_domain_node_count": int(
                    audit.get("observation_domain_node_count", audit["ice_node_count"])
                ),
                "length_score_definition": (
                    "ice-covered nodes inside declared mapped observation domain times node spacing; "
                    ">10 m downstream ice hard-rejects the chain"
                ),
                "precipitation_anomalies": "off",
                "source_spinup_state": str((source / "spinup_state.pkl.gz").resolve()),
                "source_audit": str(audit_path.resolve()),
                "preflight_pass": bool(
                    str(reference["reference_status"]) == "ok"
                    and int(reference["modern_root_count"]) == 1
                    and mismatch <= 0.10
                    and (source / "spinup_state.pkl.gz").exists()
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "geometry_contract.csv", index=False)
    if not bool(frame["preflight_pass"].all()):
        raise ValueError("geometry-contract-failed")
    return frame


def _manifest(project_root: Path, output_root: Path, work_root: Path) -> pd.DataFrame:
    rows = []
    for item in REPRESENTATIVES:
        source = project_root / item["source"]
        source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
        source_delta = float(source_summary["parameters"]["delta_t_c"])
        for value in item["forward_delta_t"]:
            tag = _tag(float(value))
            output = output_root / "runs" / f"system_{item['lia_unique_id']}" / f"forward_dt_{tag}"
            work = work_root / f"s{item['lia_unique_id']}_{tag}"
            rows.append(
                {
                    "lia_unique_id": int(item["lia_unique_id"]),
                    "representative_class": item["class"],
                    "source_dir": str(source.resolve()),
                    "spinup_delta_t_c": source_delta,
                    "forward_delta_t_c": float(value),
                    "is_baseline": math.isclose(float(value), source_delta, abs_tol=1e-12),
                    "output_dir": str(output.resolve()),
                    "workdir": str(work.resolve()),
                    "task_state": "planned",
                }
            )
    return pd.DataFrame(rows)


def _existing_ok(row: dict) -> bool:
    path = Path(row["output_dir"]) / "summary.json"
    if not path.exists():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if str(summary.get("status")) != "ok":
        return False
    if bool(row["is_baseline"]):
        return bool((summary.get("baseline_reproduction") or {}).get("pass"))
    return True


def _run_one(row: dict) -> dict:
    command = [
        sys.executable,
        "-m",
        MODULE,
        "--source-dir",
        str(row["source_dir"]),
        "--workdir",
        str(row["workdir"]),
        "--output-dir",
        str(row["output_dir"]),
        "--forward-delta-t",
        str(row["forward_delta_t_c"]),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        **row,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def _execute(frame: pd.DataFrame, *, workers: int) -> pd.DataFrame:
    records = frame.to_dict("records")
    pending = [row for row in records if not _existing_ok(row)]
    results: dict[tuple[int, float], dict] = {}
    if pending:
        with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as executor:
            futures = {executor.submit(_run_one, row): row for row in pending}
            for future in as_completed(futures):
                result = future.result()
                key = (int(result["lia_unique_id"]), float(result["forward_delta_t_c"]))
                results[key] = result
    output = []
    for row in records:
        key = (int(row["lia_unique_id"]), float(row["forward_delta_t_c"]))
        result = results.get(key, row)
        result["task_state"] = "completed" if _existing_ok(row) else "failed"
        output.append(result)
    return pd.DataFrame(output)


def _summarize(manifest: pd.DataFrame, output_root: Path) -> dict:
    rows = []
    for task in manifest.to_dict("records"):
        path = Path(task["output_dir"]) / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        metrics = summary.get("metrics") or {}
        rows.append(
            {
                "lia_unique_id": int(task["lia_unique_id"]),
                "representative_class": task["representative_class"],
                "spinup_delta_t_c": float(task["spinup_delta_t_c"]),
                "forward_delta_t_c": float(task["forward_delta_t_c"]),
                "is_baseline": bool(task["is_baseline"]),
                "status": summary.get("status", "missing"),
                "observation_domain_overflow": (summary.get("observation_domain") or {}).get("overflow"),
                "forward_peak_m": metrics.get("forward_peak_m"),
                "forward_2000_m": metrics.get("forward_2000_m"),
                "e_lia_sep": metrics.get("e_lia_sep"),
                "e_rgi_sep": metrics.get("e_rgi_sep"),
                "objective_sep": metrics.get("objective_sep"),
                "e_lia_total": metrics.get("e_lia_total"),
                "e_rgi_total": metrics.get("e_rgi_total"),
                "formal_dual_pass": bool(metrics.get("formal_dual_pass", False)),
                "baseline_reproduction_pass": (
                    (summary.get("baseline_reproduction") or {}).get("pass")
                    if bool(task["is_baseline"])
                    else None
                ),
                "runtime_seconds": summary.get("runtime_seconds"),
                "output_dir": task["output_dir"],
            }
        )
    scores = pd.DataFrame(rows)
    scores.to_csv(output_root / "scores.csv", index=False)
    valid = scores[scores["status"].eq("ok") & scores["objective_sep"].notna()].copy()
    best = (
        valid.sort_values(["lia_unique_id", "objective_sep"])
        .groupby("lia_unique_id", as_index=False)
        .first()
    )
    best.to_csv(output_root / "best_per_system.csv", index=False)
    baselines = valid[valid["is_baseline"].astype(bool)].set_index("lia_unique_id")
    improvements = 0
    for row in best.itertuples(index=False):
        if int(row.lia_unique_id) in baselines.index:
            improvements += int(
                float(row.objective_sep)
                < float(baselines.loc[int(row.lia_unique_id), "objective_sep"]) - 1e-12
            )
    formal_systems = sorted(
        scores.loc[scores["formal_dual_pass"].astype(bool), "lia_unique_id"].astype(int).unique().tolist()
    )
    technical_failures = int((~scores["status"].eq("ok")).sum())
    summary = {
        "experiment": "representative_saved_spinup_split_forward_delta_t",
        "planned_chains": int(len(scores)),
        "completed_ok": int(scores["status"].eq("ok").sum()),
        "technical_failures": technical_failures,
        "representatives_improved": int(improvements),
        "formal_pass_systems": formal_systems,
        "formal_pass_count": int(len(formal_systems)),
        "promotion_rule": (
            "promote only if at least two of four representatives improve, at least one "
            "formal dual pass occurs, and invalid/technical rate is below 30%"
        ),
        "promotion_rule_met": bool(
            improvements >= 2 and len(formal_systems) >= 1 and technical_failures / len(scores) < 0.30
        ),
        "spinup_recomputed": False,
        "physical_parameters_changed": False,
        "precipitation_anomalies": "off",
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run(args: argparse.Namespace) -> dict:
    if not 1 <= args.workers <= 12:
        raise ValueError("workers-must-be-between-1-and-12")
    project_root = Path(__file__).resolve().parents[4]
    output_root = project_root / "validation_data/results/paleo_surface_validation_v0/lia_191_split_forward_delta_pilot_v1"
    work_root = project_root / "w191_splitdt"
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    _write_geometry_contract(project_root, output_root)
    manifest = _manifest(project_root, output_root, work_root)

    baseline = manifest[manifest["is_baseline"].astype(bool)].copy()
    baseline = _execute(baseline, workers=min(4, args.workers))
    baseline.to_csv(output_root / "baseline_manifest.csv", index=False)
    baseline_ok = all(_existing_ok(row) for row in baseline.to_dict("records"))
    if not baseline_ok:
        raise RuntimeError("baseline-reproduction-gate-failed")

    directed = manifest[~manifest["is_baseline"].astype(bool)].copy()
    directed = _execute(directed, workers=args.workers)
    combined = pd.concat([baseline, directed], ignore_index=True)
    combined.to_csv(output_root / "manifest.csv", index=False)
    return _summarize(combined, output_root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    try:
        summary = run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
