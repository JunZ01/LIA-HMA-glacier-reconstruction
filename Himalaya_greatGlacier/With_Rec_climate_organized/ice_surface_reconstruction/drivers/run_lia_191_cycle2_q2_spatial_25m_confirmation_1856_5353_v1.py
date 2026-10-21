"""Run the two released Q2 spatial-downstream 25 m historical confirmations."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .run_lia_191_cycle2_q2_spatial_25m_historical_2576_v1 import (
    FORWARD_RUNNER,
    PARENT_RUNNER,
    _formal_score,
    _load_json,
    _option,
    _parent_score,
    _replace_option,
    _resolve,
    _run_subprocess,
    _runner,
    _sha256,
    _write_json,
)


AUTHORIZED_SYSTEMS = (1856, 5353)


def _run_one(*, config: dict, project_root: Path, manifest: pd.DataFrame, system_id: int) -> dict:
    geometry_root = _resolve(project_root, config["geometry_root"]).resolve()
    output_root = _resolve(project_root, config["output_root"]).resolve()
    work_root = _resolve(project_root, config["work_root"]).resolve()
    case_root = output_root / f"system_{system_id}"
    base_dir = case_root / "base_spinup_and_forward"
    final_dir = case_root / "registered_forward"
    base_work = work_root / f"base_{system_id}"
    forward_work = work_root / f"forward_{system_id}"
    case_root.mkdir(parents=True, exist_ok=True)
    base_work.mkdir(parents=True, exist_ok=True)
    forward_work.mkdir(parents=True, exist_ok=True)

    preflight_decision_path = geometry_root / "preflight_decision.json"
    preflight_decision = _load_json(preflight_decision_path)
    if preflight_decision.get("status") != "passed-preflight":
        raise ValueError("spatial-25m-preflight-not-passed")
    if system_id not in [int(value) for value in preflight_decision["system_ids"]]:
        raise ValueError(f"system-not-in-spatial-25m-preflight:{system_id}")
    geometry_audit_path = geometry_root / str(system_id) / "elevation_band_audit.json"
    geometry_audit = _load_json(geometry_audit_path)
    if geometry_audit.get("geometry_variant") != config["required_geometry_variant"]:
        raise ValueError("spatial-v2-geometry-required")
    if geometry_audit.get("geometry_remap_variant") != config["required_remap_variant"]:
        raise ValueError("wrong-spatial-25m-remap-variant")
    if not math.isclose(
        float(geometry_audit["node_dx_m"]),
        float(config["required_node_dx_m"]),
        abs_tol=1e-12,
    ):
        raise ValueError("wrong-spatial-node-spacing")

    selected = manifest.loc[
        pd.to_numeric(manifest["lia_unique_id"], errors="coerce").eq(system_id)
    ]
    if len(selected) != 1:
        raise ValueError(f"task-manifest-row-count:{system_id}:{len(selected)}")
    row = selected.iloc[0]
    parent_source = Path(str(row["parent_source_dir"])).resolve()
    parent_output = Path(str(row["parent_output_dir"])).resolve()
    parent_task_path = parent_source / "task.json"
    parent_summary_path = parent_source / "summary.json"
    parent_task = _load_json(parent_task_path)
    parent_summary = _load_json(parent_summary_path)
    base_command = [str(value) for value in parent_task["command"]]
    if _runner(base_command) != PARENT_RUNNER:
        raise ValueError("registered-spatial-parent-runner-required")
    spinup_delta = float(row["spinup_delta_t_c"])
    forward_delta = float(row["forward_delta_t_c"])
    actual_spinup_delta = float(
        parent_summary["parameters"]["temperature_offset_policy"]["spinup_delta_t_c"]
    )
    if not math.isclose(spinup_delta, actual_spinup_delta, abs_tol=1e-12):
        raise ValueError("spinup-delta-parent-conflict")
    if not math.isclose(float(_option(base_command, "--delta-t")), spinup_delta, abs_tol=1e-12):
        raise ValueError("spinup-delta-command-conflict")

    python_executable = str(_resolve(project_root, config["python_executable"]))
    base_command[0] = python_executable
    _replace_option(base_command, "--elevation-band-root", str(geometry_root))
    _replace_option(base_command, "--workdir", str(base_work))
    _replace_option(base_command, "--output-dir", str(base_dir))
    provenance = {
        "schema_version": 1,
        "experiment": "cycle2-q2-spatial-25m-confirmation-1856-5353-v1",
        "lia_unique_id": system_id,
        "facts_sha256_before_run": _sha256(project_root / "LIA_PROJECT_FACTS.md"),
        "source_products_mutated": False,
        "reuse_50m_spinup_state": False,
        "parent_source_dir": str(parent_source),
        "parent_output_dir": str(parent_output),
        "parent_task": str(parent_task_path),
        "parent_summary": str(parent_summary_path),
        "geometry_root": str(geometry_root),
        "geometry_audit": str(geometry_audit_path),
        "geometry_preflight_decision": str(preflight_decision_path),
        "spinup_delta_t_c": spinup_delta,
        "forward_delta_t_c": forward_delta,
        "base_command": base_command,
    }
    _write_json(case_root / "pilot_manifest.json", provenance)
    base_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        base_dir / "task.json",
        {
            "schema_version": 1,
            "experiment": provenance["experiment"],
            "task_id": f"{system_id}-cycle2-q2-spatial-25m-base-v1",
            "source_parent_task": str(parent_task_path),
            "geometry_root": str(geometry_root),
            "command": base_command,
        },
    )
    base_run = _run_subprocess(base_command, case_root / "base_run.log")
    base_summary_path = base_dir / "summary.json"
    if base_run["returncode"] != 0 or not base_summary_path.exists():
        result = {
            **provenance,
            "status": "stopped-base-technical-failure",
            "base_run": base_run,
            "forward_released": False,
            "promotion_signal": False,
        }
        _write_json(case_root / "pilot_decision.json", result)
        return result
    base_summary = _load_json(base_summary_path)
    spinup = base_summary.get("spinup_control") or {}
    base_gate = bool(
        base_summary.get("status") == "ok"
        and bool(spinup.get("converged", False))
        and float(spinup.get("final_stability_fraction", math.inf))
        <= float(config["spinup_stability_tolerance"])
        and not bool((base_summary.get("observation_domain") or {}).get("overflow", True))
    )
    if not base_gate:
        result = {
            **provenance,
            "status": "stopped-base-stability-or-domain-gate",
            "base_run": base_run,
            "base_summary": base_summary,
            "forward_released": False,
            "promotion_signal": False,
        }
        _write_json(case_root / "pilot_decision.json", result)
        return result

    final_dir.mkdir(parents=True, exist_ok=True)
    forward_command = [
        python_executable,
        "-m",
        FORWARD_RUNNER,
        "--source-dir",
        str(base_dir),
        "--workdir",
        str(forward_work),
        "--output-dir",
        str(final_dir),
        "--forward-delta-t",
        str(forward_delta),
    ]
    _write_json(
        final_dir / "task.json",
        {
            "schema_version": 1,
            "experiment": provenance["experiment"],
            "task_id": f"{system_id}-cycle2-q2-spatial-25m-forward-v1",
            "source_new_25m_spinup": str(base_dir / "spinup_state.pkl.gz"),
            "forward_delta_t_c": forward_delta,
            "command": forward_command,
        },
    )
    forward_run = _run_subprocess(forward_command, case_root / "forward_run.log")
    final_summary_path = final_dir / "summary.json"
    final_series_path = final_dir / "timeseries.csv"
    if (
        forward_run["returncode"] != 0
        or not final_summary_path.exists()
        or not final_series_path.exists()
    ):
        result = {
            **provenance,
            "status": "stopped-forward-technical-failure",
            "base_run": base_run,
            "forward_run": forward_run,
            "forward_released": True,
            "promotion_signal": False,
        }
        _write_json(case_root / "pilot_decision.json", result)
        return result

    final_summary = _load_json(final_summary_path)
    final_series = pd.read_csv(final_series_path)
    score = _formal_score(
        base_summary=base_summary,
        forward_summary=final_summary,
        forward_series=final_series,
        total_tolerance=float(config["total_length_tolerance"]),
        separation_tolerance=float(config["separation_tolerance"]),
        stability_tolerance=float(config["spinup_stability_tolerance"]),
    )
    parent_score = _parent_score(
        parent_dir=parent_output,
        total_tolerance=float(config["total_length_tolerance"]),
        separation_tolerance=float(config["separation_tolerance"]),
        stability_tolerance=float(config["spinup_stability_tolerance"]),
    )
    improvement = (
        float(parent_score["objective_sep"] - score["objective_sep"])
        if math.isfinite(score["objective_sep"])
        else math.nan
    )
    improved = bool(math.isfinite(improvement) and improvement > 1e-12)
    baseline = final_summary.get("baseline_reproduction")
    baseline_pass = bool(baseline is None or baseline.get("pass", False))
    promotion_signal = bool(
        score["technical_clean"]
        and baseline_pass
        and (score["formal_dual_pass"] or improved)
    )
    status = (
        "formal-pass"
        if score["formal_dual_pass"] and baseline_pass
        else (
            "formal-improvement"
            if promotion_signal
            else "stable-valid-no-formal-improvement"
        )
    )
    result = {
        **provenance,
        "status": status,
        "base_run": base_run,
        "forward_run": forward_run,
        "forward_released": True,
        "new_score": score,
        "parent_score": parent_score,
        "objective_improvement_parent_minus_new": improvement,
        "baseline_reproduction_pass": baseline_pass,
        "promotion_signal": promotion_signal,
    }
    _write_json(case_root / "formal_score.json", score)
    pd.DataFrame(
        [
            {"source": "registered_50m_parent", **parent_score},
            {"source": "spatial_25m_confirmation", **score},
        ]
    ).to_csv(case_root / "formal_score_comparison.csv", index=False)
    _write_json(case_root / "pilot_decision.json", result)
    return result


def _compact(result: dict) -> dict:
    score = result.get("new_score") or {}
    parent = result.get("parent_score") or {}
    return {
        "lia_unique_id": int(result["lia_unique_id"]),
        "status": result.get("status"),
        "technical_clean": bool(score.get("technical_clean", False)),
        "spinup_converged": bool(score.get("spinup_converged", False)),
        "spinup_stability_fraction": score.get("spinup_stability_fraction"),
        "observation_domain_valid": bool(score.get("observation_domain_valid", False)),
        "forward_peak_m": score.get("forward_peak_m"),
        "forward_2000_m": score.get("forward_2000_m"),
        "e_lia_total": score.get("e_lia_total"),
        "e_rgi_total": score.get("e_rgi_total"),
        "e_lia_sep": score.get("e_lia_sep"),
        "e_rgi_sep": score.get("e_rgi_sep"),
        "objective_sep": score.get("objective_sep"),
        "parent_objective_sep": parent.get("objective_sep"),
        "objective_improvement_parent_minus_new": result.get(
            "objective_improvement_parent_minus_new"
        ),
        "formal_dual_pass": bool(score.get("formal_dual_pass", False)),
        "promotion_signal": bool(result.get("promotion_signal", False)),
    }


def run(config_path: Path, workers: int) -> dict:
    config = _load_json(config_path)
    project_root = _resolve(Path.cwd(), config.get("project_root", ".")).resolve()
    system_ids = tuple(int(value) for value in config["system_ids"])
    if system_ids != AUTHORIZED_SYSTEMS:
        raise ValueError(f"authorized-systems-required:{AUTHORIZED_SYSTEMS}")
    if workers < 1 or workers > 12:
        raise ValueError("workers-must-be-between-1-and-12")
    manifest_path = _resolve(project_root, config["task_manifest"]).resolve()
    manifest = pd.read_csv(manifest_path)
    output_root = _resolve(project_root, config["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=min(workers, len(system_ids))) as executor:
        futures = {
            executor.submit(
                _run_one,
                config=config,
                project_root=project_root,
                manifest=manifest,
                system_id=system_id,
            ): system_id
            for system_id in system_ids
        }
        for future in as_completed(futures):
            system_id = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                failure = {
                    "lia_unique_id": system_id,
                    "status": "controller-technical-failure",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "promotion_signal": False,
                }
                _write_json(output_root / f"system_{system_id}" / "pilot_decision.json", failure)
                results.append(failure)

    compact = pd.DataFrame([_compact(result) for result in results]).sort_values(
        "lia_unique_id"
    )
    compact.to_csv(output_root / "confirmation_results.csv", index=False)
    pilot_2576 = _load_json(
        _resolve(project_root, config["pilot_2576_decision"]).resolve()
    )
    representatives = [pilot_2576, *results]
    valid_count = sum(
        bool((item.get("new_score") or {}).get("technical_clean", False))
        for item in representatives
    )
    improvement_count = sum(
        bool(item.get("promotion_signal", False)) for item in representatives
    )
    formal_pass_count = sum(
        bool((item.get("new_score") or {}).get("formal_dual_pass", False))
        for item in representatives
    )
    invalid_fraction = (len(representatives) - valid_count) / len(representatives)
    class_promote = bool(
        invalid_fraction <= float(config["maximum_invalid_fraction"])
        and (formal_pass_count >= 1 or improvement_count >= 2)
    )
    decision = {
        "schema_version": 1,
        "experiment": "cycle2-q2-spatial-25m-confirmation-1856-5353-v1",
        "system_ids": list(system_ids),
        "workers": min(workers, len(system_ids)),
        "representative_ids": [2576, *system_ids],
        "representative_valid_count": valid_count,
        "representative_invalid_fraction": invalid_fraction,
        "representative_improvement_count": improvement_count,
        "representative_formal_pass_count": formal_pass_count,
        "promotion_rule": (
            "invalid_fraction<=0.30 and (at least two valid formal-objective "
            "improvements or one formal pass)"
        ),
        "class_promote_remaining_eight": class_promote,
        "remaining_eight_auto_released": False,
        "release_decision_owner": "root-controller",
        "source_products_mutated": False,
        "results_csv": str(output_root / "confirmation_results.csv"),
    }
    _write_json(output_root / "class_promotion_decision.json", decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.workers), indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
