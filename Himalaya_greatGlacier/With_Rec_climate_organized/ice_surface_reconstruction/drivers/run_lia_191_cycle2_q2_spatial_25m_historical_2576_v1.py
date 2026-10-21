"""Run the gated Cycle-2 Q2 spatial-downstream 25 m pilot for system 2576.

The 25 m spin-up is always rerun from the dedicated spatial-downstream initial
geometry.  A 50 m spin-up state is never read.  Forward execution is released
only after the new base chain is numerically clean, stable, and inside its
registered observation domain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PARENT_RUNNER = (
    "Himalaya_greatGlacier.With_Rec_climate_organized."
    "ice_surface_reconstruction.drivers."
    "lia_lhs_single_chain_spatial_downstream"
)
FORWARD_RUNNER = (
    "Himalaya_greatGlacier.With_Rec_climate_organized."
    "ice_surface_reconstruction.drivers."
    "lia_forward_delta_from_saved_spinup"
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _replace_option(command: list[str], option: str, value: str) -> None:
    matches = [index for index, item in enumerate(command) if item == option]
    if len(matches) != 1 or matches[0] + 1 >= len(command):
        raise ValueError(f"command-option-count:{option}:{len(matches)}")
    command[matches[0] + 1] = value


def _option(command: list[str], option: str) -> str:
    index = command.index(option)
    if index + 1 >= len(command):
        raise ValueError(f"command-option-without-value:{option}")
    return command[index + 1]


def _runner(command: list[str]) -> str:
    if len(command) < 3 or command[1] != "-m":
        raise ValueError("parent-command-is-not-python-module")
    return command[2]


def _forward_observations(series: pd.DataFrame) -> pd.DataFrame:
    forward = series.loc[series["phase"].astype(str).eq("forward")].copy()
    if "observation_length_m" in forward.columns:
        forward["score_length_m"] = pd.to_numeric(
            forward["observation_length_m"], errors="coerce"
        )
    else:
        forward["score_length_m"] = pd.to_numeric(
            forward["length_m"], errors="coerce"
        )
    return forward


def _formal_score(
    *,
    base_summary: dict,
    forward_summary: dict,
    forward_series: pd.DataFrame,
    total_tolerance: float,
    separation_tolerance: float,
    stability_tolerance: float,
) -> dict:
    forward = _forward_observations(forward_series)
    domain = forward_summary.get("observation_domain") or {}
    overflow_column = forward.get("observation_domain_overflow")
    overflow_rows = (
        pd.Series(False, index=forward.index)
        if overflow_column is None
        else overflow_column.astype(bool)
    )
    domain_valid = bool(
        not bool(domain.get("overflow", False))
        and not bool(overflow_rows.any())
        and not forward.empty
        and np.isfinite(forward["score_length_m"].to_numpy(dtype=float)).all()
    )
    finite_state = bool(
        not forward.empty
        and np.isfinite(
            forward[["area_m2", "volume_m3", "raw_model_length_m"]]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        ).all()
    )
    targets = base_summary["targets"]
    lia = float(targets["lia_length_m"])
    rgi = float(targets["rgi_length_m"])
    separation = abs(lia - rgi)
    if not math.isfinite(separation) or separation <= 0.0:
        raise ValueError("nonpositive-lia-rgi-separation")
    peak = float(forward["score_length_m"].max()) if domain_valid else math.nan
    endpoint_rows = forward.loc[
        pd.to_numeric(forward["year"], errors="coerce").eq(2000)
    ]
    endpoint = (
        float(endpoint_rows.iloc[-1]["score_length_m"])
        if domain_valid and not endpoint_rows.empty
        else math.nan
    )
    e_lia_total = abs(peak - lia) / lia if math.isfinite(peak) else math.nan
    e_rgi_total = abs(endpoint - rgi) / rgi if math.isfinite(endpoint) else math.nan
    e_lia_sep = abs(peak - lia) / separation if math.isfinite(peak) else math.nan
    e_rgi_sep = abs(endpoint - rgi) / separation if math.isfinite(endpoint) else math.nan
    spinup = base_summary.get("spinup_control") or {}
    stability = float(spinup.get("final_stability_fraction", math.inf))
    stable = bool(
        base_summary.get("status") == "ok"
        and bool(spinup.get("converged", False))
        and math.isfinite(stability)
        and stability <= stability_tolerance
    )
    technical_clean = bool(
        stable
        and forward_summary.get("status") == "ok"
        and finite_state
        and domain_valid
    )
    total_pass = bool(
        technical_clean
        and e_lia_total <= total_tolerance
        and e_rgi_total <= total_tolerance
    )
    separation_pass = bool(
        technical_clean
        and e_lia_sep <= separation_tolerance
        and e_rgi_sep <= separation_tolerance
    )
    accepted = bool(total_pass and separation_pass)
    return {
        "technical_clean": technical_clean,
        "spinup_converged": bool(spinup.get("converged", False)),
        "spinup_stability_fraction": stability,
        "spinup_stability_pass": stable,
        "observation_domain_valid": domain_valid,
        "observation_domain_node_count": domain.get("node_count"),
        "forward_peak_m": peak,
        "forward_2000_m": endpoint,
        "lia_target_m": lia,
        "rgi_target_m": rgi,
        "lia_rgi_separation_m": separation,
        "e_lia_total": e_lia_total,
        "e_rgi_total": e_rgi_total,
        "e_lia_sep": e_lia_sep,
        "e_rgi_sep": e_rgi_sep,
        "objective_sep": (
            e_lia_sep + e_rgi_sep
            if math.isfinite(e_lia_sep) and math.isfinite(e_rgi_sep)
            else math.nan
        ),
        "total_length_pass": total_pass,
        "formal_separation_pass": separation_pass,
        "formal_dual_pass": accepted,
    }


def _parent_score(
    *,
    parent_dir: Path,
    total_tolerance: float,
    separation_tolerance: float,
    stability_tolerance: float,
) -> dict:
    summary = _load_json(parent_dir / "summary.json")
    series = pd.read_csv(parent_dir / "timeseries.csv")
    return _formal_score(
        base_summary=summary,
        forward_summary=summary,
        forward_series=series,
        total_tolerance=total_tolerance,
        separation_tolerance=separation_tolerance,
        stability_tolerance=stability_tolerance,
    )


def _run_subprocess(command: list[str], log_path: Path) -> dict:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    log_path.write_text(
        "COMMAND\n"
        + json.dumps(command)
        + "\n\nSTDOUT\n"
        + completed.stdout
        + "\n\nSTDERR\n"
        + completed.stderr,
        encoding="utf-8",
    )
    return {
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-3000:],
        "stderr_tail": completed.stderr[-3000:],
    }


def run(config_path: Path) -> dict:
    config = _load_json(config_path)
    project_root = _resolve(Path.cwd(), config.get("project_root", ".")).resolve()
    system_id = int(config["system_id"])
    if system_id != 2576:
        raise ValueError("this-version-is-authorized-only-for-system-2576")
    facts_path = project_root / "LIA_PROJECT_FACTS.md"
    manifest_path = _resolve(project_root, config["task_manifest"]).resolve()
    geometry_root = _resolve(project_root, config["geometry_root"]).resolve()
    output_root = _resolve(project_root, config["output_root"]).resolve()
    work_root = _resolve(project_root, config["work_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    decision_path = geometry_root / "preflight_decision.json"
    preflight_decision = _load_json(decision_path)
    if preflight_decision.get("status") != "passed-preflight":
        raise ValueError("spatial-25m-preflight-not-passed")
    if system_id not in [int(value) for value in preflight_decision["system_ids"]]:
        raise ValueError("system-not-in-spatial-25m-preflight")
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

    manifest = pd.read_csv(manifest_path)
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
        parent_summary["parameters"]["temperature_offset_policy"][
            "spinup_delta_t_c"
        ]
    )
    if not math.isclose(spinup_delta, actual_spinup_delta, abs_tol=1e-12):
        raise ValueError("spinup-delta-parent-conflict")
    if not math.isclose(float(_option(base_command, "--delta-t")), spinup_delta, abs_tol=1e-12):
        raise ValueError("spinup-delta-command-conflict")

    base_dir = output_root / "base_spinup_and_forward"
    final_dir = output_root / "registered_forward"
    base_work = work_root / "base"
    forward_work = work_root / "forward"
    python_executable = str(_resolve(project_root, config["python_executable"]))
    base_command[0] = python_executable
    _replace_option(base_command, "--elevation-band-root", str(geometry_root))
    _replace_option(base_command, "--workdir", str(base_work))
    _replace_option(base_command, "--output-dir", str(base_dir))

    provenance = {
        "schema_version": 1,
        "experiment": "cycle2-q2-spatial-25m-historical-2576-v1",
        "lia_unique_id": system_id,
        "facts_sha256_before_run": _sha256(facts_path),
        "source_products_mutated": False,
        "reuse_50m_spinup_state": False,
        "parent_source_dir": str(parent_source),
        "parent_output_dir": str(parent_output),
        "parent_task": str(parent_task_path),
        "parent_summary": str(parent_summary_path),
        "geometry_root": str(geometry_root),
        "geometry_audit": str(geometry_audit_path),
        "geometry_preflight_decision": str(decision_path),
        "spinup_delta_t_c": spinup_delta,
        "forward_delta_t_c": forward_delta,
        "base_command": base_command,
    }
    _write_json(output_root / "pilot_manifest.json", provenance)
    base_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        base_dir / "task.json",
        {
            "schema_version": 1,
            "experiment": provenance["experiment"],
            "task_id": "2576-cycle2-q2-spatial-25m-base-v1",
            "source_parent_task": str(parent_task_path),
            "geometry_root": str(geometry_root),
            "command": base_command,
        },
    )
    base_run = _run_subprocess(base_command, output_root / "base_run.log")
    base_summary_file = base_dir / "summary.json"
    if base_run["returncode"] != 0 or not base_summary_file.exists():
        decision = {
            **provenance,
            "status": "stopped-base-technical-failure",
            "base_run": base_run,
            "forward_released": False,
        }
        _write_json(output_root / "pilot_decision.json", decision)
        return decision
    base_summary = _load_json(base_summary_file)
    spinup = base_summary.get("spinup_control") or {}
    base_gate = bool(
        base_summary.get("status") == "ok"
        and bool(spinup.get("converged", False))
        and float(spinup.get("final_stability_fraction", math.inf))
        <= float(config["spinup_stability_tolerance"])
        and not bool((base_summary.get("observation_domain") or {}).get("overflow", True))
    )
    if not base_gate:
        decision = {
            **provenance,
            "status": "stopped-base-stability-or-domain-gate",
            "base_run": base_run,
            "base_summary": base_summary,
            "forward_released": False,
        }
        _write_json(output_root / "pilot_decision.json", decision)
        return decision

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
            "task_id": "2576-cycle2-q2-spatial-25m-forward-v1",
            "source_new_25m_spinup": str(base_dir / "spinup_state.pkl.gz"),
            "forward_delta_t_c": forward_delta,
            "command": forward_command,
        },
    )
    forward_run = _run_subprocess(forward_command, output_root / "forward_run.log")
    final_summary_file = final_dir / "summary.json"
    final_series_file = final_dir / "timeseries.csv"
    if (
        forward_run["returncode"] != 0
        or not final_summary_file.exists()
        or not final_series_file.exists()
    ):
        decision = {
            **provenance,
            "status": "stopped-forward-technical-failure",
            "base_run": base_run,
            "forward_run": forward_run,
            "forward_released": True,
        }
        _write_json(output_root / "pilot_decision.json", decision)
        return decision

    final_summary = _load_json(final_summary_file)
    final_series = pd.read_csv(final_series_file)
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
    objective_improvement = float(
        parent_score["objective_sep"] - score["objective_sep"]
    ) if math.isfinite(score["objective_sep"]) else math.nan
    improved = bool(
        math.isfinite(objective_improvement) and objective_improvement > 1e-12
    )
    baseline = final_summary.get("baseline_reproduction")
    baseline_pass = bool(baseline is None or baseline.get("pass", False))
    promotion_signal = bool(
        score["technical_clean"]
        and baseline_pass
        and (score["formal_dual_pass"] or improved)
    )
    final_status = (
        "formal-pass"
        if score["formal_dual_pass"] and baseline_pass
        else (
            "formal-improvement"
            if promotion_signal
            else "stable-valid-no-formal-improvement"
        )
    )
    decision = {
        **provenance,
        "status": final_status,
        "base_run": base_run,
        "forward_run": forward_run,
        "forward_released": True,
        "new_score": score,
        "parent_score": parent_score,
        "objective_improvement_parent_minus_new": objective_improvement,
        "baseline_reproduction_pass": baseline_pass,
        "promotion_signal": promotion_signal,
        "release_1856_5353": False,
        "release_decision_owner": "root-controller",
    }
    _write_json(output_root / "formal_score.json", score)
    pd.DataFrame(
        [
            {"source": "registered_50m_parent", **parent_score},
            {"source": "spatial_25m_pilot", **score},
        ]
    ).to_csv(output_root / "formal_score_comparison.csv", index=False)
    _write_json(output_root / "pilot_decision.json", decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
