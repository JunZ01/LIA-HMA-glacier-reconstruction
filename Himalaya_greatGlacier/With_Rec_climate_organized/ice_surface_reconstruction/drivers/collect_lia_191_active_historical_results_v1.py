"""Collect active historical screens without starting any OGGM process.

The collector treats the accepted registry and remaining-route table as the
authoritative population state.  It rescales every saved historical summary
against the route-table length targets and follows its linked steady-state
summary for the stability and free-domain gates.  It only *writes commands*
for independent reproduction; it never executes them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"json-root-not-object:{path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _resolve(spec: Any, project_root: Path) -> Path:
    if isinstance(spec, dict):
        anchor = str(spec.get("anchor", "project"))
        sub = Path(str(spec["sub"]))
        if anchor == "project":
            return (project_root / sub).resolve()
        if anchor == "validation":
            return (project_root / "validation_data" / sub).resolve()
        raise ValueError(f"unsupported-path-anchor:{anchor}")
    path = Path(str(spec))
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _powershell_quote(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _historical_domain_pass(summary: dict[str, Any], end_year: int) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if summary.get("runtime_error") not in (None, ""):
        reasons.append("historical-runtime-error")
    final = summary.get("final") if isinstance(summary.get("final"), dict) else {}
    if int(_finite(final.get("year")) or -1) != end_year:
        reasons.append("historical-incomplete")
    for label in ("initial", "peak", "final"):
        snapshot = summary.get(label) if isinstance(summary.get(label), dict) else {}
        if not _truth(snapshot.get("finite_geometry")):
            reasons.append(f"{label}-nonfinite-geometry")
        negative = _finite(snapshot.get("negative_section_count"))
        if negative is None or negative > 0:
            reasons.append(f"{label}-negative-section")
        boundary = _finite(snapshot.get("root_array_end_thickness_m"))
        if boundary is None or boundary > 1e-9:
            reasons.append(f"{label}-root-boundary-ice")
        free = _finite(snapshot.get("remaining_free_downstream_m"))
        if free is None or free < 0:
            reasons.append(f"{label}-observation-domain-overflow")
    return not reasons, reasons


def _linked_steady(
    config_snapshot: dict[str, Any], project_root: Path, thresholds: dict[str, float]
) -> tuple[dict[str, Any], list[str]]:
    result: dict[str, Any] = {
        "steady_summary_path": None,
        "steady_status": None,
        "steady_volume_change_fraction_50y": None,
        "steady_final_free_downstream_m": None,
        "stability_pass": False,
        "steady_domain_pass": False,
    }
    reasons: list[str] = []
    steady_spec = config_snapshot.get("steady_root")
    if steady_spec is None:
        return result, ["steady-root-missing-from-config"]
    steady_root = _resolve(steady_spec, project_root)
    path = steady_root / "steady_50y_summary.json"
    result["steady_summary_path"] = str(path)
    if not path.is_file():
        return result, ["steady-summary-missing"]
    try:
        steady = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return result, [f"steady-summary-unreadable:{type(exc).__name__}"]
    change = _finite(steady.get("volume_change_fraction_50y"))
    final = steady.get("final") if isinstance(steady.get("final"), dict) else {}
    free = _finite(final.get("remaining_free_downstream_m"))
    stable = bool(
        steady.get("status") == "steady-50y-pass"
        and change is not None
        and abs(change) <= thresholds["stability_abs_fraction_max"]
    )
    domain = bool(
        _truth(final.get("finite_geometry"))
        and (_finite(final.get("negative_section_count")) == 0.0)
        and (_finite(final.get("root_array_end_thickness_m")) or 0.0) <= 1e-9
        and free is not None
        and free >= thresholds["steady_free_downstream_m_min"]
    )
    result.update(
        {
            "steady_status": steady.get("status"),
            "steady_volume_change_fraction_50y": change,
            "steady_final_free_downstream_m": free,
            "stability_pass": stable,
            "steady_domain_pass": domain,
        }
    )
    if not stable:
        reasons.append("steady-stability-gate")
    if not domain:
        reasons.append("steady-domain-gate")
    return result, reasons


def _candidate_dirs(project_root: Path, patterns: list[str]) -> list[Path]:
    found: dict[str, Path] = {}
    for pattern in patterns:
        for path in project_root.glob(pattern):
            if path.is_dir() and (path / "config_snapshot.json").is_file():
                found[str(path.resolve()).lower()] = path.resolve()
    return sorted(found.values(), key=lambda item: item.name.lower())


def _is_historical_config(config: dict[str, Any]) -> bool:
    return bool(
        "forward_delta_t_end_c" in config
        and int(_finite(config.get("start_year")) or -1) == 1470
        and int(_finite(config.get("end_year")) or -1) == 2000
    )


def _score_candidate(
    directory: Path,
    route: pd.Series,
    accepted_ids: set[int],
    thresholds: dict[str, float],
    project_root: Path,
) -> dict[str, Any]:
    config_path = directory / "config_snapshot.json"
    summary_path = directory / "historical_summary.json"
    config = _read_json(config_path)
    system_id = int(config["system_id"])
    base: dict[str, Any] = {
        "system_id": system_id,
        "rgi_id": str(route["rgi_id"]),
        "candidate_root": str(directory),
        "config_snapshot": str(config_path),
        "historical_summary": str(summary_path),
        "accepted45_member": system_id in accepted_ids,
        "route_v3": str(route["current_route"]),
        "route_v3_gate": str(route["current_gate"]),
        "collection_state": "complete" if summary_path.is_file() else "pending",
        "formal_pass": False,
        "reproduction_required": False,
        "failure_reasons": "",
    }
    linked, steady_reasons = _linked_steady(config, project_root, thresholds)
    base.update(linked)
    lia = float(route["lia_registered_length_m"])
    rgi = float(route["rgi_registered_length_m"])
    separation = abs(lia - rgi)
    base.update(
        {
            "lia_registered_length_m": lia,
            "rgi_registered_length_m": rgi,
            "endpoint_separation_m": separation,
            "forward_delta_t_start_c": _finite(config.get("stage2", {}).get("delta_t_c")),
            "forward_delta_t_end_c": _finite(config.get("forward_delta_t_end_c")),
        }
    )
    if not summary_path.is_file():
        base["failure_reasons"] = ";".join(["historical-summary-pending", *steady_reasons])
        # The equilibrium endpoint is the historical initial state.  This makes
        # the one-sided peak lower bound available even before history finishes.
        initial_length = None
        steady_path = linked.get("steady_summary_path")
        if steady_path and Path(str(steady_path)).is_file():
            steady = _read_json(Path(str(steady_path)))
            initial_length = _finite((steady.get("final") or {}).get("length_m"))
        base.update(_initial_peak_bound(initial_length, lia, separation, thresholds))
        return base
    try:
        summary = _read_json(summary_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        base["collection_state"] = "summary-unreadable"
        base["failure_reasons"] = f"historical-summary-unreadable:{type(exc).__name__}"
        return base
    initial = summary.get("initial") if isinstance(summary.get("initial"), dict) else {}
    peak = summary.get("peak") if isinstance(summary.get("peak"), dict) else {}
    final = summary.get("final") if isinstance(summary.get("final"), dict) else {}
    initial_length = _finite(initial.get("length_m"))
    peak_length = _finite(peak.get("length_m"))
    final_length = _finite(final.get("length_m"))
    e_lia_total = None if peak_length is None else abs(peak_length - lia) / lia
    e_rgi_total = None if final_length is None else abs(final_length - rgi) / rgi
    e_lia_sep = None if peak_length is None or separation <= 0 else abs(peak_length - lia) / separation
    e_rgi_sep = None if final_length is None or separation <= 0 else abs(final_length - rgi) / separation
    total_pass = bool(
        e_lia_total is not None
        and e_rgi_total is not None
        and e_lia_total <= thresholds["total_error_max"]
        and e_rgi_total <= thresholds["total_error_max"]
    )
    separation_pass = bool(
        e_lia_sep is not None
        and e_rgi_sep is not None
        and e_lia_sep <= thresholds["separation_error_max"]
        and e_rgi_sep <= thresholds["separation_error_max"]
    )
    historical_domain_pass, domain_reasons = _historical_domain_pass(summary, 2000)
    reasons = list(steady_reasons)
    if not total_pass:
        reasons.append("total-endpoint-gate")
    if not separation_pass:
        reasons.append("separation-endpoint-gate")
    if not historical_domain_pass:
        reasons.extend(domain_reasons)
    if summary.get("status") != "historical-pass":
        reasons.append("historical-summary-status-not-pass")
    bound = _initial_peak_bound(initial_length, lia, separation, thresholds)
    if bound["initial_peak_lower_bound_unreachable"]:
        reasons.append("initial-peak-lower-bound-unreachable")
    formal_pass = bool(
        total_pass
        and separation_pass
        and historical_domain_pass
        and linked["stability_pass"]
        and linked["steady_domain_pass"]
        and summary.get("status") == "historical-pass"
        and not bound["initial_peak_lower_bound_unreachable"]
    )
    objective = None
    if None not in (e_lia_total, e_rgi_total, e_lia_sep, e_rgi_sep):
        objective = float(e_lia_total + e_rgi_total + e_lia_sep + e_rgi_sep)
    base.update(
        {
            "collection_state": "complete",
            "historical_status": summary.get("status"),
            "historical_initial_length_m": initial_length,
            "historical_peak_length_m": peak_length,
            "historical_peak_year": _finite(peak.get("year")),
            "historical_2000_length_m": final_length,
            "historical_final_free_downstream_m": _finite(final.get("remaining_free_downstream_m")),
            "historical_minimum_timestep_seconds": _finite(summary.get("minimum_timestep_seconds")),
            "e_lia_total_formal": e_lia_total,
            "e_rgi_total_formal": e_rgi_total,
            "e_lia_separation_formal": e_lia_sep,
            "e_rgi_separation_formal": e_rgi_sep,
            "formal_objective_sum": objective,
            "total_gate_pass": total_pass,
            "separation_gate_pass": separation_pass,
            "historical_domain_pass": historical_domain_pass,
            "formal_pass": formal_pass,
            "reproduction_required": formal_pass and system_id not in accepted_ids,
            "failure_reasons": ";".join(dict.fromkeys(reasons)),
            **bound,
        }
    )
    return base


def _initial_peak_bound(
    initial_length: float | None,
    lia_length: float,
    separation: float,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    total_upper = lia_length * (1.0 + thresholds["total_error_max"])
    sep_upper = lia_length + thresholds["separation_error_max"] * separation
    joint_upper = min(total_upper, sep_upper)
    if initial_length is None:
        return {
            "initial_peak_lower_bound_m": None,
            "formal_peak_joint_upper_m": joint_upper,
            "initial_peak_lower_bound_excess_m": None,
            "initial_peak_lower_bound_unreachable": False,
        }
    excess = max(initial_length - joint_upper, 0.0)
    return {
        "initial_peak_lower_bound_m": initial_length,
        "formal_peak_joint_upper_m": joint_upper,
        "initial_peak_lower_bound_excess_m": excess,
        "initial_peak_lower_bound_unreachable": excess > 1e-9,
    }


def _reproduction_rows(
    frame: pd.DataFrame, config: dict[str, Any], project_root: Path, output_dir: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    python = Path(str(config["python_executable"])).resolve()
    module_run = str(config["reproduction_runner_module"])
    module_compare = str(config["reproduction_compare_module"])
    rows: list[dict[str, Any]] = []
    commands: list[str] = ["$ErrorActionPreference = 'Stop'", ""]
    selected = frame[frame["reproduction_required"].eq(True)].copy()
    selected = selected.sort_values(["system_id", "formal_objective_sum", "candidate_root"])
    for ordinal, item in enumerate(selected.itertuples(index=False), start=1):
        original = Path(str(item.candidate_root)).resolve()
        reproduction = original.parent / f"{original.name}_independent_reproduction_v1"
        comparison = output_dir / f"reproduction_comparison_{int(item.system_id)}_{ordinal:02d}.json"
        command_run = (
            f"& {_powershell_quote(python)} -m {module_run} "
            f"--config {_powershell_quote(Path(str(item.config_snapshot)).resolve())} "
            f"--end-delta-t {float(item.forward_delta_t_end_c):.17g} "
            f"--output-root {_powershell_quote(reproduction)}"
        )
        command_compare = (
            f"& {_powershell_quote(python)} -m {module_compare} "
            f"--original {_powershell_quote(original)} "
            f"--reproduction {_powershell_quote(reproduction)} "
            f"--output-json {_powershell_quote(comparison)}"
        )
        rows.append(
            {
                "system_id": int(item.system_id),
                "candidate_root": str(original),
                "reproduction_root": str(reproduction),
                "comparison_json": str(comparison),
                "output_collision_now": reproduction.exists(),
                "run_command": command_run,
                "compare_command": command_compare,
            }
        )
        commands.extend([f"# system {int(item.system_id)}", command_run, command_compare, ""])
    return rows, commands


def run(config_path: Path) -> dict[str, Any]:
    config = _read_json(config_path.resolve())
    project_root = _resolve(config.get("project_root", "."), PROJECT_ROOT)
    output_dir = _resolve(config["output_dir"], project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = _resolve(config["accepted_registry_csv"], project_root)
    route_path = _resolve(config["remaining_route_csv"], project_root)
    accepted = pd.read_csv(accepted_path)
    route = pd.read_csv(route_path)
    accepted_ids = set(pd.to_numeric(accepted["system_id"], errors="raise").astype(int))
    route["lia_unique_id"] = pd.to_numeric(route["lia_unique_id"], errors="raise").astype(int)
    route = route.set_index("lia_unique_id", drop=False)
    thresholds = {key: float(value) for key, value in config["thresholds"].items()}
    rows: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    for directory in _candidate_dirs(project_root, [str(value) for value in config["scan_globs"]]):
        try:
            snapshot = _read_json(directory / "config_snapshot.json")
            if not _is_historical_config(snapshot):
                ignored.append({"candidate_root": str(directory), "reason": "not-historical-config"})
                continue
            system_id = int(snapshot["system_id"])
            if system_id not in route.index:
                ignored.append({"candidate_root": str(directory), "reason": f"system-not-in-route-v3:{system_id}"})
                continue
            rows.append(
                _score_candidate(
                    directory, route.loc[system_id], accepted_ids, thresholds, project_root
                )
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            rows.append(
                {
                    "candidate_root": str(directory),
                    "collection_state": "collector-error",
                    "formal_pass": False,
                    "reproduction_required": False,
                    "failure_reasons": f"{type(exc).__name__}:{exc}",
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(
            columns=["system_id", "candidate_root", "collection_state", "formal_pass", "reproduction_required"]
        )
    frame.to_csv(output_dir / "candidate_results.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(ignored).to_csv(output_dir / "ignored_roots.csv", index=False)
    complete = frame[frame["collection_state"].eq("complete")].copy()
    if not complete.empty:
        complete["_pass_rank"] = ~complete["formal_pass"].astype(bool)
        complete["_objective_rank"] = pd.to_numeric(complete["formal_objective_sum"], errors="coerce").fillna(float("inf"))
        best = (
            complete.sort_values(["system_id", "_pass_rank", "_objective_rank", "candidate_root"])
            .groupby("system_id", as_index=False)
            .first()
            .drop(columns=["_pass_rank", "_objective_rank"])
        )
    else:
        best = complete
    best.to_csv(output_dir / "system_best_results.csv", index=False)
    reproduction_rows, commands = _reproduction_rows(frame, config, project_root, output_dir)
    pd.DataFrame(reproduction_rows).to_csv(output_dir / "reproduction_queue.csv", index=False)
    (output_dir / "reproduction_commands.ps1").write_text("\n".join(commands), encoding="utf-8")
    status_counts = frame["collection_state"].value_counts(dropna=False).to_dict()
    summary = {
        "status": "collection-complete-no-process-started",
        "collector_version": "v1",
        "config": str(config_path.resolve()),
        "accepted_registry": str(accepted_path),
        "accepted_registry_count": len(accepted_ids),
        "remaining_route": str(route_path),
        "route_row_count": int(len(route)),
        "scan_globs": list(config["scan_globs"]),
        "candidate_row_count": int(len(frame)),
        "collection_state_counts": {str(key): int(value) for key, value in status_counts.items()},
        "new_formal_pass_count": int(frame["reproduction_required"].fillna(False).astype(bool).sum()),
        "formal_pass_including_registry_count": int(frame["formal_pass"].fillna(False).astype(bool).sum()),
        "initial_peak_lower_bound_unreachable_count": int(
            frame.get("initial_peak_lower_bound_unreachable", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
        ),
        "reproduction_command_pair_count": len(reproduction_rows),
        "thresholds": thresholds,
        "oggm_started": False,
        "worker_directories_modified": False,
        "outputs": {
            "candidate_results": str(output_dir / "candidate_results.csv"),
            "system_best_results": str(output_dir / "system_best_results.csv"),
            "reproduction_queue": str(output_dir / "reproduction_queue.csv"),
            "reproduction_commands": str(output_dir / "reproduction_commands.ps1"),
            "ignored_roots": str(output_dir / "ignored_roots.csv"),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
