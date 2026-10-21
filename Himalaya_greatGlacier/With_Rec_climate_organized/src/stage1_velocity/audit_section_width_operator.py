"""Audit section/width structure for the turnover residual union."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import math
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("section_width_audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workers", type=int)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_float(value: object, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return math.nan
    return numerator / denominator if denominator != 0 else math.nan


def gdir_path(root: Path, rgi_id: str) -> Path:
    return root / rgi_id[:-6] / rgi_id[:-3] / rgi_id


def inversion_path(root: Path, rgi_id: str, suffix: str) -> Path:
    return gdir_path(root, rgi_id) / f"inversion_output{suffix}.pkl"


def load_pickle(path: Path):
    with gzip.open(path, "rb") as source:
        return pickle.load(source)


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            output_dir / "section_width_audit.log", mode="w", encoding="utf-8"
        ),
    ):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def interval_overlap(
    model_lower: float,
    model_upper: float,
    observed_low: float,
    observed_high: float,
) -> bool:
    return observed_high >= model_lower and observed_low <= model_upper


def audit_member_worker(task: tuple) -> tuple[str, list[dict], dict | None]:
    rgi_id, records, settings = task
    try:
        root = Path(settings["gdir_root"])
        suffixes = settings["scenarios"]
        outputs = {}
        load_errors = {}
        for scenario, suffix in suffixes.items():
            path = inversion_path(root, rgi_id, suffix)
            try:
                outputs[scenario] = load_pickle(path)
            except Exception as exc:
                load_errors[scenario] = f"{type(exc).__name__}: {exc}"

        audited = []
        for row in records:
            scenario = str(row["scenario"])
            alternate = "millan" if scenario == "farinotti" else "farinotti"
            if scenario not in outputs:
                raise FileNotFoundError(load_errors.get(scenario, scenario))
            line_index = int(row["main_flowline_list_index"])
            gate_index = int(row["gate_flowline_index"])
            line = outputs[scenario][line_index]
            dx = finite_float(line["dx"])
            volume = finite_float(line["volume"][gate_index])
            section = volume / dx
            parent_section = finite_float(row["fixed_section_m2"])
            section_relerror = abs(section - parent_section) / parent_section
            width = finite_float(line["width"][gate_index])
            gate_width = finite_float(row["local_width_m"])
            width_relerror = abs(width - gate_width) / gate_width
            thickness = finite_float(line["thick"][gate_index])
            shape_factor = safe_ratio(section, width * thickness)

            alternate_status = "available" if alternate in outputs else "missing"
            alternate_section = math.nan
            alternate_thickness = math.nan
            if alternate in outputs:
                alternate_line = outputs[alternate][line_index]
                alternate_dx = finite_float(alternate_line["dx"])
                alternate_section = (
                    finite_float(alternate_line["volume"][gate_index])
                    / alternate_dx
                )
                alternate_thickness = finite_float(
                    alternate_line["thick"][gate_index]
                )
            alternate_factor = safe_ratio(alternate_section, section)

            observed = finite_float(row["observed_speed_m_per_year"])
            observed_low = finite_float(row["observed_low_m_per_year"])
            observed_high = finite_float(row["observed_high_m_per_year"])
            model_lower = finite_float(
                row["extreme_imbalance_lower_speed_m_per_year"]
            )
            model_upper = finite_float(row["balance_modeled_speed_m_per_year"])
            median_inside = model_lower <= observed <= model_upper
            registered_overlap = interval_overlap(
                model_lower, model_upper, observed_low, observed_high
            )
            if model_lower > observed_high:
                strict_direction = "strict-too-fast"
                required_factor = model_lower / observed_high
            elif model_upper < observed_low:
                strict_direction = "strict-too-slow"
                required_factor = model_upper / observed_low
            else:
                strict_direction = "interval-overlap"
                required_factor = 1.0

            alternate_lower = safe_ratio(model_lower, alternate_factor)
            alternate_upper = safe_ratio(model_upper, alternate_factor)
            alternate_overlap = (
                alternate_status == "available"
                and interval_overlap(
                    alternate_lower,
                    alternate_upper,
                    observed_low,
                    observed_high,
                )
            )

            outline_width = (
                finite_float(row["gate_intersection_area_km2"])
                * 1e6
                / finite_float(row["gate_along_length_m"])
            )
            outline_width_factor = safe_ratio(outline_width, width)
            outline_lower = safe_ratio(model_lower, outline_width_factor)
            outline_upper = safe_ratio(model_upper, outline_width_factor)
            outline_overlap = interval_overlap(
                outline_lower, outline_upper, observed_low, observed_high
            )

            if registered_overlap:
                routing = "observation-interval-overlap"
            elif alternate_overlap:
                routing = "alternate-thickness-can-close"
            elif outline_overlap:
                routing = "outline-width-proxy-can-close"
            elif strict_direction == "strict-too-fast":
                routing = "larger-section-or-lower-flux-required"
            else:
                routing = "smaller-section-or-higher-flux-required"

            audited.append(
                {
                    "scenario": scenario,
                    "alternate_scenario": alternate,
                    "lia_unique_id": int(row["lia_unique_id"]),
                    "rgi_id": rgi_id,
                    "candidate_id": row["candidate_id"],
                    "residual_in_parent_scenario": row[
                        "three_way_attribution_class"
                    ]
                    == "residual-outside-envelope",
                    "robust_parent_median_residual": bool(
                        row["robust_parent_median_residual"]
                    ),
                    "qc_status": row["qc_status"],
                    "main_flowline_list_index": line_index,
                    "gate_flowline_index": gate_index,
                    "gate_flowline_fraction": finite_float(
                        row["gate_flowline_fraction"]
                    ),
                    "fixed_section_parent_m2": parent_section,
                    "fixed_section_reproduced_m2": section,
                    "section_reproduction_relative_error": section_relerror,
                    "flowline_width_m": width,
                    "velocity_gate_local_width_m": gate_width,
                    "width_reproduction_relative_error": width_relerror,
                    "product_thickness_m": thickness,
                    "section_shape_factor": shape_factor,
                    "is_rectangular": bool(line["is_rectangular"][gate_index]),
                    "is_trapezoid": bool(line["is_trapezoid"][gate_index]),
                    "alternate_product_status": alternate_status,
                    "alternate_section_m2": alternate_section,
                    "alternate_thickness_m": alternate_thickness,
                    "alternate_to_current_section_factor": alternate_factor,
                    "outline_width_proxy_m": outline_width,
                    "flowline_to_outline_width_ratio": safe_ratio(
                        width, outline_width
                    ),
                    "outline_width_section_factor": outline_width_factor,
                    "observed_speed_low_m_per_year": observed_low,
                    "observed_speed_median_m_per_year": observed,
                    "observed_speed_high_m_per_year": observed_high,
                    "model_extreme_lower_speed_m_per_year": model_lower,
                    "model_balance_upper_speed_m_per_year": model_upper,
                    "parent_median_inside_envelope": median_inside,
                    "observation_interval_overlaps_envelope": registered_overlap,
                    "strict_residual_direction": strict_direction,
                    "section_factor_to_interval_boundary": required_factor,
                    "section_m2_at_interval_boundary": section * required_factor,
                    "width_m_at_interval_boundary_hold_thickness": (
                        width * required_factor
                    ),
                    "alternate_model_lower_speed_m_per_year": alternate_lower,
                    "alternate_model_upper_speed_m_per_year": alternate_upper,
                    "alternate_section_interval_overlap": alternate_overlap,
                    "outline_width_model_lower_speed_m_per_year": outline_lower,
                    "outline_width_model_upper_speed_m_per_year": outline_upper,
                    "outline_width_interval_overlap": outline_overlap,
                    "gate_operator_routing": routing,
                    "width_proxy_scope": "rgi-outline-480m-slab-average",
                }
            )
        return rgi_id, audited, None
    except Exception as exc:
        return rgi_id, [], {
            "rgi_id": rgi_id,
            "stage": "section-width-worker",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def direction_label(series: pd.Series) -> str:
    directions = sorted(set(series) - {"interval-overlap"})
    if not directions:
        return "interval-overlap"
    if len(directions) > 1:
        return "mixed-direction"
    return directions[0]


def aggregate_system_scenarios(gates: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (scenario, system_id), frame in gates.groupby(
        ["scenario", "lia_unique_id"], sort=True
    ):
        residual = bool(frame["residual_in_parent_scenario"].iloc[0])
        strict = frame[
            frame["strict_residual_direction"] != "interval-overlap"
        ]
        strict_direction = direction_label(frame["strict_residual_direction"])
        all_registered = bool(
            frame["observation_interval_overlaps_envelope"].all()
        )
        alt_closes = bool(
            (
                frame["observation_interval_overlaps_envelope"]
                | frame["alternate_section_interval_overlap"]
            ).all()
        )
        width_closes = bool(
            (
                frame["observation_interval_overlaps_envelope"]
                | frame["outline_width_interval_overlap"]
            ).all()
        )
        if not residual:
            routing = "not-parent-residual-in-this-scenario"
        elif all_registered:
            routing = "median-only-residual-observation-interval-overlap"
        elif strict_direction == "mixed-direction":
            routing = "strict-cross-gate-direction-conflict"
        elif alt_closes:
            routing = "alternate-thickness-can-close"
        elif width_closes:
            routing = "outline-width-proxy-can-close"
        elif strict_direction == "strict-too-fast":
            routing = "larger-section-or-lower-flux-required"
        else:
            routing = "smaller-section-or-higher-flux-required"
        records.append(
            {
                "scenario": scenario,
                "lia_unique_id": int(system_id),
                "parent_residual_in_scenario": residual,
                "robust_parent_median_residual": bool(
                    frame["robust_parent_median_residual"].iloc[0]
                ),
                "gate_count": int(len(frame)),
                "parent_median_outside_gate_count": int(
                    (~frame["parent_median_inside_envelope"]).sum()
                ),
                "strict_residual_gate_count": int(len(strict)),
                "strict_too_fast_gate_count": int(
                    (strict["strict_residual_direction"] == "strict-too-fast").sum()
                ),
                "strict_too_slow_gate_count": int(
                    (strict["strict_residual_direction"] == "strict-too-slow").sum()
                ),
                "strict_residual_direction": strict_direction,
                "all_gate_observation_intervals_overlap": all_registered,
                "alternate_section_closes_all_strict_gates": alt_closes,
                "outline_width_proxy_closes_all_strict_gates": width_closes,
                "system_scenario_routing": routing,
                "max_section_reproduction_relative_error": finite_float(
                    frame["section_reproduction_relative_error"].max()
                ),
                "max_width_reproduction_relative_error": finite_float(
                    frame["width_reproduction_relative_error"].max()
                ),
                "median_alternate_to_current_section_factor": finite_float(
                    frame["alternate_to_current_section_factor"].median()
                ),
                "median_flowline_to_outline_width_ratio": finite_float(
                    frame["flowline_to_outline_width_ratio"].median()
                ),
            }
        )
    return pd.DataFrame(records)


def aggregate_consensus(
    systems: pd.DataFrame, union_ids: set[int], robust_ids: set[int]
) -> pd.DataFrame:
    records = []
    for system_id in sorted(union_ids):
        frame = systems[systems["lia_unique_id"] == system_id].set_index("scenario")
        f = frame.loc["farinotti"] if "farinotti" in frame.index else None
        m = frame.loc["millan"] if "millan" in frame.index else None
        strict_scenarios = [
            row
            for row in (f, m)
            if row is not None and int(row["strict_residual_gate_count"]) > 0
        ]
        strict_direction = direction_label(
            pd.Series(
                [row["strict_residual_direction"] for row in strict_scenarios],
                dtype=object,
            )
        )
        robust = system_id in robust_ids
        strict_both = len(strict_scenarios) == 2
        if robust and strict_both and strict_direction == "strict-too-fast":
            consensus = "robust-strict-too-fast"
        elif robust and strict_both and strict_direction == "strict-too-slow":
            consensus = "robust-strict-too-slow"
        elif robust and strict_both:
            consensus = "robust-strict-mixed-direction"
        elif robust and strict_scenarios:
            consensus = "robust-parent-median-one-scenario-strict"
        elif robust:
            consensus = "robust-parent-median-only-interval-overlap"
        elif strict_scenarios:
            consensus = "scenario-sensitive-strict-residual"
        else:
            consensus = "scenario-sensitive-median-only-residual"
        records.append(
            {
                "lia_unique_id": int(system_id),
                "robust_parent_median_residual": robust,
                "strict_residual_in_both_scenarios": strict_both,
                "consensus_strict_direction": strict_direction,
                "consensus_class": consensus,
                "farinotti_routing": (
                    f["system_scenario_routing"] if f is not None else "missing"
                ),
                "millan_routing": (
                    m["system_scenario_routing"] if m is not None else "missing"
                ),
                "farinotti_strict_gate_count": (
                    int(f["strict_residual_gate_count"]) if f is not None else 0
                ),
                "millan_strict_gate_count": (
                    int(m["strict_residual_gate_count"]) if m is not None else 0
                ),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    output_dir = (REPO_ROOT / config["output_dir"]).resolve()
    configure_logging(output_dir)
    started = time.time()
    workers = int(args.workers or config["workers"])
    attribution_dir = (REPO_ROOT / config["attribution_dir"]).resolve()
    system_attribution_path = attribution_dir / "system_attribution.csv"
    gate_attribution_path = attribution_dir / "gate_attribution.csv"
    velocity_path = (REPO_ROOT / config["velocity_gate_path"]).resolve()
    gdir_root = (REPO_ROOT / config["gdir_root"]).resolve()

    parent_systems = pd.read_csv(system_attribution_path)
    parent_residual = parent_systems[
        parent_systems["three_way_attribution_class"]
        == "residual-outside-envelope"
    ]
    scenario_ids = {
        scenario: set(frame["lia_unique_id"].astype(int))
        for scenario, frame in parent_residual.groupby("scenario")
    }
    union_ids = set().union(*scenario_ids.values())
    robust_ids = set.intersection(*scenario_ids.values())

    gates = pd.read_csv(gate_attribution_path)
    gates = gates[gates["lia_unique_id"].isin(union_ids)].copy()
    expected_gate_row_count = len(gates)
    expected_system_scenario_count = len(
        gates[["scenario", "lia_unique_id"]].drop_duplicates()
    )
    velocity = pd.read_csv(velocity_path)
    velocity_fields = [
        "rgi_id",
        "main_flowline_list_index",
        "gate_flowline_index",
        "local_width_m",
        "gate_intersection_area_km2",
        "gate_along_length_m",
    ]
    gates = gates.merge(
        velocity[velocity_fields], on="rgi_id", validate="many_to_one"
    )
    parent_classes = parent_systems[
        ["scenario", "lia_unique_id", "three_way_attribution_class"]
    ]
    gates = gates.merge(
        parent_classes,
        on=["scenario", "lia_unique_id"],
        validate="many_to_one",
    )
    if len(gates) != expected_gate_row_count:
        raise RuntimeError(
            "System-attribution join changed gate rows: "
            f"{expected_gate_row_count} -> {len(gates)}"
        )
    gates["robust_parent_median_residual"] = gates["lia_unique_id"].isin(
        robust_ids
    )

    suffixes = config["scenarios"]
    involved_rgi_ids = sorted(gates["rgi_id"].unique())
    source_paths = {
        f"{rgi_id}:{scenario}": inversion_path(gdir_root, rgi_id, suffix)
        for rgi_id in involved_rgi_ids
        for scenario, suffix in suffixes.items()
        if inversion_path(gdir_root, rgi_id, suffix).exists()
    }
    source_hashes_before = {
        key: sha256(path) for key, path in source_paths.items()
    }
    settings = {
        "gdir_root": str(gdir_root),
        "scenarios": suffixes,
    }
    tasks = [
        (rgi_id, frame.to_dict("records"), settings)
        for rgi_id, frame in gates.groupby("rgi_id", sort=True)
    ]
    total_tasks = len(tasks)

    write_json_atomic(
        output_dir / "run_manifest.json",
        {
            "run_id": config["run_id"],
            "method_version": config["method_version"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": sha256(config_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "input_hashes": {
                "system_attribution": sha256(system_attribution_path),
                "gate_attribution": sha256(gate_attribution_path),
                "velocity_gates": sha256(velocity_path),
            },
            "source_pickle_hashes_before": source_hashes_before,
            "gdir_read_only": True,
            "worker_count": workers,
            "worker_write_policy": "workers-return-rows-parent-writes",
            "parent_residual_union_system_count": len(union_ids),
            "parent_robust_median_residual_count": len(robust_ids),
            "member_gate_task_count": total_tasks,
            "expected_gate_row_count": expected_gate_row_count,
            "expected_system_scenario_count": expected_system_scenario_count,
        },
    )
    write_json_atomic(
        output_dir / "progress.json",
        {
            "run_id": config["run_id"],
            "phase": "section-width-audit",
            "state": "running",
            "completed_task_count": 0,
            "total_task_count": total_tasks,
            "failure_count": 0,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    gate_rows = []
    failures = []
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(audit_member_worker, task): task[0] for task in tasks
        }
        for future in as_completed(futures):
            rgi_id = futures[future]
            try:
                _, rows, failure = future.result()
            except Exception as exc:
                rows = []
                failure = {
                    "rgi_id": rgi_id,
                    "stage": "process-pool",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            gate_rows.extend(rows)
            if failure:
                failures.append(failure)
            completed += 1
            write_json_atomic(
                output_dir / "progress.json",
                {
                    "run_id": config["run_id"],
                    "phase": "section-width-audit",
                    "state": "running" if completed < total_tasks else "complete",
                    "completed_task_count": completed,
                    "total_task_count": total_tasks,
                    "last_rgi_id": rgi_id,
                    "failure_count": len(failures),
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            if completed % 10 == 0 or completed == total_tasks:
                LOGGER.info("Progress %d/%d last=%s", completed, total_tasks, rgi_id)

    gate_frame = pd.DataFrame(gate_rows).sort_values(
        ["lia_unique_id", "scenario", "rgi_id"]
    )
    failure_frame = pd.DataFrame(
        failures, columns=["rgi_id", "stage", "error_type", "error"]
    )
    system_frame = aggregate_system_scenarios(gate_frame)
    if len(gate_frame) != expected_gate_row_count:
        raise RuntimeError(
            f"Audited gate count mismatch: {len(gate_frame)} != "
            f"{expected_gate_row_count}"
        )
    if len(system_frame) != expected_system_scenario_count:
        raise RuntimeError(
            f"Audited system-scenario count mismatch: {len(system_frame)} != "
            f"{expected_system_scenario_count}"
        )
    consensus_frame = aggregate_consensus(system_frame, union_ids, robust_ids)
    priority_frame = consensus_frame[
        (
            consensus_frame["farinotti_strict_gate_count"]
            + consensus_frame["millan_strict_gate_count"]
        )
        > 0
    ].copy()

    gate_frame.to_csv(output_dir / "gate_section_audit.csv", index=False)
    system_frame.to_csv(output_dir / "system_scenario_audit.csv", index=False)
    consensus_frame.to_csv(output_dir / "system_consensus_audit.csv", index=False)
    priority_frame.to_csv(output_dir / "priority_strict_residuals.csv", index=False)
    failure_frame.to_csv(output_dir / "failures.csv", index=False)

    source_hashes_after = {
        key: sha256(path) for key, path in source_paths.items()
    }
    source_unchanged = source_hashes_before == source_hashes_after
    section_tolerance = float(
        config["section_reproduction_relative_tolerance"]
    )
    width_tolerance = float(config["width_reproduction_relative_tolerance"])
    strict_by_scenario = {}
    for scenario, frame in system_frame.groupby("scenario"):
        residual_frame = frame[frame["parent_residual_in_scenario"]]
        strict_by_scenario[scenario] = {
            "parent_median_residual_system_count": int(len(residual_frame)),
            "all_gate_observation_interval_overlap_count": int(
                residual_frame["all_gate_observation_intervals_overlap"].sum()
            ),
            "strict_residual_system_count": int(
                (~residual_frame["all_gate_observation_intervals_overlap"]).sum()
            ),
            "routing_counts": residual_frame[
                "system_scenario_routing"
            ].value_counts().to_dict(),
        }
    summary = {
        "run_id": config["run_id"],
        "status": "completed-read-only-section-width-audit",
        "worker_count": workers,
        "gate_row_count": int(len(gate_frame)),
        "system_scenario_count": int(len(system_frame)),
        "consensus_system_count": int(len(consensus_frame)),
        "parent_residual_union_system_count": int(len(union_ids)),
        "parent_robust_median_residual_count": int(len(robust_ids)),
        "strict_residual_consensus_count": int(len(priority_frame)),
        "failure_count": int(len(failure_frame)),
        "source_pickle_hashes_unchanged": source_unchanged,
        "section_reproduction_pass": bool(
            gate_frame["section_reproduction_relative_error"].max()
            <= section_tolerance
        ),
        "width_reproduction_pass": bool(
            gate_frame["width_reproduction_relative_error"].max()
            <= width_tolerance
        ),
        "maximum_section_reproduction_relative_error": finite_float(
            gate_frame["section_reproduction_relative_error"].max()
        ),
        "maximum_width_reproduction_relative_error": finite_float(
            gate_frame["width_reproduction_relative_error"].max()
        ),
        "strict_residual_by_scenario": strict_by_scenario,
        "consensus_class_counts": consensus_frame[
            "consensus_class"
        ].value_counts().to_dict(),
        "elapsed_seconds": round(time.time() - started, 3),
        "scope_note": (
            "outline width is a 480 m RGI slab proxy; velocity alone cannot "
            "separate section error from flux error"
        ),
    }
    write_json_atomic(output_dir / "diagnostic_summary.json", summary)
    LOGGER.info("Completed %s", json.dumps(summary, ensure_ascii=True))
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
