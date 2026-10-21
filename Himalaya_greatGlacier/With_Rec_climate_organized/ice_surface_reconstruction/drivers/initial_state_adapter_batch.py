"""Batch smoke/bulk runner for the LIA v0.1 initial-state adapter."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
import warnings
from pathlib import Path

import pandas as pd

from ..paths import resolve
from .initial_state_adapter import adapt_system


_OGGM_INITIALIZED = False


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def find_system_netcdf(source_dir: Path, lia_unique_id: int) -> Path:
    matches = sorted(source_dir.glob(f"lia_{int(lia_unique_id)}_*_system_grid_v1.nc"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected-one-system-netcdf:{lia_unique_id}:found={len(matches)}"
        )
    return matches[0]


def read_flowlines(path: Path) -> list:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with gzip.open(path, "rb") as source:
            return pickle.load(source)


def try_fluxbased_init(path: Path) -> tuple[str, float | None, float | None, str]:
    try:
        from oggm import cfg
        from oggm.core.flowline import FluxBasedModel

        global _OGGM_INITIALIZED
        if not _OGGM_INITIALIZED:
            cfg.initialize(logging_level="WARNING")
            _OGGM_INITIALIZED = True
        flowlines = read_flowlines(path)
        model = FluxBasedModel(flowlines, y0=0.0, glen_a=2.4e-24, fs=0.0)
        return "ok", float(model.volume_km3), float(model.area_km2), ""
    except Exception as exc:  # pragma: no cover - exercised in OGGM env smoke runs
        return "failed", None, None, f"{type(exc).__name__}: {exc}"


def flatten_summary(summary: dict, check_oggm_init: bool) -> list[dict]:
    rows = []
    for scenario in summary["scenarios"]:
        if scenario["status"] != "adapted":
            rows.append(
                {
                    "lia_unique_id": summary["lia_unique_id"],
                    "main_rgi_id": summary["main_rgi_id"],
                    "bed_scenario": scenario["bed_scenario"],
                    "status": scenario["status"],
                    "adapter_qc_status": "skipped",
                    "skip_reason": scenario.get("skip_reason", ""),
                }
            )
            continue
        init_statuses = []
        init_errors = []
        init_volume = 0.0
        init_area = 0.0
        if check_oggm_init:
            for member in scenario["member_flowlines"]:
                status, volume, area, error = try_fluxbased_init(
                    Path(member["output_pickle"])
                )
                init_statuses.append(status)
                if volume is not None:
                    init_volume += volume
                if area is not None:
                    init_area += area
                if error:
                    init_errors.append(f"{member['rgi_id']}={error}")
        thickness_scale_factor = float(scenario["thickness_scale_factor"])
        width_scale_factor = float(scenario.get("width_scale_factor", 1.0))
        if width_scale_factor < 0.5 or width_scale_factor > 2.0:
            adapter_qc_status = "width-remap-review"
        elif thickness_scale_factor < 0.5 or thickness_scale_factor > 2.0:
            adapter_qc_status = "thickness-remap-review"
        elif init_statuses and any(item != "ok" for item in init_statuses):
            adapter_qc_status = "oggm-init-review"
        else:
            adapter_qc_status = "production-initial-state"
        rows.append(
            {
                "lia_unique_id": summary["lia_unique_id"],
                "main_rgi_id": summary["main_rgi_id"],
                "bed_scenario": scenario["bed_scenario"],
                "status": scenario["status"],
                "adapter_qc_status": adapter_qc_status,
                "member_count": len(scenario["member_flowlines"]),
                "source_volume_km3": scenario["source_volume_km3"],
                "flowline_volume_km3": scenario["aggregate_flowline_volume_km3"],
                "volume_error_fraction": scenario["flowline_minus_source_volume_fraction"],
                "source_positive_area_km2": scenario["source_positive_area_km2"],
                "flowline_area_km2": scenario["aggregate_flowline_area_km2"],
                "area_error_fraction": (
                    (scenario["aggregate_flowline_area_km2"] - scenario["source_positive_area_km2"])
                    / scenario["source_positive_area_km2"]
                    if scenario["source_positive_area_km2"] > 0
                    else None
                ),
                "pre_scale_flowline_volume_km3": scenario["pre_scale_flowline_volume_km3"],
                "thickness_scale_factor": thickness_scale_factor,
                "width_scale_factor": width_scale_factor,
                "bed_fallback_node_count": sum(
                    int(member["bed_fallback_node_count"])
                    for member in scenario["member_flowlines"]
                ),
                "invalid_node_count": sum(
                    int(member["invalid_node_count"])
                    for member in scenario["member_flowlines"]
                ),
                "maximum_sampled_thickness_m": max(
                    float(member["maximum_sampled_thickness_m"])
                    for member in scenario["member_flowlines"]
                ),
                "oggm_init_status": (
                    "not-run"
                    if not check_oggm_init
                    else (
                        "ok"
                        if init_statuses and all(item == "ok" for item in init_statuses)
                        else "failed"
                    )
                ),
                "oggm_init_volume_km3": init_volume if check_oggm_init else None,
                "oggm_init_area_km2": init_area if check_oggm_init else None,
                "oggm_init_errors": ";".join(init_errors),
                "skip_reason": "",
            }
        )
    return rows


def parse_system_ids(values: list[str] | None, source_dir: Path) -> list[int]:
    if values:
        return [int(value) for item in values for value in str(item).split(",") if value]
    ids = []
    for path in source_dir.glob("lia_*_*_system_grid_v1.nc"):
        ids.append(int(path.name.split("_")[1]))
    return sorted(ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--gdir-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--system-manifest", required=True)
    parser.add_argument("--system-id", action="append")
    parser.add_argument("--check-oggm-init", action="store_true")
    parser.add_argument("--volume-policy", default="scale-thickness-to-source-volume")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = resolve(args.source_dir)
    output_root = resolve(args.output_root)
    system_ids = parse_system_ids(args.system_id, source_dir)
    rows = []
    failures = []
    for lia_unique_id in system_ids:
        try:
            summary = adapt_system(
                system_netcdf=find_system_netcdf(source_dir, lia_unique_id),
                gdir_root=resolve(args.gdir_root),
                output_root=output_root,
                system_manifest=resolve(args.system_manifest),
                volume_policy=args.volume_policy,
            )
            rows.extend(flatten_summary(summary, args.check_oggm_init))
        except Exception as exc:
            failures.append(
                {
                    "lia_unique_id": int(lia_unique_id),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    frame = pd.DataFrame(rows)
    failure_frame = pd.DataFrame(failures)
    frame.to_csv(output_root / "initial_state_adapter_summary.csv", index=False)
    failure_frame.to_csv(output_root / "initial_state_adapter_failures.csv", index=False)
    status = "completed" if not failures else "completed-with-failures"
    payload = {
        "status": status,
        "system_count": len(system_ids),
        "row_count": int(len(frame)),
        "adapted_member_count": (
            int(frame["status"].eq("adapted").sum()) if "status" in frame else 0
        ),
        "skipped_member_count": (
            int(frame["status"].eq("skipped").sum()) if "status" in frame else 0
        ),
        "failure_count": int(len(failure_frame)),
        "check_oggm_init": bool(args.check_oggm_init),
        "all_adapted_rows_init_ok": bool(
            (not frame.empty)
            and frame.loc[frame["status"].eq("adapted"), "oggm_init_status"].eq("ok").all()
        )
        if args.check_oggm_init
        else None,
        "maximum_abs_volume_error_fraction": (
            float(frame["volume_error_fraction"].dropna().abs().max())
            if "volume_error_fraction" in frame and not frame.empty
            else None
        ),
        "minimum_thickness_scale_factor": (
            float(frame["thickness_scale_factor"].dropna().min())
            if "thickness_scale_factor" in frame and not frame.empty
            else None
        ),
        "maximum_thickness_scale_factor": (
            float(frame["thickness_scale_factor"].dropna().max())
            if "thickness_scale_factor" in frame and not frame.empty
            else None
        ),
        "median_area_error_fraction": (
            float(frame["area_error_fraction"].dropna().median())
            if "area_error_fraction" in frame and not frame.empty
            else None
        ),
        "maximum_abs_area_error_fraction": (
            float(frame["area_error_fraction"].dropna().abs().max())
            if "area_error_fraction" in frame and not frame.empty
            else None
        ),
        "area_error_abs_over_0p15_count": (
            int((frame["area_error_fraction"].dropna().abs() > 0.15).sum())
            if "area_error_fraction" in frame and not frame.empty
            else 0
        ),
        "area_error_abs_over_0p25_count": (
            int((frame["area_error_fraction"].dropna().abs() > 0.25).sum())
            if "area_error_fraction" in frame and not frame.empty
            else 0
        ),
        "scale_factor_outside_0p5_2p0_count": (
            int(
                (
                    (frame["thickness_scale_factor"].dropna() < 0.5)
                    | (frame["thickness_scale_factor"].dropna() > 2.0)
                ).sum()
            )
            if "thickness_scale_factor" in frame and not frame.empty
            else 0
        ),
        "minimum_width_scale_factor": (
            float(frame["width_scale_factor"].dropna().min())
            if "width_scale_factor" in frame and not frame.empty
            else None
        ),
        "maximum_width_scale_factor": (
            float(frame["width_scale_factor"].dropna().max())
            if "width_scale_factor" in frame and not frame.empty
            else None
        ),
        "width_scale_factor_outside_0p5_2p0_count": (
            int(
                (
                    (frame["width_scale_factor"].dropna() < 0.5)
                    | (frame["width_scale_factor"].dropna() > 2.0)
                ).sum()
            )
            if "width_scale_factor" in frame and not frame.empty
            else 0
        ),
        "production_initial_state_count": (
            int(frame["adapter_qc_status"].eq("production-initial-state").sum())
            if "adapter_qc_status" in frame and not frame.empty
            else 0
        ),
        "width_remap_review_count": (
            int(frame["adapter_qc_status"].eq("width-remap-review").sum())
            if "adapter_qc_status" in frame and not frame.empty
            else 0
        ),
        "thickness_remap_review_count": (
            int(frame["adapter_qc_status"].eq("thickness-remap-review").sum())
            if "adapter_qc_status" in frame and not frame.empty
            else 0
        ),
    }
    write_json(output_root / "initial_state_adapter_summary.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
