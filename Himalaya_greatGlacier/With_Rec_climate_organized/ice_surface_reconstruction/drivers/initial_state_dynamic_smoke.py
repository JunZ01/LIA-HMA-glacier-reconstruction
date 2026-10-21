"""Short dynamic smoke test for adapted LIA OGGM flowline initial states.

This runner intentionally does not process or branch climate scenarios. It
loads adapted LIA flowline pickles, instantiates OGGM's FluxBasedModel, runs a
short zero-mass-balance geometry step, and records numerical stability metrics.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import resolve


_OGGM_INITIALIZED = False


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_flowlines(path: Path) -> list:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with gzip.open(path, "rb") as source:
            return pickle.load(source)


def ensure_oggm_initialized() -> None:
    global _OGGM_INITIALIZED
    if _OGGM_INITIALIZED:
        return
    from oggm import cfg

    cfg.initialize(logging_level="WARNING")
    _OGGM_INITIALIZED = True


def find_main_pickle(adapter_root: Path, lia_unique_id: int, scenario: str, rgi_id: str) -> Path:
    """Return the main-RGI pickle for backward-compatible reporting only."""
    path = (
        adapter_root
        / str(int(lia_unique_id))
        / scenario
        / rgi_id
        / f"model_flowlines_lia_{scenario}.pkl"
    )
    if not path.exists():
        raise FileNotFoundError(f"missing-adapted-flowlines:{path}")
    return path


def find_member_pickles(
    adapter_root: Path, lia_unique_id: int, scenario: str
) -> list[Path]:
    """Return every adapted RGI member in one LIA system/bed package."""
    root = adapter_root / str(int(lia_unique_id)) / scenario
    paths = sorted(root.glob(f"*/model_flowlines_lia_{scenario}.pkl"))
    if not paths:
        raise FileNotFoundError(f"missing-adapted-system-package:{root}")
    return paths


def run_one(path: Path, years: float, glen_a: float, fs: float) -> dict:
    ensure_oggm_initialized()
    from oggm.core.flowline import FluxBasedModel
    from oggm.core.massbalance import ScalarMassBalance

    flowlines = read_flowlines(path)
    model = FluxBasedModel(
        flowlines,
        mb_model=ScalarMassBalance(mb=0.0),
        y0=0.0,
        glen_a=float(glen_a),
        fs=float(fs),
    )
    initial_volume = float(model.volume_km3)
    initial_area = float(model.area_km2)
    initial_length = float(model.length_m)
    status = "ok"
    error = ""
    try:
        model.run_until(float(years))
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    final_volume = float(model.volume_km3) if status == "ok" else math.nan
    final_area = float(model.area_km2) if status == "ok" else math.nan
    final_length = float(model.length_m) if status == "ok" else math.nan
    finite = bool(
        status == "ok"
        and np.isfinite([final_volume, final_area, final_length]).all()
        and final_volume >= 0
        and final_area >= 0
    )
    return {
        "initial_volume_km3": initial_volume,
        "final_volume_km3": final_volume,
        "volume_change_fraction": (
            (final_volume - initial_volume) / initial_volume
            if status == "ok" and initial_volume > 0
            else math.nan
        ),
        "initial_area_km2": initial_area,
        "final_area_km2": final_area,
        "area_change_fraction": (
            (final_area - initial_area) / initial_area
            if status == "ok" and initial_area > 0
            else math.nan
        ),
        "initial_length_m": initial_length,
        "final_length_m": final_length,
        "finite_nonnegative": finite,
        "dynamic_smoke_status": status,
        "dynamic_smoke_error": error,
    }


def aggregate_member_results(results: list[dict]) -> dict:
    if not results:
        raise ValueError("At least one member result is required")
    ok = [row["dynamic_smoke_status"] == "ok" for row in results]
    status = "ok" if all(ok) else "failed"
    initial_volume = float(sum(row["initial_volume_km3"] for row in results))
    initial_area = float(sum(row["initial_area_km2"] for row in results))
    initial_length = float(sum(row["initial_length_m"] for row in results))
    if status == "ok":
        final_volume = float(sum(row["final_volume_km3"] for row in results))
        final_area = float(sum(row["final_area_km2"] for row in results))
        final_length = float(sum(row["final_length_m"] for row in results))
    else:
        final_volume = math.nan
        final_area = math.nan
        final_length = math.nan
    return {
        "initial_volume_km3": initial_volume,
        "final_volume_km3": final_volume,
        "volume_change_fraction": (
            (final_volume - initial_volume) / initial_volume
            if status == "ok" and initial_volume > 0
            else math.nan
        ),
        "initial_area_km2": initial_area,
        "final_area_km2": final_area,
        "area_change_fraction": (
            (final_area - initial_area) / initial_area
            if status == "ok" and initial_area > 0
            else math.nan
        ),
        "initial_length_m": initial_length,
        "final_length_m": final_length,
        "finite_nonnegative": bool(
            status == "ok" and all(row["finite_nonnegative"] for row in results)
        ),
        "dynamic_smoke_status": status,
        "dynamic_smoke_error": ";".join(
            str(row["dynamic_smoke_error"])
            for row in results
            if row["dynamic_smoke_error"]
        ),
        "member_run_count": len(results),
        "member_ok_count": int(sum(ok)),
        "member_failure_count": int(len(results) - sum(ok)),
    }


def select_rows(frame: pd.DataFrame, system_ids: list[int] | None, max_rows: int | None) -> pd.DataFrame:
    rows = frame[frame["adapter_qc_status"].eq("production-initial-state")].copy()
    if system_ids:
        rows = rows[rows["lia_unique_id"].astype(int).isin(system_ids)].copy()
    rows = rows.sort_values(["lia_unique_id", "bed_scenario"]).reset_index(drop=True)
    if max_rows is not None:
        rows = rows.head(int(max_rows)).copy()
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--system-id", action="append")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--years", type=float, default=5.0)
    parser.add_argument("--glen-a", type=float, default=2.4e-24)
    parser.add_argument("--fs", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_root = resolve(args.adapter_root)
    output_root = resolve(args.output_root)
    summary_path = adapter_root / "initial_state_adapter_summary.csv"
    frame = pd.read_csv(summary_path)
    system_ids = (
        [int(value) for item in args.system_id for value in str(item).split(",") if value]
        if args.system_id
        else None
    )
    rows = select_rows(frame, system_ids, args.max_rows)
    results = []
    member_results = []
    for row in rows.itertuples(index=False):
        main_path = find_main_pickle(
            adapter_root,
            int(row.lia_unique_id),
            str(row.bed_scenario),
            str(row.main_rgi_id),
        )
        paths = find_member_pickles(
            adapter_root, int(row.lia_unique_id), str(row.bed_scenario)
        )
        result = {
            "lia_unique_id": int(row.lia_unique_id),
            "main_rgi_id": str(row.main_rgi_id),
            "bed_scenario": str(row.bed_scenario),
            "adapter_qc_status": str(row.adapter_qc_status),
            "input_pickle": str(main_path),
            "input_package_dir": str(main_path.parents[1]),
            "input_pickle_count": len(paths),
            "input_pickles": ";".join(str(path) for path in paths),
        }
        system_members = []
        for path in paths:
            member_result = run_one(path, args.years, args.glen_a, args.fs)
            member_result.update(
                {
                    "lia_unique_id": int(row.lia_unique_id),
                    "main_rgi_id": str(row.main_rgi_id),
                    "bed_scenario": str(row.bed_scenario),
                    "member_rgi_id": path.parent.name,
                    "input_pickle": str(path),
                }
            )
            member_results.append(member_result)
            system_members.append(member_result)
        result.update(aggregate_member_results(system_members))
        results.append(result)
    output_root.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(results)
    out.to_csv(output_root / "initial_state_dynamic_smoke.csv", index=False)
    member_out = pd.DataFrame(member_results)
    member_out.to_csv(
        output_root / "initial_state_dynamic_smoke_members.csv", index=False
    )
    ok = out["dynamic_smoke_status"].eq("ok") if not out.empty else pd.Series(dtype=bool)
    payload = {
        "status": "completed" if bool(ok.all()) else "completed-with-failures",
        "row_count": int(len(out)),
        "member_run_count": int(len(member_out)),
        "ok_count": int(ok.sum()) if not out.empty else 0,
        "failure_count": int((~ok).sum()) if not out.empty else 0,
        "member_ok_count": int(
            member_out["dynamic_smoke_status"].eq("ok").sum()
        ) if not member_out.empty else 0,
        "member_failure_count": int(
            member_out["dynamic_smoke_status"].ne("ok").sum()
        ) if not member_out.empty else 0,
        "years": float(args.years),
        "maximum_abs_volume_change_fraction": (
            float(out["volume_change_fraction"].dropna().abs().max())
            if not out.empty and not out["volume_change_fraction"].dropna().empty
            else None
        ),
        "maximum_abs_area_change_fraction": (
            float(out["area_change_fraction"].dropna().abs().max())
            if not out.empty and not out["area_change_fraction"].dropna().empty
            else None
        ),
    }
    write_json(output_root / "initial_state_dynamic_smoke_summary.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
