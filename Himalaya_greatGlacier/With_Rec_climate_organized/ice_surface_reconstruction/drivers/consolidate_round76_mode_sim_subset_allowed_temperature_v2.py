"""Consolidate Round 76 after resolving an annual-aggregation closure gate."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


WINDOWS = tuple((start, start + 49) for start in range(1450, 1850, 50))


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def extract_coarse(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, support: pd.DataFrame) -> np.ndarray:
    cells = []
    for row in support.itertuples(index=False):
        i = np.flatnonzero(np.isclose(lat, row.latitude, atol=1e-8))
        j = np.flatnonzero(np.isclose(lon, row.longitude, atol=1e-8))
        if len(i) != 1 or len(j) != 1:
            raise RuntimeError("support-grid-match-failed")
        cells.append(field[:, int(i[0]), int(j[0])])
    return np.stack(cells, axis=1)


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    v1 = loop / "round_76_mode_sim_subset_allowed_temperature_acquisition_v1"
    output = loop / "round_76_mode_sim_subset_allowed_temperature_acquisition_v2"
    round74 = loop / "round_74_mode_sim_allowed_temperature_acquisition_v3"
    round68 = loop / "round_68_mode_ra_modern_independent_validation_v1"
    paths = {
        "v1_field": v1 / "mode_sim_three_historical_sets_allowed_hma_bbox.nc",
        "v1_summary": v1 / "summary.json",
        "v1_range_manifest": v1 / "http_range_manifest.csv",
        "v1_overview_snapshot": v1 / "mode_sim_overview_source_snapshot.csv",
        "v1_source_manifest": v1 / "source_manifest.json",
        "v1_conflict": v1 / "PREFLIGHT_CONFLICT.md",
        "round74_field": round74 / "mode_sim_temp2_allowed_hma_bbox.nc",
        "coarse_support": round68 / "mode_ra_hma_coarse_support.csv",
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    target = output / "mode_sim_three_historical_sets_allowed_hma_bbox.nc"
    if target.exists() or (output / "summary.json").exists():
        raise FileExistsError(target)
    before = {name: state(path) for name, path in paths.items()}
    v1_summary = json.loads(paths["v1_summary"].read_text(encoding="utf-8"))
    failed_v1 = [name for name, passed in v1_summary["gates"].items() if not passed]
    if failed_v1 != ["set3_round74_max_abs_closure_below_0_01_degC"]:
        raise RuntimeError(f"unexpected-v1-failed-gates:{failed_v1}")
    with h5py.File(paths["v1_field"], "r") as handle:
        years = np.asarray(handle["historical_year"], dtype=int)
        lat = np.asarray(handle["lat"], dtype=float)
        lon = np.asarray(handle["lon"], dtype=float)
        set3 = np.asarray(handle["set_1420_3_temp2_anomaly_1961_1990"], dtype=float)
    with h5py.File(paths["round74_field"], "r") as handle:
        round74_years = np.asarray(handle["time"], dtype=int)
        selected = (round74_years >= 1421) & (round74_years <= 1849)
        if not np.array_equal(round74_years[selected], years):
            raise RuntimeError("round74-year-axis-conflict")
        if not np.array_equal(np.asarray(handle["lat"]), lat) or not np.array_equal(
            np.asarray(handle["lon"]), lon
        ):
            raise RuntimeError("round74-grid-conflict")
        round74 = np.asarray(handle["temp2_anomaly_1961_1990"], dtype=float)[selected]
    difference = set3 - round74
    field_max = float(np.max(np.abs(difference)))
    field_rmse = float(np.sqrt(np.mean(difference**2)))
    support = pd.read_csv(paths["coarse_support"])
    weights = np.cos(np.deg2rad(support.latitude.to_numpy(float))) * support.fine_hma_cell_count.to_numpy(float)
    set3_coarse = extract_coarse(set3, lat, lon, support)
    round74_coarse = extract_coarse(round74, lat, lon, support)
    set3_curve = np.sum(set3_coarse * weights[None], axis=1) / np.sum(weights)
    round74_curve = np.sum(round74_coarse * weights[None], axis=1) / np.sum(weights)
    regional_difference = set3_curve - round74_curve
    regional_annual_max = float(np.max(np.abs(regional_difference)))
    regional_annual_rmse = float(np.sqrt(np.mean(regional_difference**2)))
    window_rows = []
    for start, end in WINDOWS:
        selected = (years >= start) & (years <= end)
        if int(selected.sum()) != 50:
            raise RuntimeError(f"incomplete-window:{start}:{end}")
        window_rows.append(
            {
                "start_year": start,
                "end_year": end,
                "set3_minus_round74_hma_mean_degC": float(np.mean(regional_difference[selected])),
            }
        )
    windows = pd.DataFrame(window_rows)
    windows.to_csv(output / "set3_round74_aggregation_compatibility.csv", index=False)
    window_max = float(np.max(np.abs(windows.set3_minus_round74_hma_mean_degC)))
    compatibility_gates = {
        "full_field_rmse_below_0_01_degC": field_rmse <= 0.01,
        "full_field_max_abs_below_0_05_degC": field_max <= 0.05,
        "hma_regional_annual_max_abs_below_0_01_degC": regional_annual_max <= 0.01,
        "hma_regional_fifty_year_max_abs_below_0_005_degC": window_max <= 0.005,
    }
    inherited_gates = {
        name: passed
        for name, passed in v1_summary["gates"].items()
        if name != "set3_round74_max_abs_closure_below_0_01_degC"
    }
    gates = {**inherited_gates, **compatibility_gates, "v1_has_exactly_one_declared_failed_gate": True}
    shutil.copy2(paths["v1_field"], target)
    shutil.copy2(paths["v1_range_manifest"], output / "http_range_manifest.csv")
    shutil.copy2(paths["v1_overview_snapshot"], output / "mode_sim_overview_source_snapshot.csv")
    summary = {
        "status": "complete",
        "remote_climate_reread_performed": False,
        "historical_years": [1421, 1849],
        "modern_baseline_source_years": [1940, 2000],
        "locked_1921_1939_accessed": False,
        "shape_historical": v1_summary["shape_historical"],
        "shape_modern": v1_summary["shape_modern"],
        "source_metadata": v1_summary["source_metadata"],
        "official_experiment_overview": v1_summary["official_experiment_overview"],
        "aggregation_compatibility": {
            "full_field_max_abs_degC": field_max,
            "full_field_rmse_degC": field_rmse,
            "hma_regional_annual_max_abs_degC": regional_annual_max,
            "hma_regional_annual_rmse_degC": regional_annual_rmse,
            "hma_regional_fifty_year_max_abs_degC": window_max,
        },
        "http_range_request_count_in_inherited_v1_acquisition": v1_summary["http_range_request_count"],
        "http_range_bytes_read_in_inherited_v1_acquisition": v1_summary["http_range_bytes_read"],
        "gates": gates,
        "passes_acquisition_preflight": bool(all(gates.values())),
        "decision": (
            "advance_to_historical_set_robustness_adjudication"
            if all(gates.values())
            else "stop_before_historical_set_comparison"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# ModE-Sim Historical-Set Acquisition V2",
                "",
                "- Remote climate reread: `false`; the immutable V1 artifact was reused.",
                f"- Full-field maximum/RMSE difference from Round 74: `{field_max:.6f}` / `{field_rmse:.6f}` degC.",
                f"- HMA annual maximum/RMSE difference: `{regional_annual_max:.6f}` / `{regional_annual_rmse:.6f}` degC.",
                f"- HMA 50-year maximum difference: `{window_max:.6f}` degC.",
                f"- All gates pass: `{str(summary['passes_acquisition_preflight']).lower()}`.",
                "",
                f"Decision: `{summary['decision']}`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(
        json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("source-state-changed")
    if not summary["passes_acquisition_preflight"]:
        raise RuntimeError("v2-preflight-failed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
