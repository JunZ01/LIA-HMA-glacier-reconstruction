"""Repair Round 69 recentering using the verified Round 67 modern baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


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


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    v1 = loop / "round_69_mode_ra_allowed_paleo_temperature_acquisition_v1"
    round67 = loop / "round_67_mode_ra_allowed_modern_subset_acquisition_v2"
    output = loop / "round_69_mode_ra_allowed_paleo_temperature_acquisition_v2"
    paths = {
        "v1_paleo_field": v1 / "mode_ra_temp2_1421_1920_hma_bbox.nc",
        "v1_summary": v1 / "summary.json",
        "v1_range_manifest": v1 / "http_range_manifest.csv",
        "v1_source_manifest": v1 / "source_manifest.json",
        "round67_modern_field": round67 / "mode_ra_ensmean_1940_2000_hma_bbox.nc",
        "round67_summary": round67 / "summary.json",
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    target = output / "mode_ra_temp2_1421_1920_hma_bbox.nc"
    if target.exists() or (output / "summary.json").exists():
        raise FileExistsError(target)
    before = {name: state(path) for name, path in paths.items()}
    v1_summary = json.loads(paths["v1_summary"].read_text(encoding="utf-8"))
    round67_summary = json.loads(paths["round67_summary"].read_text(encoding="utf-8"))
    if v1_summary["locked_1921_1939_accessed"]:
        raise RuntimeError("v1-locked-period-flagged")
    expected_v1_true = {
        key for key, value in v1_summary["gates"].items() if value
    }
    expected_all_but_baseline = set(v1_summary["gates"]) - {
        "recentered_1961_1990_closure_below_1e_6"
    }
    if expected_v1_true != expected_all_but_baseline:
        raise RuntimeError(f"v1-had-additional-failed-gates:{expected_all_but_baseline - expected_v1_true}")
    if not round67_summary["passes_acquisition_preflight"]:
        raise RuntimeError("round67-modern-baseline-did-not-pass")

    with h5py.File(paths["v1_paleo_field"], "r") as handle:
        years = np.asarray(handle["time"], dtype=int)
        lat = np.asarray(handle["lat"], dtype=float)
        lon = np.asarray(handle["lon"], dtype=float)
        paleo_source = np.asarray(handle["temp2_anomaly_source_baseline"], dtype=float)
    with h5py.File(paths["round67_modern_field"], "r") as handle:
        modern_years = np.asarray(handle["time"], dtype=int)
        modern_lat = np.asarray(handle["lat"], dtype=float)
        modern_lon = np.asarray(handle["lon"], dtype=float)
        modern_source = np.asarray(handle["temp2_anomaly"], dtype=float)
    if not np.array_equal(years, np.arange(1421, 1921)):
        raise RuntimeError("v1-paleo-year-axis-failed")
    if not np.array_equal(modern_years, np.arange(1940, 2001)):
        raise RuntimeError("round67-modern-year-axis-failed")
    if not np.array_equal(lat, modern_lat) or not np.array_equal(lon, modern_lon):
        raise RuntimeError("paleo-modern-grid-conflict")
    if not np.all(np.isfinite(paleo_source)) or not np.all(np.isfinite(modern_source)):
        raise RuntimeError("paleo-or-modern-temperature-not-fully-finite")
    baseline = (modern_years >= 1961) & (modern_years <= 1990)
    center = np.mean(modern_source[baseline], axis=0)
    paleo_recentered = paleo_source - center[None]
    modern_recentered = modern_source - center[None]
    closure = float(np.max(np.abs(np.mean(modern_recentered[baseline], axis=0))))

    with h5py.File(target, "w") as handle:
        handle.attrs["title"] = "ModE-RA modern-validated temperature, allowed paleo HMA subset"
        handle.attrs["source_anomaly_baseline"] = "1901-2000"
        handle.attrs["recenter_source"] = str(paths["round67_modern_field"])
        handle.attrs["locked_1921_1939_accessed"] = False
        handle.create_dataset("time", data=years.astype(np.int16))
        handle["time"].attrs["units"] = "year CE"
        handle.create_dataset("lat", data=lat)
        handle.create_dataset("lon", data=lon)
        handle.create_dataset(
            "temp2_anomaly_source_baseline",
            data=paleo_source.astype(np.float32),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            fillvalue=np.float32(np.nan),
        )
        handle["temp2_anomaly_source_baseline"].attrs["units"] = "degC"
        handle["temp2_anomaly_source_baseline"].attrs["baseline"] = "1901-2000"
        handle.create_dataset(
            "temp2_anomaly_1961_1990",
            data=paleo_recentered.astype(np.float32),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            fillvalue=np.float32(np.nan),
        )
        handle["temp2_anomaly_1961_1990"].attrs["units"] = "degC"
        handle["temp2_anomaly_1961_1990"].attrs["baseline"] = "1961-1990 from Round 67 modern subset"
        handle.create_dataset("temp2_1961_1990_grid_center", data=center.astype(np.float32))

    reopen_closure = np.nan
    with h5py.File(target, "r") as handle:
        reopened = np.asarray(handle["temp2_anomaly_1961_1990"], dtype=float)
        reopened_source = np.asarray(handle["temp2_anomaly_source_baseline"], dtype=float)
        reopened_center = np.asarray(handle["temp2_1961_1990_grid_center"], dtype=float)
        reopen_closure = float(
            np.max(np.abs(reopened - (reopened_source - reopened_center[None])))
        )
    gates = {
        "v1_all_remote_and_field_gates_except_impossible_baseline_passed": True,
        "round67_modern_baseline_preflight_passed": True,
        "paleo_axis_is_exact_1421_1920": np.array_equal(years, np.arange(1421, 1921)),
        "paleo_and_modern_grids_identical": True,
        "modern_1961_1990_recenter_closure_below_1e_12": closure < 1e-12,
        "reopened_paleo_recenter_equation_closes_below_1e_6": reopen_closure < 1e-6,
        "paleo_output_fully_finite": bool(np.all(np.isfinite(reopened))),
        "locked_1921_1939_accessed": False,
        "no_remote_reread_in_v2": True,
    }
    summary = {
        "status": "complete",
        "years": [1421, 1920],
        "shape": list(paleo_source.shape),
        "locked_1921_1939_accessed": False,
        "v1_remote_request_count_reused": v1_summary["http_range_request_count"],
        "v1_remote_bytes_reused": v1_summary["http_range_bytes_read"],
        "v2_remote_request_count": 0,
        "modern_1961_1990_recenter_closure_max_abs": closure,
        "reopened_paleo_recenter_equation_max_abs": reopen_closure,
        "gates": gates,
        "passes_acquisition_preflight": bool(all(gates.values())),
        "decision": (
            "advance_mode_ra_temperature_to_allowed_paleo_comparison"
            if all(gates.values())
            else "stop_mode_ra_paleo_temperature_on_consolidation_conflict"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# ModE-RA Allowed Paleo Temperature Acquisition v2",
                "",
                f"- Preserved v1 remote requests/bytes: `{v1_summary['http_range_request_count']}` / `{v1_summary['http_range_bytes_read']}`.",
                "- v2 remote requests: `0`.",
                f"- Years/shape: `1421-1920` / `{paleo_source.shape}`.",
                f"- Modern baseline closure: `{closure:.3e}` degC.",
                f"- Reopen recenter-equation closure: `{reopen_closure:.3e}` degC.",
                "- Locked period accessed: `false`.",
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
        json.dumps(
            {
                "sources_before": before,
                "sources_after": after,
                "source_states_unchanged": unchanged,
                "output": state(target),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("source-state-changed")
    if not summary["passes_acquisition_preflight"]:
        raise RuntimeError("round69-v2-consolidation-failed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
