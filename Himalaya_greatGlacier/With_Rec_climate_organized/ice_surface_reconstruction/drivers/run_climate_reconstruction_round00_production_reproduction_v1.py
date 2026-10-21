"""Round 00: close the legacy six-member production lineage without locked years."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import netCDF4
import numpy as np
import rasterio


EXPERIMENT_ID = "climate_reconstruction_improvement_loop_v1/round_00_production_reproduction"
NETCDF = Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc")
TIF_ROOT = Path(r"E:\2Data\6compared_result\Rec_result")
STRUCTURED_MAT = Path(r"E:\2Data\climate_all_data_structured1.mat")
BUILDER = Path(r"E:\3spyderProject\HMA_500\Down_scale_new\mat_climate_anomaly_visualizer_v1.py")
ASSEMBLER = Path(r"E:\3spyderProject\HMA_500\HMA_500\no_compared_result.py")
LOCKED_START = 1921
LOCKED_END = 1939
REFERENCE_START = 1961
REFERENCE_END = 1990
MEMBER_TOLERANCE = 5e-5
ENSEMBLE_TOLERANCE = 2e-6

METHODS = {"AO": "AOEnKF", "HGAO": "HGAOEnKF", "O": "OEnKF"}


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stat(path: Path) -> dict[str, object]:
    value = path.stat()
    return {"path": str(path), "size_bytes": int(value.st_size), "mtime_ns": int(value.st_mtime_ns)}


def clean(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    result[(result <= -9000) | (result >= 1e20)] = np.nan
    return result


def finite_mean(values: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(values, axis=0)
    finite = np.isfinite(stack)
    count = finite.sum(axis=0)
    total = np.where(finite, stack, 0.0).sum(axis=0)
    return np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)


def compare(expected: np.ndarray, archived: np.ndarray) -> dict[str, object]:
    expected_finite = np.isfinite(expected)
    archived_finite = np.isfinite(archived)
    common = expected_finite & archived_finite
    if not np.any(common):
        raise RuntimeError("no-finite-values-for-closure")
    delta = expected[common] - archived[common]
    return {
        "finite_pair_count": int(common.sum()),
        "finite_mask_mismatch_count": int(np.count_nonzero(expected_finite != archived_finite)),
        "maximum_absolute_error": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "mean_error": float(np.mean(delta)),
    }


def source_tif(variable: str, method: str, network: int) -> Path:
    return TIF_ROOT / variable / "05degree" / f"{METHODS[method]}_{network}_Xa_m_clipped_0.5deg.tif"


def member_variable(variable: str, method: str, network: int) -> str:
    return f"{variable}_rec_result_{method}_{network}_m_c05d"


def read_tif_allowed(path: Path, band_indexes: list[int]) -> tuple[np.ndarray, dict[str, object]]:
    with rasterio.open(path) as dataset:
        if dataset.count != 1001:
            raise RuntimeError(f"unexpected-band-count:{path}:{dataset.count}")
        values = clean(dataset.read(band_indexes))
        metadata = {
            "driver": dataset.driver,
            "shape": [int(dataset.height), int(dataset.width), int(dataset.count)],
            "dtype": str(dataset.dtypes[0]),
            "nodata": None if dataset.nodata is None else float(dataset.nodata),
            "crs": None if dataset.crs is None else str(dataset.crs),
            "transform": [float(value) for value in dataset.transform],
        }
    return values, metadata


def main() -> int:
    root = project_root()
    output = root / "validation_data/results/paleo_surface_validation_v0" / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")

    facts = root / "LIA_PROJECT_FACTS.md"
    lock_contract = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_instrumental_baseline_v1/LOCKED_FINAL_PERIOD.json"
    plan = root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/climate_reconstruction_improvement_loop_v1.md"
    driver = Path(__file__).resolve()
    members = [(variable, method, network) for variable in ("tas", "pr") for method in METHODS for network in (6, 9)]
    tifs = [source_tif(*member) for member in members]
    required = [NETCDF, STRUCTURED_MAT, BUILDER, ASSEMBLER, facts, lock_contract, plan, *tifs]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing-round00-inputs:" + ";".join(missing))

    locked = json.loads(lock_contract.read_text(encoding="utf-8"))
    if locked.get("status") != "locked_not_evaluated":
        raise RuntimeError("locked-period-contract-failed")

    source_stats_before = {str(path): stat(path) for path in required}
    content_hashes = {str(path): sha256(path) for path in [NETCDF, BUILDER, ASSEMBLER, *tifs]}

    with netCDF4.Dataset(NETCDF, "r") as dataset:
        years = np.asarray(dataset.variables["time"][:], dtype=int)
        if not np.array_equal(years, np.arange(1000, 2001, dtype=int)):
            raise RuntimeError("unexpected-production-time-axis")
        allowed_mask = (years < LOCKED_START) | (years > LOCKED_END)
        allowed_indices = np.flatnonzero(allowed_mask)
        allowed_years = years[allowed_indices]
        if np.any((allowed_years >= LOCKED_START) & (allowed_years <= LOCKED_END)):
            raise RuntimeError("locked-period-requested")
        reference_mask = (allowed_years >= REFERENCE_START) & (allowed_years <= REFERENCE_END)
        if int(reference_mask.sum()) != 30:
            raise RuntimeError("reference-period-incomplete")
        band_indexes = [int(year - 999) for year in allowed_years]

        latitudes = np.asarray(dataset.variables["lat"][:], dtype=float)
        longitudes = np.asarray(dataset.variables["lon"][:], dtype=float)
        if latitudes.shape != (43,) or longitudes.shape != (78,):
            raise RuntimeError(f"unexpected-production-grid:{latitudes.shape}:{longitudes.shape}")

        archived_members: dict[str, dict[str, np.ndarray]] = {"tas": {}, "pr": {}}
        lineage_rows: list[dict[str, object]] = []

        for variable, method, network in members:
            label = f"{method}_{network}"
            tif_path = source_tif(variable, method, network)
            source_values, tif_metadata = read_tif_allowed(tif_path, band_indexes)
            source_anomaly = source_values - np.nanmean(source_values[reference_mask], axis=0)[None, :, :]
            variable_name = member_variable(variable, method, network)
            archived = clean(dataset.variables[variable_name][allowed_indices, :, :])
            direct = compare(source_anomaly, archived)
            flipped = compare(source_anomaly[:, ::-1, :], archived)
            if direct["maximum_absolute_error"] <= flipped["maximum_absolute_error"]:
                orientation, closure = "as_stored", direct
            else:
                orientation, closure = "latitude_flipped", flipped
            if closure["maximum_absolute_error"] > MEMBER_TOLERANCE or closure["finite_mask_mismatch_count"] != 0:
                raise RuntimeError(f"member-closure-failed:{variable}:{label}:{closure}")
            archived_members[variable][label] = archived
            reference_residual = np.nanmean(archived[reference_mask], axis=0)
            lineage_rows.append({
                "variable": variable,
                "member": label,
                "method": method,
                "proxy_network": network,
                "source_tif": str(tif_path),
                "source_tif_sha256": content_hashes[str(tif_path)],
                "netcdf_variable": variable_name,
                "orientation": orientation,
                "source_shape": "x".join(str(value) for value in tif_metadata["shape"]),
                "allowed_year_count": int(len(allowed_years)),
                "locked_year_count_skipped": LOCKED_END - LOCKED_START + 1,
                "maximum_absolute_error": closure["maximum_absolute_error"],
                "rmse": closure["rmse"],
                "finite_mask_mismatch_count": closure["finite_mask_mismatch_count"],
                "reference_period_grid_mean_abs_max": float(np.nanmax(np.abs(reference_residual))),
            })

        closure_rows: list[dict[str, object]] = []
        for variable in ("tas", "pr"):
            for network in (6, 9):
                labels = [f"{method}_{network}" for method in METHODS]
                rebuilt = finite_mean([archived_members[variable][label] for label in labels])
                archived = clean(dataset.variables[f"{variable}_ensemble_mean_{network}"][allowed_indices, :, :])
                closure_rows.append({"variable": variable, "ensemble": f"network_{network}", **compare(rebuilt, archived)})
            rebuilt = finite_mean([archived_members[variable][f"{method}_{network}"] for method in METHODS for network in (6, 9)])
            archived = clean(dataset.variables[f"{variable}_ensemble_mean_rec_result"][allowed_indices, :, :])
            closure_rows.append({"variable": variable, "ensemble": "six_member", **compare(rebuilt, archived)})

    if max(row["maximum_absolute_error"] for row in closure_rows) > ENSEMBLE_TOLERANCE:
        raise RuntimeError(f"ensemble-closure-failed:{closure_rows}")
    if any(row["finite_mask_mismatch_count"] != 0 for row in closure_rows):
        raise RuntimeError(f"ensemble-mask-closure-failed:{closure_rows}")

    source_stats_after = {str(path): stat(path) for path in required}
    if source_stats_before != source_stats_after:
        raise RuntimeError("source-changed-during-round00")

    output.mkdir(parents=True)
    config = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "read_only_reproduction_audit",
        "allowed_periods": [[1000, 1920], [1940, 2000]],
        "locked_period": [LOCKED_START, LOCKED_END],
        "reference_period": [REFERENCE_START, REFERENCE_END],
        "member_tolerance": MEMBER_TOLERANCE,
        "ensemble_tolerance": ENSEMBLE_TOLERANCE,
        "python_executable": r"D:\Anaconda\envs\oggm_env2\python.exe",
        "oggm_launched": False,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({
        "before": source_stats_before,
        "after": source_stats_after,
        "content_sha256": content_hashes,
        "driver_sha256": sha256(driver),
        "structured_mat_note": "Metadata only; direct closure uses 12 source GeoTIFFs and skips locked bands.",
    }, indent=2), encoding="utf-8")
    with (output / "member_lineage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(lineage_rows[0]))
        writer.writeheader()
        writer.writerows(lineage_rows)
    with (output / "closure_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(closure_rows[0]))
        writer.writeheader()
        writer.writerows(closure_rows)

    max_member_error = max(float(row["maximum_absolute_error"]) for row in lineage_rows)
    max_ensemble_error = max(float(row["maximum_absolute_error"]) for row in closure_rows)
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "round00_allowed_years_passed",
        "member_count": len(lineage_rows),
        "allowed_year_count": int(len(allowed_years)),
        "locked_1921_1939_accessed": False,
        "maximum_member_source_to_netcdf_error": max_member_error,
        "maximum_ensemble_rebuild_error": max_ensemble_error,
        "all_finite_masks_close": all(int(row["finite_mask_mismatch_count"]) == 0 for row in lineage_rows + closure_rows),
        "all_declared_sources_unchanged": source_stats_before == source_stats_after,
        "gate_00": "pass_except_predeclared_locked_period_deferred_to_gate_5b",
        "next_round": "round_01_blocked_cv_baseline",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "ROUND_CONTRACT.md").write_text(
        "# Round 00 Contract\n\nDecision: determine whether the archived six-member temperature and precipitation product is reproducible enough to be the v2 baseline.\n\nThe driver reads only 1000-1920 and 1940-2000. It never reads 1921-1939 values.\n",
        encoding="utf-8",
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Round 00 Production Reproduction\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Source-to-member maximum error: `{max_member_error:.9g}`\n"
        f"- Ensemble maximum error: `{max_ensemble_error:.9g}`\n"
        f"- Allowed years checked: `{len(allowed_years)}`\n"
        "- Locked 1921-1939 values accessed: `false`\n"
        "- Decision: allowed-year legacy product passes Gate 00; locked closure remains Gate 5B.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
