"""Audit precipitation reconstruction provenance, units, coverage, and replay readiness."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import scipy.io

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ARCHIVE_ROOT = Path(
    r"D:\Users\JUNZ\Desktop\文件整理\1 论文\work1\compared_result_all"
)
PHYDA_ROOT = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
HISTORICAL_LOADER = Path(r"D:\Users\JUNZ\Desktop\select\PHYDA\load\load_S_720360.m")
CESM_ROOT = Path(r"D:\VM\gcm_7\cesm")
CLIMATE_NETCDF = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data"
    r"\climate_anomaly_05deg_with_ensemble_1000-2000.nc"
)
STRUCTURED_CLIMATE_MAT = Path(r"E:\2Data\climate_all_data_structured1.mat")


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


def to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [to_builtin(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: to_builtin(value) for key, value in row.items()} for row in rows])


def struct_dict(path: Path, variable: str) -> dict[str, Any]:
    data = scipy.io.loadmat(
        path,
        variable_names=[variable],
        squeeze_me=True,
        struct_as_record=False,
    )[variable]
    return {name: to_builtin(getattr(data, name)) for name in data._fieldnames}


def attrs_dict(dataset: h5py.Dataset) -> dict[str, Any]:
    return {str(key): to_builtin(value) for key, value in dataset.attrs.items()}


def finite_mask(dataset: h5py.Dataset, values: np.ndarray) -> np.ndarray:
    mask = np.isfinite(values)
    if "_FillValue" in dataset.attrs:
        fill = float(np.asarray(dataset.attrs["_FillValue"]).reshape(-1)[0])
        mask &= values != fill
    mask &= values > -9000
    return mask


def observation_coverage(handle: h5py.File) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    years = np.asarray(handle["time"], dtype=int)
    rows = []
    series: dict[str, np.ndarray] = {}
    for variable in ("pr_obs_cru_pre_05deg", "pr_obs_gpcp_05deg"):
        dataset = handle[variable]
        values = np.asarray(dataset, dtype=float)
        valid = finite_mask(dataset, values)
        counts = valid.reshape(valid.shape[0], -1).sum(axis=1)
        valid_years = years[counts > 0]
        valid_period = valid[counts > 0]
        complete_cells = int(np.all(valid_period, axis=0).sum()) if valid_years.size else 0
        finite_values = values[valid]
        rows.append(
            {
                "variable": variable,
                "units": attrs_dict(dataset).get("units"),
                "reference_period": attrs_dict(dataset).get("reference_period"),
                "data_type": attrs_dict(dataset).get("data_type"),
                "first_year_with_any_data": int(valid_years[0]) if valid_years.size else None,
                "last_year_with_any_data": int(valid_years[-1]) if valid_years.size else None,
                "valid_year_count": int(valid_years.size),
                "grid_cell_count": int(values.shape[1] * values.shape[2]),
                "maximum_valid_cells_in_one_year": int(counts.max()),
                "minimum_valid_cells_in_valid_years": int(counts[counts > 0].min()),
                "cells_complete_over_valid_years": complete_cells,
                "finite_value_count": int(valid.sum()),
                "minimum_anomaly_mm_day": float(finite_values.min()),
                "maximum_anomaly_mm_day": float(finite_values.max()),
            }
        )
        series[variable] = counts
    series["years"] = years
    return rows, series


def compact_prior_summary(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        group = handle["prior_fields"]
        output: dict[str, Any] = {
            "path": str(path),
            "sha256": sha256(path),
            "classification": "later evaluation export; not the fixed 336x848 member pool",
        }
        for name in ("temperature", "precipitation"):
            values = np.asarray(group[name], dtype=float)
            output[name] = {
                "hdf_shape": list(values.shape),
                "finite_count": int(np.isfinite(values).sum()),
                "minimum": float(np.nanmin(values)),
                "p01": float(np.nanpercentile(values, 1)),
                "median": float(np.nanmedian(values)),
                "mean": float(np.nanmean(values)),
                "p99": float(np.nanpercentile(values, 99)),
                "maximum": float(np.nanmax(values)),
                "zero_count": int(np.sum(values == 0)),
            }
        return output


def result_inventory(pr_root: Path) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows = []
    posterior_samples: dict[str, np.ndarray] = {}
    for dataset_id in (6, 9):
        network = f"proxy{dataset_id}"
        result_dir = pr_root / "results" / network
        individual = sorted(result_dir.glob("AOEnKF_ind_exp*.mat"))
        for method, file_name in (
            ("EnKF", "EnKF_results.mat"),
            ("AOEnKF", "AOEnKF_results.mat"),
            ("HGAOEnKF", "HGAOEnKF_results.mat"),
        ):
            path = result_dir / file_name
            with h5py.File(path, "r") as handle:
                status = np.asarray(handle["experiment_status"]).reshape(-1)
                row = {
                    "proxy_network": network,
                    "method": method,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "Xa_m_all_shape_hdf": json.dumps(list(handle["Xa_m_all"].shape)),
                    "Xa_sigma_all_shape_hdf": json.dumps(list(handle["Xa_sigma_all"].shape)),
                    "Xa_prct_all_shape_hdf": json.dumps(list(handle["Xa_prct_all"].shape)),
                    "experiment_count": int(status.size),
                    "status_nonzero_count": int(np.count_nonzero(status)),
                    "individual_ao_file_count": len(individual),
                }
                if method == "EnKF":
                    dataset = handle["Xa_m_all"]
                    blocks = []
                    total = 0.0
                    count = 0
                    minimum = np.inf
                    maximum = -np.inf
                    for start in range(0, dataset.shape[0], 5):
                        block = np.asarray(dataset[start : start + 5], dtype=float)
                        finite = block[np.isfinite(block)]
                        total += float(finite.sum())
                        count += int(finite.size)
                        minimum = min(minimum, float(finite.min()))
                        maximum = max(maximum, float(finite.max()))
                        if start == 0:
                            blocks.append(block.reshape(-1)[:: max(1, block.size // 50000)])
                    row.update(
                        {
                            "posterior_mean_minimum": minimum,
                            "posterior_mean_mean": total / count,
                            "posterior_mean_maximum": maximum,
                        }
                    )
                    posterior_samples[network] = np.concatenate(blocks)
                rows.append(row)
    return rows, posterior_samples


def loader_summary(path: Path, label: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "label": label,
        "path": str(path),
        "sha256": sha256(path),
        "last_write_time": path.stat().st_mtime,
        "accepts_hma_ranges_as_arguments": "hma_lat_range,hma_lon_range" in text.replace(" ", ""),
        "target_grid_180x90": "linspace(-179, 179, 180)" in text and "linspace(-89, 89, 90)" in text,
        "target_grid_720x360": "linspace(-179.75, 179.75, 720)" in text and "linspace(-89.75, 89.75, 360)" in text,
        "uses_merged_precc": "cesm_PRECC_merged.nc" in text,
        "uses_merged_precl": "cesm_PRECL_merged.nc" in text,
        "converts_m_s_to_mm_day": "86400 * 1000" in text,
        "uses_crucl11_pr": "load_obsclim('CRUCL11_pr'" in text,
        "uses_gpcp_active": bool(re.search(r"^\s*X_clim1\s*=\s*load_obsclim\('gpcp'", text, re.MULTILINE)),
        "replaces_nonpositive_with_gamma": "gamrnd(1, 0.05" in text,
        "removes_full_loaded_model_monthly_climatology": "mean(Xv_mon12, 4)" in text,
    }


def cesm_metadata(path: Path, variable: str) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        dataset = handle[variable]
        return {
            "path": str(path),
            "sha256": sha256(path),
            "variable": variable,
            "shape_hdf": list(dataset.shape),
            "units": attrs_dict(dataset).get("units"),
            "long_name": attrs_dict(dataset).get("long_name"),
            "time_count": int(handle["time"].shape[0]),
            "time_units": attrs_dict(handle["time"]).get("units"),
            "calendar": attrs_dict(handle["time"]).get("calendar"),
            "lat_count": int(handle["lat"].shape[0]),
            "lon_count": int(handle["lon"].shape[0]),
        }


def manifest(paths: dict[str, Path]) -> dict[str, Any]:
    output = {}
    for name, path in paths.items():
        output[name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    return output


def main() -> int:
    root = project_root()
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_provenance_audit_v1"
    )
    output.mkdir(parents=True, exist_ok=True)

    pr_root = ARCHIVE_ROOT / "fixed_proxy_experiments_69" / "pr"
    compact_prior = ARCHIVE_ROOT / "python_data" / "prior_fields.mat"
    current_loader = PHYDA_ROOT / "load_S_720360.m"
    fixed_driver = PHYDA_ROOT / "fix_proxys_experiments.m"
    evaluation_driver = PHYDA_ROOT / "evaluate_reconstructions_all.m"
    source_paths = {
        "archived_config": pr_root / "experiment_config.mat",
        "archived_indices": pr_root / "indices" / "all_Xb_indices.mat",
        "current_loader": current_loader,
        "historical_loader": HISTORICAL_LOADER,
        "fixed_driver": fixed_driver,
        "evaluation_driver": evaluation_driver,
        "compact_prior": compact_prior,
        "combined_climate_netcdf": CLIMATE_NETCDF,
        "structured_climate_mat": STRUCTURED_CLIMATE_MAT,
        "proxy6_enkf_result": pr_root / "results" / "proxy6" / "EnKF_results.mat",
        "proxy9_enkf_result": pr_root / "results" / "proxy9" / "EnKF_results.mat",
    }
    missing = [str(path) for path in source_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing-required-inputs:" + ";".join(missing))

    before = manifest(source_paths)
    config = struct_dict(source_paths["archived_config"], "config")
    with h5py.File(source_paths["archived_indices"], "r") as handle:
        indices = np.asarray(handle["all_Xb_indices"], dtype=int)
    index_summary = {
        "hdf_shape": list(indices.shape),
        "minimum": int(indices.min()),
        "maximum": int(indices.max()),
        "unique_across_all_experiments": int(np.unique(indices).size),
        "all_rows_have_500_unique_members": bool(
            indices.shape == (50, 500) and all(np.unique(row).size == 500 for row in indices)
        ),
    }

    with h5py.File(CLIMATE_NETCDF, "r") as handle:
        coverage_rows, coverage_series = observation_coverage(handle)
        precipitation_variables = [name for name in handle if name.startswith("pr_")]
        variable_rows = [
            {
                "variable": name,
                "shape_hdf": json.dumps(list(handle[name].shape)),
                "units": attrs_dict(handle[name]).get("units"),
                "reference_period": attrs_dict(handle[name]).get("reference_period"),
                "data_type": attrs_dict(handle[name]).get("data_type"),
            }
            for name in precipitation_variables
        ]

    prior = compact_prior_summary(compact_prior)
    inventory_rows, posterior_samples = result_inventory(pr_root)
    loader_rows = [
        loader_summary(HISTORICAL_LOADER, "historical_2025-03-06"),
        loader_summary(current_loader, "current_2025-05-26"),
    ]
    cesm_rows = [
        cesm_metadata(CESM_ROOT / "cesm_PRECC_merged.nc", "PRECC"),
        cesm_metadata(CESM_ROOT / "cesm_PRECL_merged.nc", "PRECL"),
    ]
    evaluation_text = evaluation_driver.read_text(encoding="utf-8", errors="replace")
    later_export_multiplies_precipitation_by_30 = bool(
        re.search(r"Xb_o_full_p\s*=\s*Xb_o_full_p\s*\*\s*30", evaluation_text)
    )

    cru = next(row for row in coverage_rows if row["variable"] == "pr_obs_cru_pre_05deg")
    gpcp = next(row for row in coverage_rows if row["variable"] == "pr_obs_gpcp_05deg")
    archived_pool_available = any(
        path.name.lower().endswith(("precipitation_prior_336x848.npy", "precipitation_prior_336x848.mat"))
        for path in root.rglob("*")
        if path.is_file()
    )
    conflicts = []
    if gpcp["first_year_with_any_data"] < 1979:
        conflicts.append(
            "pr_obs_gpcp_05deg is finite from "
            f"{gpcp['first_year_with_any_data']}, which is incompatible with direct GPCP "
            "satellite-era validation unless a documented extension or merge is recovered."
        )
    if not archived_pool_available:
        conflicts.append("No frozen 336x848 precipitation prior member pool was found.")
    if loader_rows[0]["sha256"] != loader_rows[1]["sha256"]:
        conflicts.append(
            "The surviving historical and current precipitation loaders differ in grid/signature; "
            "the exact May 17 loader snapshot is not recovered."
        )
    if later_export_multiplies_precipitation_by_30:
        conflicts.append(
            "prior_fields.precipitation is a later evaluation export multiplied by 30 and must not "
            "be substituted for the archived mm/day EnKF member pool."
        )

    provenance_gate = bool(
        config.get("num_experiments") == 50
        and config.get("num_samples_xb") == 500
        and config.get("psm_ot_list") == 7
        and config.get("r_o") == 1000
        and config.get("r_f") == 2000
        and index_summary["all_rows_have_500_unique_members"]
    )
    replay_ready = bool(provenance_gate and archived_pool_available and not conflicts)

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), constrained_layout=True)
    years = coverage_series["years"]
    axes[0].plot(years, coverage_series["pr_obs_cru_pre_05deg"], label="CRU", lw=1.8)
    axes[0].plot(years, coverage_series["pr_obs_gpcp_05deg"], label="GPCP label", lw=1.8)
    axes[0].set_xlim(1880, 2000)
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Finite 0.5-degree cells")
    axes[0].set_title("Stored observation coverage")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    with h5py.File(compact_prior, "r") as handle:
        compact_values = np.asarray(handle["prior_fields/precipitation"], dtype=float).reshape(-1)
    axes[1].hist(compact_values, bins=80, color="#2f7f5f", alpha=0.82)
    axes[1].set_xlim(0, np.nanpercentile(compact_values, 99.5))
    axes[1].set_xlabel("Stored precipitation value")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Later compact prior export (x30)")

    for network, values in posterior_samples.items():
        axes[2].hist(values, bins=70, alpha=0.48, density=True, label=network)
    axes[2].set_xlim(0, 8)
    axes[2].set_xlabel("Archived EnKF posterior mean (mm/day)")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Archived precipitation posterior sample")
    axes[2].legend(frameon=False)
    figure.savefig(output / "figure_precipitation_provenance_audit.png", dpi=220)
    plt.close(figure)

    write_csv(output / "observation_coverage.csv", coverage_rows)
    write_csv(output / "precipitation_variable_inventory.csv", variable_rows)
    write_csv(output / "archived_result_inventory.csv", inventory_rows)
    write_csv(output / "loader_comparison.csv", loader_rows)
    write_csv(output / "cesm_precipitation_source_inventory.csv", cesm_rows)

    summary = {
        "experiment_id": "climate_reconstruction_precipitation_provenance_audit_v1",
        "date": "2026-08-13",
        "no_oggm": True,
        "rec_ref_role": "comparison-only",
        "archived_configuration": config,
        "fixed_member_indices": index_summary,
        "precipitation_variable_count": len(variable_rows),
        "observation_coverage": coverage_rows,
        "compact_prior": prior,
        "later_export_multiplies_precipitation_by_30": later_export_multiplies_precipitation_by_30,
        "cesm_sources": cesm_rows,
        "loader_comparison": loader_rows,
        "archived_result_inventory": inventory_rows,
        "structured_climate_mat_top_variables": [list(item) for item in scipy.io.whosmat(STRUCTURED_CLIMATE_MAT)],
        "provenance_gate_passed": provenance_gate,
        "frozen_336x848_precipitation_prior_found": archived_pool_available,
        "replay_ready": replay_ready,
        "conflicts": conflicts,
        "decision": (
            "Do not begin precipitation EnKF replay or CRU/GPCP validation. First recover or rebuild "
            "a frozen 336x848 mm/day precipitation prior in MATLAB R2024b, demonstrate closure against "
            "the archived compact chronology without using its x30 export as the member pool, and resolve "
            "the pre-satellite GPCP-labelled coverage provenance."
        ),
    }
    after = manifest(source_paths)
    hashes_unchanged = before == after
    summary["source_hashes_unchanged"] = hashes_unchanged
    (output / "input_manifest_before.json").write_text(
        json.dumps(before, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "input_manifest_after.json").write_text(
        json.dumps(after, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(to_builtin(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = [
        "# Precipitation Provenance Audit Result",
        "",
        f"- Provenance gate passed: `{provenance_gate}`",
        f"- Frozen 336 x 848 precipitation prior found: `{archived_pool_available}`",
        f"- Ready for member-level EnKF replay: `{replay_ready}`",
        f"- Source hashes unchanged: `{hashes_unchanged}`",
        f"- CRU stored coverage: {cru['first_year_with_any_data']}-{cru['last_year_with_any_data']}, "
        f"{cru['cells_complete_over_valid_years']} complete cells",
        f"- GPCP-labelled stored coverage: {gpcp['first_year_with_any_data']}-{gpcp['last_year_with_any_data']}, "
        f"{gpcp['cells_complete_over_valid_years']} complete cells",
        "",
        "## Conflicts",
        "",
        *[f"- {item}" for item in conflicts],
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(to_builtin(summary), indent=2, ensure_ascii=False))
    if not hashes_unchanged:
        raise RuntimeError("source-purity-failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

