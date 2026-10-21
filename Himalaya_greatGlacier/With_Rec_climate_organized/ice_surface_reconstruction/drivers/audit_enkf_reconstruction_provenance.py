from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.io


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ARCHIVE_ROOT = Path(
    r"D:\Users\JUNZ\Desktop\文件整理\1 论文\work1\compared_result_all"
)
DEFAULT_DATA_ROOT = Path(r"E:\2Data")
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "enkf_reconstruction_provenance_audit_v1"
)

METHOD_FILE_NAMES = {
    "AOEnKF": "AOEnKF_results.mat",
    "OEnKF": "EnKF_results.mat",
    "HGAOEnKF": "HGAOEnKF_results.mat",
}


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_to_builtin(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{k: _to_builtin(v) for k, v in row.items()} for row in rows])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_struct(path: Path, variable: str = "config") -> Any:
    data = scipy.io.loadmat(
        path,
        variable_names=[variable],
        squeeze_me=True,
        struct_as_record=False,
    )
    return data[variable]


def _struct_dict(struct: Any) -> dict[str, Any]:
    return {name: _to_builtin(getattr(struct, name)) for name in struct._fieldnames}


def _decode_char(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset).ravel(order="F")
    return "".join(chr(int(value)) for value in values if int(value) != 0)


def _scalar(group: h5py.Group, name: str) -> float:
    return float(np.asarray(group[name]).reshape(-1)[0])


def _summarize_prior(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        group = handle["prior_fields"]
        temperature = group["temperature"]
        precipitation = group["precipitation"]
        xmeta_ref = group["xmeta"][0, 0]
        xmeta = handle[xmeta_ref]
        lat = np.asarray(xmeta["lat"]).reshape(-1)
        lon = np.asarray(xmeta["lon"]).reshape(-1)
        return {
            "path": str(path),
            "sha256": _sha256(path),
            "temperature_shape_hdf": list(temperature.shape),
            "precipitation_shape_hdf": list(precipitation.shape),
            "temperature_finite": int(np.isfinite(temperature[()]).sum()),
            "precipitation_finite": int(np.isfinite(precipitation[()]).sum()),
            "xmeta_varnm": _decode_char(xmeta["varnm"]),
            "xmeta_varnmlng": _decode_char(xmeta["varnmlng"]),
            "lat_count": int(lat.size),
            "lat_min": float(np.nanmin(lat)),
            "lat_max": float(np.nanmax(lat)),
            "lon_count": int(lon.size),
            "lon_min": float(np.nanmin(lon)),
            "lon_max": float(np.nanmax(lon)),
            "grid_cell_count": int(lat.size * lon.size),
            "id_X": np.asarray(xmeta["id_X"]).reshape(-1).tolist(),
        }


def _proxy_record(handle: h5py.File, reference: h5py.Reference, index: int) -> dict[str, Any]:
    group = handle[reference]
    location = np.asarray(group["location"]).reshape(-1, order="F")
    coefficients = np.asarray(group["b"]).reshape(-1, order="F")
    data = np.asarray(group["data"]).reshape(-1, order="F")
    age_yng = np.asarray(group["age_yng"]).reshape(-1, order="F")
    finite = np.isfinite(data)
    finite_ages = age_yng[finite] if age_yng.size == data.size else np.array([])
    return {
        "proxy_index_1based": index + 1,
        "latitude": float(location[0]),
        "longitude": float(location[1]),
        "proxy_type": _decode_char(group["ptype"]),
        "psm_method": _decode_char(group["psm_mthd"]),
        "R": _scalar(group, "R"),
        "pcr": _scalar(group, "pcr"),
        "pcr_t": _scalar(group, "pcr_t"),
        "pcr_p": _scalar(group, "pcr_p"),
        "pval": _scalar(group, "pval"),
        "b0": float(coefficients[0]) if coefficients.size > 0 else np.nan,
        "b1": float(coefficients[1]) if coefficients.size > 1 else np.nan,
        "b2": float(coefficients[2]) if coefficients.size > 2 else np.nan,
        "data_length": int(data.size),
        "data_finite_count": int(finite.sum()),
        "first_finite_age": float(np.nanmin(finite_ages)) if finite_ages.size else np.nan,
        "last_finite_age": float(np.nanmax(finite_ages)) if finite_ages.size else np.nan,
        "Rvec_length": int(group["Rvec"].size),
        "HXb_length": int(group["HXb"].size),
        "M_t_length": int(group["M_t"].size),
        "M_p_length": int(group["M_p"].size),
    }


def _summarize_proxy_file(path: Path, dataset_id: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with h5py.File(path, "r") as handle:
        group = handle["proxy_data"]
        total = int(np.asarray(group["num_proxies"]).reshape(-1)[0])
        references = np.asarray(group["full_proxy"]).reshape(-1, order="F")
        temp_indices = np.asarray(group["temp_proxy_indices"], dtype=int)
        precip_indices = np.asarray(group["precip_proxy_indices"], dtype=int)
        records = [_proxy_record(handle, reference, index) for index, reference in enumerate(references)]

    all_indices = set(range(1, total + 1))
    holdout_frequency: Counter[int] = Counter()
    split_rows = []
    for repeat, row in enumerate(temp_indices, start=1):
        train = [int(value) for value in row if int(value) > 0]
        holdout = sorted(all_indices.difference(train))
        holdout_frequency.update(holdout)
        split_rows.append(
            {
                "repeat": repeat,
                "train_count": len(train),
                "train_unique_count": len(set(train)),
                "holdout_count": len(holdout),
                "indices_in_bounds": all(1 <= value <= total for value in train),
            }
        )

    ptype_counts = Counter(record["proxy_type"] for record in records)
    psm_counts = Counter(record["psm_method"] for record in records)
    r_values = np.array([record["R"] for record in records], dtype=float)
    pcr_values = np.array([record["pcr"] for record in records], dtype=float)
    finite_counts = np.array([record["data_finite_count"] for record in records], dtype=float)
    holdout_counts = np.array([holdout_frequency[index] for index in range(1, total + 1)], dtype=int)

    summary = {
        "dataset_id": dataset_id,
        "path": str(path),
        "sha256": _sha256(path),
        "total_proxies": total,
        "stored_proxy_references": len(records),
        "split_repeats": int(temp_indices.shape[0]),
        "temperature_train_count_per_repeat": int(temp_indices.shape[1]),
        "precipitation_train_count_per_repeat": int(precip_indices.shape[1]),
        "holdout_count_per_repeat": int(total - temp_indices.shape[1]),
        "training_fraction": float(temp_indices.shape[1] / total),
        "temperature_precipitation_splits_identical": bool(np.array_equal(temp_indices, precip_indices)),
        "all_split_indices_valid": bool(all(row["indices_in_bounds"] for row in split_rows)),
        "all_split_indices_unique": bool(
            all(row["train_count"] == row["train_unique_count"] for row in split_rows)
        ),
        "holdout_frequency_min": int(holdout_counts.min()),
        "holdout_frequency_mean": float(holdout_counts.mean()),
        "holdout_frequency_max": int(holdout_counts.max()),
        "never_held_out_count": int((holdout_counts == 0).sum()),
        "proxy_type_counts": dict(sorted(ptype_counts.items())),
        "psm_method_counts": dict(sorted(psm_counts.items())),
        "R_min": float(np.nanmin(r_values)),
        "R_median": float(np.nanmedian(r_values)),
        "R_max": float(np.nanmax(r_values)),
        "pcr_min": float(np.nanmin(pcr_values)),
        "pcr_median": float(np.nanmedian(pcr_values)),
        "pcr_max": float(np.nanmax(pcr_values)),
        "proxy_data_finite_count_min": int(np.nanmin(finite_counts)),
        "proxy_data_finite_count_median": float(np.nanmedian(finite_counts)),
        "proxy_data_finite_count_max": int(np.nanmax(finite_counts)),
    }
    for record in records:
        record["dataset_id"] = dataset_id
        record["holdout_frequency_across_10_repeats"] = holdout_frequency[
            record["proxy_index_1based"]
        ]
    return summary, records


def _fixed_result_inventory(archive_root: Path) -> list[dict[str, Any]]:
    rows = []
    result_root = archive_root / "fixed_proxy_experiments_69" / "tas" / "results"
    for dataset_id in (6, 9):
        for method, file_name in METHOD_FILE_NAMES.items():
            path = result_root / f"proxy{dataset_id}" / file_name
            with h5py.File(path, "r") as handle:
                status = np.asarray(handle["experiment_status"]).reshape(-1)
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "method": method,
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                        "Xa_m_all_shape_hdf": list(handle["Xa_m_all"].shape),
                        "Xa_prct_all_shape_hdf": list(handle["Xa_prct_all"].shape),
                        "Xa_sigma_all_shape_hdf": list(handle["Xa_sigma_all"].shape),
                        "experiment_status_shape_hdf": list(handle["experiment_status"].shape),
                        "experiment_count": int(status.size),
                        "status_unique": sorted(np.unique(status).tolist()),
                        "status_nonzero_count": int(np.count_nonzero(status)),
                    }
                )
    return rows


def _read_validation_summary(archive_root: Path) -> list[dict[str, Any]]:
    path = archive_root / "save_result" / "evaluation_results" / "temperature_reconstruction_summary.csv"
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["source_path"] = str(path)
    return rows


def _verify_lineage(archive_root: Path, data_root: Path) -> list[dict[str, Any]]:
    combined_path = data_root / "ENKF_Three_ensemble50.mat"
    export_path = data_root / "5rec_result" / "tas" / "tas_result_matlab_for_py.mat"
    export_struct = scipy.io.loadmat(
        export_path,
        variable_names=["export_data_for_python"],
        struct_as_record=True,
        squeeze_me=False,
    )["export_data_for_python"][0, 0]

    rows = []
    result_root = archive_root / "fixed_proxy_experiments_69" / "tas" / "results"
    for dataset_id in (6, 9):
        for method, file_name in METHOD_FILE_NAMES.items():
            result_var = f"{method}_results_{dataset_id}"
            summary_var = f"{method}_{dataset_id}"
            combined = scipy.io.loadmat(
                combined_path,
                variable_names=[result_var, summary_var],
                struct_as_record=True,
                squeeze_me=False,
            )
            result_struct = combined[result_var][0, 0]
            summary_struct = combined[summary_var][0, 0]
            combined_members = np.asarray(result_struct["Xa_m_all"], dtype=float)
            combined_mean = np.asarray(summary_struct["Xa_m_all"], dtype=float)
            combined_status = np.asarray(result_struct["experiment_status"]).reshape(-1)

            fixed_path = result_root / f"proxy{dataset_id}" / file_name
            with h5py.File(fixed_path, "r") as handle:
                fixed_members_hdf = np.asarray(handle["Xa_m_all"], dtype=float)
                fixed_status = np.asarray(handle["experiment_status"]).reshape(-1)

            fixed_members_matlab = np.transpose(fixed_members_hdf, (2, 1, 0))
            fixed_mean_matlab = np.nanmean(fixed_members_hdf, axis=0).T
            export_field = f"{method}_{dataset_id}_Xa_m"
            exported_data = np.asarray(export_struct[export_field][0, 0]["data"], dtype=float)
            exported_as_state = exported_data.reshape((336, 1001), order="F")

            rows.append(
                {
                    "dataset_id": dataset_id,
                    "method": method,
                    "fixed_to_combined_members_max_abs_diff": float(
                        np.nanmax(np.abs(fixed_members_matlab - combined_members))
                    ),
                    "fixed_mean_to_combined_summary_max_abs_diff": float(
                        np.nanmax(np.abs(fixed_mean_matlab - combined_mean))
                    ),
                    "combined_summary_to_python_export_max_abs_diff": float(
                        np.nanmax(np.abs(exported_as_state - combined_mean))
                    ),
                    "fixed_and_combined_status_equal": bool(
                        np.array_equal(fixed_status, combined_status)
                    ),
                    "fixed_member_shape_matlab": list(fixed_members_matlab.shape),
                    "combined_member_shape": list(combined_members.shape),
                    "combined_summary_shape": list(combined_mean.shape),
                    "exported_data_shape": list(exported_data.shape),
                }
            )
            del combined, result_struct, summary_struct
            del combined_members, combined_mean, fixed_members_hdf, fixed_members_matlab
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-lineage",
        action="store_true",
        help="Skip the memory-intensive fixed-result to combined-MAT numerical comparison.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fixed_config_path = (
        args.archive_root
        / "fixed_proxy_experiments_69"
        / "tas"
        / "experiment_config.mat"
    )
    validation_config_path = args.archive_root / "python_data" / "config.mat"
    fixed_config = _struct_dict(_load_struct(fixed_config_path))
    validation_config = _struct_dict(_load_struct(validation_config_path))

    prior_summary = _summarize_prior(args.archive_root / "python_data" / "prior_fields.mat")
    proxy_summaries = []
    proxy_records = []
    for dataset_id in (6, 9):
        summary, records = _summarize_proxy_file(
            args.archive_root / "python_data" / f"proxy_data_{dataset_id}.mat",
            dataset_id,
        )
        proxy_summaries.append(summary)
        proxy_records.extend(records)

    result_inventory = _fixed_result_inventory(args.archive_root)
    validation_summary = _read_validation_summary(args.archive_root)
    lineage_rows = [] if args.skip_lineage else _verify_lineage(args.archive_root, args.data_root)

    config_rows = []
    for config_name, path, config in (
        ("fixed_reconstruction", fixed_config_path, fixed_config),
        ("heldout_validation", validation_config_path, validation_config),
    ):
        for key, value in config.items():
            config_rows.append(
                {
                    "config_name": config_name,
                    "source_path": str(path),
                    "parameter": key,
                    "value": json.dumps(_to_builtin(value), ensure_ascii=False),
                }
            )

    _write_csv(args.output_dir / "configuration_table.csv", config_rows)
    _write_csv(args.output_dir / "proxy_dataset_summary.csv", proxy_summaries)
    _write_csv(args.output_dir / "proxy_records.csv", proxy_records)
    _write_csv(args.output_dir / "fixed_result_inventory.csv", result_inventory)
    _write_csv(args.output_dir / "validation_metrics_summary.csv", validation_summary)
    _write_csv(args.output_dir / "lineage_comparison.csv", lineage_rows)

    audit = {
        "fixed_reconstruction_config": fixed_config,
        "heldout_validation_config": validation_config,
        "prior_summary": prior_summary,
        "proxy_dataset_summaries": proxy_summaries,
        "fixed_result_inventory": result_inventory,
        "validation_metrics_summary": validation_summary,
        "lineage_comparison": lineage_rows,
        "reproducibility_gaps": [
            "No MATLAB assimilation source files were found in the archived result tree.",
            "No random seed or saved RNG state is present in the discovered config files.",
            "No explicit localization radius or inflation parameter is present in the discovered config files.",
            "The archived results preserve outputs and proxy splits, but not enough information for a bitwise rerun.",
        ],
    }
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(_to_builtin(audit), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(_to_builtin(audit), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
