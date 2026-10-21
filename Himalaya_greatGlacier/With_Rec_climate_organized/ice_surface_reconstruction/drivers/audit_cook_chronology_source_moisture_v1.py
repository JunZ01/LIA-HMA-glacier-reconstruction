from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat


EXPERIMENT_ID = "climate_reconstruction_cook_chronology_source_moisture_v1"
TARGET_YEARS = np.arange(1905, 1921, dtype=int)
CALIBRATION_YEARS = np.arange(1941, 2001, dtype=int)
BASELINE_MASK = (TARGET_YEARS >= 1905) & (TARGET_YEARS <= 1914)
EVENT_MASK = (TARGET_YEARS >= 1915) & (TARGET_YEARS <= 1918)
PAGES_TS = Path(
    r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master\proxy-data"
    r"\PAGES2k_v2\proxy-data\pages2k_proxy\Pages2kTSv1_13_1.mat"
)
LMR_META = Path(
    r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master\proxy-data"
    r"\LMR_NCDC_database_v.0.2.0\lmr_meta.csv"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot-load-module:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest(path),
    }


def text_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bytes):
        return value.decode("latin-1").strip()
    array = np.asarray(value)
    if array.size == 0:
        return ""
    if array.dtype.kind in "US":
        return " ".join(str(item).strip() for item in array.reshape(-1) if str(item).strip()).strip()
    if array.dtype == object:
        parts = [text_value(item) for item in array.reshape(-1)]
        return " ".join(part for part in parts if part).strip()
    if array.size <= 24:
        return " ".join(str(item) for item in array.reshape(-1)).strip()
    return ""


def clean_text(value: object) -> str:
    text = text_value(value)
    return text if text and text.lower() not in {"nan", "none"} else "<missing>"


def extract_proxy_series(proxy_path: Path, proxy_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target = np.full((len(TARGET_YEARS), len(proxy_indices)), np.nan)
    calibration = np.full((len(CALIBRATION_YEARS), len(proxy_indices)), np.nan)
    with h5py.File(proxy_path, "r") as handle:
        refs = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
        for column, proxy_index in enumerate(proxy_indices):
            proxy = handle[refs[int(proxy_index) - 1]]
            years = np.asarray(proxy["age_yng"], dtype=int).reshape(-1, order="F")
            values = np.asarray(proxy["data"], dtype=float).reshape(-1, order="F")
            lookup = {int(year): float(value) for year, value in zip(years, values)}
            target[:, column] = [lookup.get(int(year), np.nan) for year in TARGET_YEARS]
            calibration[:, column] = [lookup.get(int(year), np.nan) for year in CALIBRATION_YEARS]
    return target, calibration


def pairwise_summary(matrix: np.ndarray) -> dict[str, float]:
    correlation = np.corrcoef(matrix, rowvar=False)
    upper = correlation[np.triu_indices_from(correlation, k=1)]
    eigenvalues = np.linalg.eigvalsh(correlation)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    participation = float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum())
    mean_r = float(np.mean(upper))
    classical_neff = float(matrix.shape[1] / (1.0 + (matrix.shape[1] - 1.0) * max(mean_r, 0.0)))
    return {
        "mean_pairwise_correlation": mean_r,
        "median_pairwise_correlation": float(np.median(upper)),
        "minimum_pairwise_correlation": float(np.min(upper)),
        "maximum_pairwise_correlation": float(np.max(upper)),
        "spectral_effective_dimension": participation,
        "classical_effective_record_count": classical_neff,
    }


def period_change(matrix: np.ndarray) -> np.ndarray:
    return np.mean(matrix[EVENT_MASK], axis=0) - np.mean(matrix[BASELINE_MASK], axis=0)


def publication_group(row: pd.Series) -> str:
    cite_key = clean_text(row.get("publication_cite_key", ""))
    if cite_key != "<missing>":
        return cite_key
    name = clean_text(row.get("pages_reconstruction_name", ""))
    return name.rsplit(".", 1)[-1] if name != "<missing>" else "<missing>"


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    drivers = Path(__file__).resolve().parent
    base = load_module(drivers / "audit_positive_b2_event_seasonal_psm_v1.py", "cook_source_paths")
    result_root = root / "validation_data/results/paleo_surface_validation_v0"
    output = result_root / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")

    sources = {
        "study_metadata": result_root / "climate_reconstruction_pages2k_fixed_network_study_batch_v1/fixed_network_study_metadata.csv",
        "proxy_archive": Path(base.archive_proxy_path()),
        "pages_ts": PAGES_TS,
        "lmr_metadata": LMR_META,
        "target_components": result_root / "climate_reconstruction_clean_gpcc_psm_rolling_pseudo_event_fixed_network_v1/fixed_network_target_1905_components.csv",
        "gpcc_quality": result_root / "climate_reconstruction_clean_gpcc_v2022_positive_b2_psm_v2/clean_gpcc_quality_at_proxy_sites.csv",
        "facts": root / "LIA_PROJECT_FACTS.md",
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: source_state(path) for name, path in sources.items()}

    metadata = pd.read_csv(sources["study_metadata"])
    cook = metadata[metadata["data_author"].eq("Cook, E.R.")].sort_values("proxy_index").reset_index(drop=True)
    if len(cook) != 15:
        raise RuntimeError(f"cook-count-contract-failed:{len(cook)}")
    cook["publication_group"] = cook.apply(publication_group, axis=1)
    proxy_indices = cook["proxy_index"].to_numpy(dtype=int)

    target, calibration = extract_proxy_series(sources["proxy_archive"], proxy_indices)
    if not np.isfinite(target).all() or not np.isfinite(calibration).all():
        raise RuntimeError("cook-series-not-complete-in-declared-periods")
    calibration_mean = calibration.mean(axis=0)
    calibration_scale = calibration.std(axis=0)
    target_z = (target - calibration_mean) / calibration_scale
    calibration_z = (calibration - calibration_mean) / calibration_scale
    event_change = period_change(target_z)
    closure = np.max(np.abs(event_change - cook["event_change_z"].to_numpy(dtype=float)))
    if closure > 1e-10:
        raise RuntimeError(f"event-change-closure-failed:{closure}")

    _, singular, vectors_t = np.linalg.svd(calibration_z, full_matrices=False)
    loadings = vectors_t[0].copy()
    if float(np.mean(loadings)) < 0:
        loadings *= -1.0
    pc1_variance = float(singular[0] ** 2 / np.square(singular).sum())
    event_projection = float(np.dot(event_change, loadings)) * loadings
    event_energy_fraction = float(np.square(event_projection).sum() / np.square(event_change).sum())
    target_pairwise = pairwise_summary(target_z)
    calibration_pairwise = pairwise_summary(calibration_z)

    record_rows = []
    for column, row in cook.iterrows():
        record_rows.append({
            **row.to_dict(),
            "calibration_pc1_loading": float(loadings[column]),
            "event_change_z_recomputed": float(event_change[column]),
            "event_change_negative": bool(event_change[column] < 0),
            "event_vector_pc1_projection_z": float(event_projection[column]),
            "event_vector_pc1_residual_z": float(event_change[column] - event_projection[column]),
        })
    records = pd.DataFrame(record_rows)

    annual = pd.DataFrame({"year": TARGET_YEARS})
    for column, row in cook.iterrows():
        label = clean_text(row["site_name"])
        if label == "<missing>":
            label = str(row["official_dataset_name"])
        annual[f"proxy_{int(row['proxy_index'])}_{label}"] = target_z[:, column]
    annual["cook_record_mean_z"] = target_z.mean(axis=1)
    annual["cook_record_median_z"] = np.median(target_z, axis=1)

    publication_rows = []
    for group, subset in records.groupby("publication_group", dropna=False):
        values = subset["event_change_z_recomputed"].to_numpy(dtype=float)
        retained = records[~records["publication_group"].eq(group)]["event_change_z_recomputed"].to_numpy(dtype=float)
        publication_rows.append({
            "publication_group": group,
            "record_count": int(len(subset)),
            "unique_site_count": int(subset["site_id"].nunique()),
            "mean_event_change_z": float(values.mean()),
            "negative_event_fraction": float(np.mean(values < 0)),
            "retained_after_group_removal_count": int(len(retained)),
            "retained_after_group_removal_mean_event_change_z": float(retained.mean()) if len(retained) else np.nan,
        })
    publication = pd.DataFrame(publication_rows).sort_values("record_count", ascending=False)

    duplicate_rows = []
    for site_id, subset in records.groupby("site_id"):
        if len(subset) < 2:
            continue
        positions = [int(records.index[records["proxy_index"].eq(value)][0]) for value in subset["proxy_index"]]
        for first in range(len(positions)):
            for second in range(first + 1, len(positions)):
                a, b = positions[first], positions[second]
                duplicate_rows.append({
                    "site_id": site_id,
                    "proxy_index_a": int(records.iloc[a]["proxy_index"]),
                    "proxy_index_b": int(records.iloc[b]["proxy_index"]),
                    "calibration_correlation": float(np.corrcoef(calibration_z[:, a], calibration_z[:, b])[0, 1]),
                    "target_correlation": float(np.corrcoef(target_z[:, a], target_z[:, b])[0, 1]),
                    "event_change_a_z": float(event_change[a]),
                    "event_change_b_z": float(event_change[b]),
                })
    duplicate_sites = pd.DataFrame(duplicate_rows)

    ts_values = np.asarray(loadmat(PAGES_TS, squeeze_me=True, struct_as_record=False, variable_names=["TS"])["TS"]).reshape(-1)
    ts_index: dict[tuple[str, str], list[object]] = defaultdict(list)
    for item in ts_values:
        ts_index[(clean_text(getattr(item, "dataSetName", "")), clean_text(getattr(item, "paleoData_TSid", "")))].append(item)
    metadata_long = []
    excluded_arrays = {"year", "age", "paleodatavalues", "paleodatayear", "paleodata_values"}
    for row in records.itertuples(index=False):
        key = (clean_text(row.official_dataset_name), clean_text(row.official_tsid))
        matches = ts_index.get(key, [])
        if len(matches) != 1:
            raise RuntimeError(f"cook-ts-identity-not-unique:{row.proxy_index}:{key}:{len(matches)}")
        item = matches[0]
        for field in getattr(item, "_fieldnames", []):
            if field.lower() in excluded_arrays:
                continue
            value = text_value(getattr(item, field, ""))
            if value:
                metadata_long.append({"proxy_index": int(row.proxy_index), "field": field, "value": value})
    ts_metadata = pd.DataFrame(metadata_long)

    lmr = pd.read_csv(LMR_META, low_memory=False)
    lmr_name_column = next(column for column in lmr.columns if "site" in column.lower() or "name" in column.lower())
    lmr_text = lmr.astype(str).agg("|".join, axis=1)
    site_tokens = [value for value in records["site_name"].astype(str) if value != "<missing>"]
    lmr_subset = lmr[lmr_text.apply(lambda value: any(token in value for token in site_tokens))].copy()
    lmr_subset.insert(0, "matched_name_column", lmr_subset[lmr_name_column].astype(str))

    components = pd.read_csv(sources["target_components"])
    components = components[components["proxy_index"].isin(proxy_indices)].copy()
    quality = pd.read_csv(sources["gpcc_quality"])
    quality = records[["proxy_index", "site_name", "publication_group", "latitude", "longitude"]].merge(
        quality, on=["proxy_index", "latitude", "longitude"], validate="one_to_one"
    )
    if len(quality) != 15 or len(components) != 30:
        raise RuntimeError(f"cook-climate-evidence-count-failed:{len(quality)}:{len(components)}")
    component_summary = components.groupby("branch", as_index=False).agg(
        record_count=("proxy_index", "size"),
        mean_temperature_event_change=("temperature_event_change", "mean"),
        mean_precipitation_event_change_mm_day=("precipitation_event_change_mm_day", "mean"),
        mean_temperature_component_z=("temperature_component_z", "mean"),
        mean_precipitation_component_z=("precipitation_component_z", "mean"),
        mean_total_component_z=("total_component_z", "mean"),
        mean_actual_proxy_event_change_z=("actual_proxy_event_change_z", "mean"),
    )

    output.mkdir(parents=True)
    records.to_csv(output / "cook_chronology_record_audit.csv", index=False)
    annual.to_csv(output / "cook_chronology_annual_1905_1920.csv", index=False)
    publication.to_csv(output / "cook_publication_group_audit.csv", index=False)
    duplicate_sites.to_csv(output / "cook_same_site_pair_audit.csv", index=False)
    ts_metadata.to_csv(output / "cook_pages_ts_nonempty_metadata_long.csv", index=False)
    lmr_subset.to_csv(output / "cook_lmr_metadata_matches.csv", index=False)
    quality.to_csv(output / "cook_precipitation_quality_and_event.csv", index=False)
    components.to_csv(output / "cook_psm_event_components.csv", index=False)
    component_summary.to_csv(output / "cook_psm_event_component_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), constrained_layout=True)
    palette = plt.get_cmap("tab20")
    for column in range(target_z.shape[1]):
        axes[0, 0].plot(TARGET_YEARS, target_z[:, column], color=palette(column), alpha=0.45, linewidth=1.0)
    axes[0, 0].plot(TARGET_YEARS, target_z.mean(axis=1), color="black", linewidth=2.6, label="15-record mean")
    axes[0, 0].axvspan(1915, 1918, color="#D55E00", alpha=0.12)
    axes[0, 0].axhline(0, color="#777777", linewidth=0.8)
    axes[0, 0].set_title("Cook-label chronology curves")
    axes[0, 0].set_ylabel("Proxy anomaly (1941-2000 SD)")
    axes[0, 0].legend(frameon=False)

    sorted_records = records.sort_values("event_change_z_recomputed")
    group_names = list(publication["publication_group"])
    group_colors = {name: palette(index) for index, name in enumerate(group_names)}
    axes[0, 1].bar(
        np.arange(len(sorted_records)),
        sorted_records["event_change_z_recomputed"],
        color=[group_colors[value] for value in sorted_records["publication_group"]],
    )
    axes[0, 1].set_xticks(np.arange(len(sorted_records)), sorted_records["site_name"], rotation=65, ha="right")
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_title("1915-1918 minus 1905-1914")
    axes[0, 1].set_ylabel("Event change (z)")

    corr = np.corrcoef(calibration_z, rowvar=False)
    image = axes[1, 0].imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    labels = records["site_name"].tolist()
    axes[1, 0].set_xticks(np.arange(len(labels)), labels, rotation=65, ha="right")
    axes[1, 0].set_yticks(np.arange(len(labels)), labels)
    axes[1, 0].set_title("Calibration pairwise correlation")
    fig.colorbar(image, ax=axes[1, 0], shrink=0.8)

    direct = quality["direct_gauge_month_fraction_1905_1918"].to_numpy(dtype=float)
    sizes = 45.0 + 260.0 * direct
    axes[1, 1].scatter(
        quality["clean_gpcc_event_change_mm_day"],
        quality["cru_event_change_mm_day"],
        s=sizes,
        c=quality["mean_interpolation_error_mm_month_1905_1918"],
        cmap="viridis",
        edgecolor="black",
        linewidth=0.5,
    )
    for row in quality.itertuples(index=False):
        axes[1, 1].annotate(str(row.site_name), (row.clean_gpcc_event_change_mm_day, row.cru_event_change_mm_day), fontsize=7, alpha=0.8)
    axes[1, 1].axhline(0, color="#777777", linewidth=0.8)
    axes[1, 1].axvline(0, color="#777777", linewidth=0.8)
    axes[1, 1].set_xlabel("Clean GPCC event change (mm/day)")
    axes[1, 1].set_ylabel("CRU event change (mm/day)")
    axes[1, 1].set_title("Precipitation evidence at Cook sites")
    fig.savefig(output / "cook_chronology_source_moisture_audit.png", dpi=220)
    plt.close(fig)

    quality_summary = {
        "mean_clean_gpcc_event_change_mm_day": float(quality["clean_gpcc_event_change_mm_day"].mean()),
        "mean_cru_event_change_mm_day": float(quality["cru_event_change_mm_day"].mean()),
        "clean_gpcc_negative_site_fraction": float(np.mean(quality["clean_gpcc_event_change_mm_day"] < 0)),
        "cru_negative_site_fraction": float(np.mean(quality["cru_event_change_mm_day"] < 0)),
        "mean_direct_gauge_month_fraction_1905_1918": float(quality["direct_gauge_month_fraction_1905_1918"].mean()),
        "sites_with_any_direct_gauge_month": int(np.sum(quality["direct_gauge_month_fraction_1905_1918"] > 0)),
        "sites_with_any_infilled_gauge_month": int(np.sum(quality["infilled_gauge_month_fraction_1905_1918"] > 0)),
        "mean_interpolation_error_mm_month_1905_1918": float(quality["mean_interpolation_error_mm_month_1905_1918"].mean()),
    }
    chronology_summary = {
        "record_count": 15,
        "unique_site_count": int(records["site_id"].nunique()),
        "publication_group_count": int(records["publication_group"].nunique()),
        "negative_event_record_count": int(np.sum(event_change < 0)),
        "mean_event_change_z": float(event_change.mean()),
        "median_event_change_z": float(np.median(event_change)),
        "calibration_pc1_explained_variance_fraction": pc1_variance,
        "event_vector_energy_fraction_along_calibration_pc1": event_energy_fraction,
        "calibration_structure": calibration_pairwise,
        "target_structure": target_pairwise,
        "maximum_event_change_closure_error": float(closure),
    }
    all_negative = chronology_summary["negative_event_record_count"] == 15
    no_direct_gauges = quality_summary["sites_with_any_direct_gauge_month"] == 0
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "event_definition": "mean(1915-1918) minus mean(1905-1914)",
        "chronology_summary": chronology_summary,
        "publication_group_summary": publication.to_dict(orient="records"),
        "precipitation_quality_summary": quality_summary,
        "psm_component_summary": component_summary.to_dict(orient="records"),
        "classification": (
            "shared_chronology_decline_with_weak_direct_early_precipitation_constraint"
            if all_negative and no_direct_gauges
            else "mixed_chronology_or_precipitation_evidence"
        ),
        "interpretation": (
            "A shared decline across the Cook-label chronologies is real in the archived proxy series, but the author label spans multiple publication groups. "
            "Clean GPCC and CRU can indicate whether drying is directionally plausible; absent direct early gauges, they cannot establish that moisture explains the proxy amplitude."
        ),
        "read_periods": [[1905, 1920], [1941, 2000]],
        "locked_1921_1939_accessed": False,
    }
    after = {name: source_state(path) for name, path in sources.items()}
    summary["all_declared_sources_unchanged"] = before == after
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("declared-source-changed-during-cook-chronology-audit")
    (output / "FINAL_RESULT.md").write_text(
        "# Cook Chronology Source and Moisture Audit v1\n\n"
        f"- Records/sites/publication groups: `{len(records)}` / `{records['site_id'].nunique()}` / `{records['publication_group'].nunique()}`.\n"
        f"- Negative event records: `{int(np.sum(event_change < 0))}/{len(event_change)}`; mean event change `{event_change.mean():+.3f} z`.\n"
        f"- Calibration PC1 variance: `{pc1_variance:.3f}`; event-vector energy on that PC1: `{event_energy_fraction:.3f}`.\n"
        f"- Mean clean GPCC / CRU event change: `{quality_summary['mean_clean_gpcc_event_change_mm_day']:+.3f}` / `{quality_summary['mean_cru_event_change_mm_day']:+.3f}` mm/day.\n"
        f"- Sites with any direct GPCC gauge month in 1905-1918: `{quality_summary['sites_with_any_direct_gauge_month']}/15`.\n\n"
        f"Classification: `{summary['classification']}`. No source data or production operator was changed.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
