from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

EXPERIMENT_ID = "climate_reconstruction_positive_b2_fixed_network_common_mode_robustness_v1"
CALIBRATION_YEARS = np.arange(1941, 2001, dtype=int)
SPATIAL_THRESHOLDS_KM = (250, 500, 750)
RNG_SEED = 20260815


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


def source_stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def haversine_matrix(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(latitude)
    lon = np.deg2rad(longitude)
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    value = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    return 6371.0088 * 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(value)))


def bootstrap_interval(values: np.ndarray, groups: np.ndarray | None, rng: np.random.Generator, count: int = 20000) -> dict[str, float]:
    estimates = np.empty(count, dtype=float)
    if groups is None:
        for index in range(count):
            estimates[index] = float(np.mean(rng.choice(values, size=len(values), replace=True)))
    else:
        unique = np.unique(groups)
        by_group = {group: values[groups == group] for group in unique}
        for index in range(count):
            sampled = rng.choice(unique, size=len(unique), replace=True)
            estimates[index] = float(np.mean(np.concatenate([by_group[group] for group in sampled])))
    return {
        "mean": float(np.mean(estimates)),
        "q025": float(np.quantile(estimates, 0.025)),
        "q500": float(np.quantile(estimates, 0.500)),
        "q975": float(np.quantile(estimates, 0.975)),
        "fraction_negative": float(np.mean(estimates < 0)),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    drivers = Path(__file__).resolve().parent
    base = load_module(drivers / "audit_positive_b2_event_seasonal_psm_v1.py", "positive_b2_base_common_mode")
    fixed_root = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_clean_gpcc_psm_rolling_pseudo_event_fixed_network_v1"
    event_root = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_positive_b2_event_site_instrumental_consistency_v1"
    common_root = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_proxy_common_mode_source_audit_v2"
    output = root / f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_ID}"
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    proxy6 = base.archive_proxy_path()
    sources = {
        "fixed_indices": fixed_root / "fixed_complete_proxy_indices.csv",
        "fixed_target": fixed_root / "fixed_network_target_1905_components.csv",
        "event_metadata": event_root / "site_instrumental_event_consistency.csv",
        "source_mapping": common_root / "proxy_source_mapping.csv",
        "proxy6": proxy6,
        "facts": root / "LIA_PROJECT_FACTS.md",
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: source_stat(path) for name, path in sources.items()}

    fixed_indices = pd.read_csv(sources["fixed_indices"])["proxy_index"].to_numpy(dtype=int)
    target_all = pd.read_csv(sources["fixed_target"])
    target = target_all[target_all["branch"] == "BEA_clean_GPCC"][
        ["proxy_index", "actual_proxy_event_change_z"]
    ].rename(columns={"actual_proxy_event_change_z": "event_change_z"})
    metadata = pd.read_csv(sources["event_metadata"])
    source_mapping = pd.read_csv(sources["source_mapping"])[["proxy_index", "source_flag", "data_name", "source_family"]]
    records = (
        pd.DataFrame({"proxy_index": fixed_indices})
        .merge(target, on="proxy_index", validate="one_to_one")
        .merge(
            metadata[[
                "proxy_index", "latitude", "longitude", "site_id", "latitude_band", "longitude_band",
                "site_record_count", "site_multiplicity", "pc1_loading", "calibration_valid_count",
            ]],
            on="proxy_index",
            validate="one_to_one",
        )
        .merge(source_mapping, on="proxy_index", validate="one_to_one")
        .sort_values("proxy_index")
        .reset_index(drop=True)
    )
    if len(records) != 50 or records["proxy_index"].nunique() != 50:
        raise RuntimeError("fixed-record-identity-contract-failed")
    if records["site_id"].nunique() != 47:
        raise RuntimeError("fixed-site-count-contract-failed")

    calibration = np.full((len(CALIBRATION_YEARS), len(records)), np.nan)
    with h5py.File(proxy6, "r") as handle:
        refs = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
        for column, proxy_index in enumerate(records["proxy_index"].to_numpy(dtype=int)):
            proxy = handle[refs[proxy_index - 1]]
            ages = np.asarray(proxy["age_yng"][0, :], dtype=float)
            positions = []
            for year in CALIBRATION_YEARS:
                match = np.where(ages == year)[0]
                if len(match) != 1:
                    raise RuntimeError(f"proxy-calibration-year-match-failed:{proxy_index}:{year}:{len(match)}")
                positions.append(int(match[0]))
            calibration[:, column] = np.asarray(proxy["data"][positions, 0], dtype=float)
    finite_counts = np.sum(np.isfinite(calibration), axis=0)
    if int(np.min(finite_counts)) < 58:
        raise RuntimeError(f"unexpected-calibration-missingness:{int(np.min(finite_counts))}")
    records["calibration_1941_2000_finite_count"] = finite_counts

    site_table = records.groupby("site_id", as_index=False).agg(
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        latitude_band=("latitude_band", "first"),
        longitude_band=("longitude_band", "first"),
        event_change_z=("event_change_z", "mean"),
        record_count=("proxy_index", "size"),
        proxy_indices=("proxy_index", lambda values: ";".join(str(int(value)) for value in sorted(values))),
    )
    record_mean = float(records["event_change_z"].mean())
    site_mean = float(site_table["event_change_z"].mean())

    leave_record_rows = []
    for row in records.itertuples(index=False):
        retained = records[records["proxy_index"] != row.proxy_index]["event_change_z"]
        leave_record_rows.append({
            "omitted_proxy_index": int(row.proxy_index),
            "omitted_event_change_z": float(row.event_change_z),
            "retained_count": int(len(retained)),
            "retained_mean_event_change_z": float(retained.mean()),
            "change_from_full_record_mean_z": float(retained.mean() - record_mean),
        })
    leave_record = pd.DataFrame(leave_record_rows)

    leave_site_rows = []
    for row in site_table.itertuples(index=False):
        retained = site_table[site_table["site_id"] != row.site_id]["event_change_z"]
        leave_site_rows.append({
            "omitted_site_id": row.site_id,
            "omitted_record_count": int(row.record_count),
            "omitted_site_event_change_z": float(row.event_change_z),
            "retained_site_count": int(len(retained)),
            "retained_mean_event_change_z": float(retained.mean()),
            "change_from_full_site_mean_z": float(retained.mean() - site_mean),
        })
    leave_site = pd.DataFrame(leave_site_rows)

    distance = haversine_matrix(site_table["latitude"].to_numpy(dtype=float), site_table["longitude"].to_numpy(dtype=float))
    hierarchy = linkage(squareform(distance, checks=False), method="complete")
    assignment_rows = []
    spatial_rows = []
    assignment_by_threshold = {}
    for threshold in SPATIAL_THRESHOLDS_KM:
        labels = fcluster(hierarchy, t=float(threshold), criterion="distance")
        assignment_by_threshold[threshold] = labels
        for site_id, label in zip(site_table["site_id"], labels):
            assignment_rows.append({"threshold_km": threshold, "site_id": site_id, "cluster_id": int(label)})
        for label in np.unique(labels):
            omitted = labels == label
            retained = site_table.loc[~omitted, "event_change_z"]
            spatial_rows.append({
                "threshold_km": threshold,
                "cluster_id": int(label),
                "omitted_site_count": int(omitted.sum()),
                "retained_site_count": int((~omitted).sum()),
                "omitted_cluster_mean_event_change_z": float(site_table.loc[omitted, "event_change_z"].mean()),
                "retained_mean_event_change_z": float(retained.mean()),
                "change_from_full_site_mean_z": float(retained.mean() - site_mean),
            })
    assignments = pd.DataFrame(assignment_rows)
    spatial_jackknife = pd.DataFrame(spatial_rows)

    categorical_rows = []
    for grouping, frame, value_column in (
        ("source_family", records, "source_family"),
        ("longitude_band", site_table, "longitude_band"),
        ("latitude_band", site_table, "latitude_band"),
    ):
        full_mean = record_mean if grouping == "source_family" else site_mean
        unit_name = "record" if grouping == "source_family" else "site"
        for group in sorted(frame[value_column].astype(str).unique()):
            omitted = frame[value_column].astype(str) == group
            retained = frame.loc[~omitted, "event_change_z"]
            categorical_rows.append({
                "grouping": grouping,
                "omitted_group": group,
                "unit": unit_name,
                "omitted_count": int(omitted.sum()),
                "retained_count": int((~omitted).sum()),
                "retained_mean_event_change_z": float(retained.mean()) if len(retained) else float("nan"),
                "change_from_full_mean_z": float(retained.mean() - full_mean) if len(retained) else float("nan"),
                "eligible_interpretation": bool(int(omitted.sum()) >= 2 and int((~omitted).sum()) >= 10),
            })
    categorical = pd.DataFrame(categorical_rows)

    column_mean = np.nanmean(calibration, axis=0)
    column_std = np.nanstd(calibration, axis=0)
    standardized = (calibration - column_mean[None, :]) / column_std[None, :]
    standardized[~np.isfinite(standardized)] = 0.0
    proxy_correlation = np.corrcoef(standardized, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(proxy_correlation)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    off_diagonal = proxy_correlation[~np.eye(len(records), dtype=bool)]
    mean_correlation = float(np.mean(off_diagonal))
    denominator = 1.0 + (len(records) - 1.0) * mean_correlation
    effective_n_mean_correlation = float(len(records) / denominator) if denominator > 0 else float("nan")
    effective_rank = float(np.sum(eigenvalues) ** 2 / np.sum(eigenvalues ** 2))
    pc1_explained = float(eigenvalues[-1] / np.sum(eigenvalues))
    pc1_loadings = eigenvectors[:, -1]
    if np.corrcoef(pc1_loadings, records["event_change_z"])[0, 1] > 0:
        pc1_loadings = -pc1_loadings
    records["calibration_pc1_loading"] = pc1_loadings
    pc1_event_correlation = float(np.corrcoef(pc1_loadings, records["event_change_z"])[0, 1])
    cluster_500 = assignment_by_threshold[500]
    site_cluster_map = dict(zip(site_table["site_id"], cluster_500))
    record_clusters_500 = records["site_id"].map(site_cluster_map).to_numpy(dtype=int)
    same_cluster = record_clusters_500[:, None] == record_clusters_500[None, :]
    diagonal = np.eye(len(records), dtype=bool)
    within_cluster_correlation = float(np.mean(proxy_correlation[same_cluster & ~diagonal]))
    between_cluster_correlation = float(np.mean(proxy_correlation[~same_cluster]))

    sorted_sites = site_table.sort_values("event_change_z").reset_index(drop=True)
    adversarial_removal_count = None
    adversarial_rows = []
    for removed in range(0, len(sorted_sites)):
        retained = sorted_sites.iloc[removed:]["event_change_z"]
        retained_mean = float(retained.mean())
        adversarial_rows.append({
            "removed_most_negative_site_count": removed,
            "retained_site_count": int(len(retained)),
            "retained_mean_event_change_z": retained_mean,
        })
        if retained_mean >= 0 and adversarial_removal_count is None:
            adversarial_removal_count = removed
    adversarial = pd.DataFrame(adversarial_rows)

    rng = np.random.default_rng(RNG_SEED)
    naive_bootstrap = bootstrap_interval(site_table["event_change_z"].to_numpy(dtype=float), None, rng)
    cluster_bootstrap = bootstrap_interval(site_table["event_change_z"].to_numpy(dtype=float), cluster_500, rng)
    eligible_categorical = categorical[categorical["eligible_interpretation"]]
    all_jackknife_values = np.concatenate([
        leave_site["retained_mean_event_change_z"].to_numpy(dtype=float),
        spatial_jackknife["retained_mean_event_change_z"].to_numpy(dtype=float),
        eligible_categorical["retained_mean_event_change_z"].to_numpy(dtype=float),
    ])
    robust_negative = bool(np.all(all_jackknife_values < 0))
    classification = "robust_negative_common_mode_under_tested_group_deletions" if robust_negative else "negative_common_mode_not_robust_to_all_tested_group_deletions"

    output.mkdir(parents=True)
    records.to_csv(output / "fixed_record_event_and_correlation_metadata.csv", index=False)
    site_table.to_csv(output / "site_collapsed_event_table.csv", index=False)
    leave_record.to_csv(output / "leave_one_record_out.csv", index=False)
    leave_site.to_csv(output / "leave_one_site_out.csv", index=False)
    assignments.to_csv(output / "spatial_cluster_assignments.csv", index=False)
    spatial_jackknife.to_csv(output / "spatial_block_jackknife.csv", index=False)
    categorical.to_csv(output / "categorical_group_jackknife.csv", index=False)
    adversarial.to_csv(output / "adversarial_negative_site_removal.csv", index=False)
    pd.DataFrame(proxy_correlation, index=records["proxy_index"], columns=records["proxy_index"]).to_csv(output / "calibration_proxy_correlation_matrix.csv")

    spatial_summary = []
    for threshold in SPATIAL_THRESHOLDS_KM:
        group = spatial_jackknife[spatial_jackknife["threshold_km"] == threshold]
        spatial_summary.append({
            "threshold_km": threshold,
            "cluster_count": int(len(group)),
            "minimum_retained_mean_event_change_z": float(group["retained_mean_event_change_z"].min()),
            "maximum_retained_mean_event_change_z": float(group["retained_mean_event_change_z"].max()),
            "all_retained_means_negative": bool((group["retained_mean_event_change_z"] < 0).all()),
        })
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "locked_final_period": [1921, 1939],
        "locked_final_period_accessed": False,
        "proxy_read_period": [1941, 2000],
        "record_count": int(len(records)),
        "unique_site_count": int(len(site_table)),
        "record_weighted_event_mean_z": record_mean,
        "site_collapsed_event_mean_z": site_mean,
        "duplicate_site_collapse_change_z": site_mean - record_mean,
        "negative_record_fraction": float(np.mean(records["event_change_z"] < 0)),
        "negative_site_fraction": float(np.mean(site_table["event_change_z"] < 0)),
        "leave_one_record_retained_mean_range_z": [float(leave_record["retained_mean_event_change_z"].min()), float(leave_record["retained_mean_event_change_z"].max())],
        "leave_one_site_retained_mean_range_z": [float(leave_site["retained_mean_event_change_z"].min()), float(leave_site["retained_mean_event_change_z"].max())],
        "spatial_block_summary": spatial_summary,
        "eligible_categorical_retained_mean_range_z": [float(eligible_categorical["retained_mean_event_change_z"].min()), float(eligible_categorical["retained_mean_event_change_z"].max())],
        "calibration_correlation_structure": {
            "mean_off_diagonal_correlation": mean_correlation,
            "effective_sample_size_mean_correlation": effective_n_mean_correlation,
            "effective_rank": effective_rank,
            "pc1_explained_variance_fraction": pc1_explained,
            "pc1_loading_event_change_correlation": pc1_event_correlation,
            "mean_within_500km_cluster_correlation": within_cluster_correlation,
            "mean_between_500km_cluster_correlation": between_cluster_correlation,
            "missing_value_rule": "Each record standardized over 1941-2000; at most two missing years were replaced by its zero standardized mean before the correlation eigendecomposition.",
        },
        "site_bootstrap": naive_bootstrap,
        "spatial_500km_cluster_bootstrap": cluster_bootstrap,
        "adversarial_most_negative_sites_needed_to_reach_nonnegative_mean": adversarial_removal_count,
        "classification": classification,
        "metadata_limit": "The fixed archive exposes source family, location, proxy type, and PSM metadata but no species or measurement-type field; no species exclusion is claimed.",
        "interpretation_limit": "Bootstrap and spatial blocks are robustness sensitivities, not a formal observation-error model or authorization to alter production proxies.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.6), constrained_layout=True)
    scatter = axes[0, 0].scatter(site_table["longitude"], site_table["latitude"], c=site_table["event_change_z"], cmap="RdBu_r", vmin=-2.0, vmax=2.0, s=45 + 24 * site_table["record_count"], edgecolor="#333333", linewidth=0.4)
    axes[0, 0].set_xlabel("Longitude"); axes[0, 0].set_ylabel("Latitude"); axes[0, 0].set_title("Site-collapsed 1905-1918 event change")
    figure.colorbar(scatter, ax=axes[0, 0], label="Event change (z)")

    plot_groups = []
    for row in spatial_jackknife.itertuples(index=False):
        plot_groups.append((f"{int(row.threshold_km)}km-{int(row.cluster_id)}", row.retained_mean_event_change_z, f"{int(row.threshold_km)} km"))
    for row in eligible_categorical.itertuples(index=False):
        plot_groups.append((f"{row.grouping}:{row.omitted_group}", row.retained_mean_event_change_z, "metadata band"))
    jackknife_plot = pd.DataFrame(plot_groups, columns=["omitted_group", "retained_mean", "family"]).sort_values("retained_mean")
    family_colors = {"250 km": "#0072B2", "500 km": "#009E73", "750 km": "#D55E00", "metadata band": "#CC79A7"}
    for family, group in jackknife_plot.groupby("family"):
        axes[0, 1].scatter(group["retained_mean"], np.arange(len(group)) + list(jackknife_plot.index).index(group.index[0]) if False else group.index, color=family_colors[family], s=28, label=family)
    axes[0, 1].axvline(site_mean, color="#1B1B1B", linewidth=1.2, label="full site mean")
    axes[0, 1].axvline(0, color="#888888", linewidth=0.8)
    axes[0, 1].set_xlabel("Retained site mean event change (z)"); axes[0, 1].set_ylabel("Omitted-group ordering")
    axes[0, 1].set_title("Spatial and metadata block deletions"); axes[0, 1].legend(frameon=False, fontsize=8)

    order = np.argsort(record_clusters_500 * 1000 + records["longitude"].to_numpy(dtype=float))
    image = axes[1, 0].imshow(proxy_correlation[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes[1, 0].set_xlabel("Records ordered by 500 km cluster"); axes[1, 0].set_ylabel("Records ordered by 500 km cluster")
    axes[1, 0].set_title("1941-2000 proxy correlation")
    figure.colorbar(image, ax=axes[1, 0], label="Correlation")

    bootstrap_rng = np.random.default_rng(RNG_SEED + 1)
    cluster_ids = np.unique(cluster_500)
    group_values = {group: site_table.loc[cluster_500 == group, "event_change_z"].to_numpy(dtype=float) for group in cluster_ids}
    bootstrap_values = np.empty(20000, dtype=float)
    for index in range(len(bootstrap_values)):
        sampled = bootstrap_rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        bootstrap_values[index] = float(np.mean(np.concatenate([group_values[group] for group in sampled])))
    axes[1, 1].hist(bootstrap_values, bins=45, color="#56B4E9", alpha=0.8)
    axes[1, 1].axvline(site_mean, color="#1B1B1B", linewidth=1.4, label="site mean")
    axes[1, 1].axvline(0, color="#888888", linewidth=0.8)
    axes[1, 1].set_xlabel("500 km cluster-bootstrap event mean (z)"); axes[1, 1].set_ylabel("Replicate count")
    axes[1, 1].set_title("Correlation-aware spatial sensitivity"); axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    figure.savefig(output / "fixed_network_common_mode_robustness.png", dpi=190)
    plt.close(figure)

    report = [
        "# Fixed-network Common-mode Robustness Result",
        "",
        f"- Record/site count: `{len(records)}` / `{len(site_table)}`.",
        f"- Record-weighted/site-collapsed event mean: `{record_mean:.3f}` / `{site_mean:.3f} z`.",
        f"- Leave-one-site retained range: `{summary['leave_one_site_retained_mean_range_z'][0]:.3f}` to `{summary['leave_one_site_retained_mean_range_z'][1]:.3f} z`.",
        f"- 500 km cluster-bootstrap 95% interval: `{cluster_bootstrap['q025']:.3f}` to `{cluster_bootstrap['q975']:.3f} z`; negative fraction `{cluster_bootstrap['fraction_negative']:.1%}`.",
        f"- Mean calibration correlation/effective N/effective rank: `{mean_correlation:.3f}` / `{effective_n_mean_correlation:.2f}` / `{effective_rank:.2f}`.",
        f"- Calibration PC1 explained variance: `{pc1_explained:.1%}`; loading-event correlation `{pc1_event_correlation:.3f}`.",
        f"- Adversarial removal needed to erase the negative site mean: `{adversarial_removal_count}` most-negative sites.",
        f"- Classification: `{classification}`.",
        "",
        "No values from 1921-1939 were read. Original inputs were not modified.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    after = {name: source_stat(path) for name, path in sources.items()}
    if before != after:
        raise RuntimeError("source-inputs-changed-during-audit")
    manifest = {
        "sources_before": before,
        "sources_after": after,
        "driver_sha256": digest(Path(__file__)),
        "facts_sha256": digest(root / "LIA_PROJECT_FACTS.md"),
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())