"""Audit concentration of the LIA proxy contributions that orient the update warm."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_climate_reconstruction_loop_v2_round25_temperature_prior_update_decomposition_v1 import (
    interpolate_fields,
    native_coordinates,
    read_proxy_allowed,
    weighted_series,
)
from run_climate_reconstruction_loop_v2_round27_temperature_innovation_gain_sign_decomposition_v1 import (
    REPRESENTATIVE_EXPERIMENTS,
    YEARS,
    run_experiment,
)


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def decode_char(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset).reshape(-1, order="F")
    return "".join(chr(int(value)) for value in values if int(value) != 0).strip()


def metadata(path: Path) -> pd.DataFrame:
    rows = []
    with h5py.File(path, "r") as archive:
        references = np.asarray(archive["proxy_data/full_proxy"]).reshape(-1, order="F")
        for index, reference in enumerate(references):
            proxy = archive[reference]
            coefficients = np.asarray(proxy["b"], dtype=float).reshape(-1, order="F")
            location = np.asarray(proxy["location"], dtype=float).reshape(-1, order="F")
            rows.append({
                "record_index_1based": index + 1,
                "latitude": float(location[0]),
                "longitude": float(location[1]),
                "proxy_type": decode_char(proxy["ptype"]),
                "psm_method": decode_char(proxy["psm_mthd"]),
                "b1_temperature": float(coefficients[1]) if len(coefficients) > 1 else np.nan,
            })
    return pd.DataFrame(rows)


def target_prior(prior: np.ndarray, reference_netcdf: Path, mask_path: Path) -> np.ndarray:
    source_lat, source_lon = native_coordinates()
    fields = prior.T.reshape(prior.shape[1], len(source_lat), len(source_lon), order="F")
    with netcdf_file(reference_netcdf, "r", mmap=False) as dataset:
        target_lat = np.asarray(dataset.variables["lat"].data, dtype=float).copy()
        target_lon = np.asarray(dataset.variables["lon"].data, dtype=float).copy()
    mask = np.load(mask_path).astype(bool)
    return weighted_series(interpolate_fields(fields, source_lat, source_lon, target_lat, target_lon), mask, target_lat)


def shares(values: np.ndarray) -> dict[str, float | int]:
    positive = np.sort(values[values > 0])[::-1]
    absolute = np.sort(np.abs(values[np.abs(values) > 1e-15]))[::-1]
    positive_total = float(np.sum(positive))
    absolute_total = float(np.sum(absolute))
    positive_weights = positive / positive_total if positive_total > 0 else np.asarray([])
    absolute_weights = absolute / absolute_total if absolute_total > 0 else np.asarray([])
    cumulative = np.cumsum(positive_weights)
    return {
        "positive_record_count": int(len(positive)),
        "nonzero_record_count": int(len(absolute)),
        "positive_fraction": float(len(positive) / len(absolute)) if len(absolute) else np.nan,
        "top5_positive_share": float(np.sum(positive_weights[:5])) if len(positive_weights) else np.nan,
        "top10_positive_share": float(np.sum(positive_weights[:10])) if len(positive_weights) else np.nan,
        "top5_absolute_share": float(np.sum(absolute_weights[:5])) if len(absolute_weights) else np.nan,
        "top10_absolute_share": float(np.sum(absolute_weights[:10])) if len(absolute_weights) else np.nan,
        "effective_absolute_contributor_count": float(1.0 / np.sum(absolute_weights**2)) if len(absolute_weights) else np.nan,
        "positive_records_for_50percent": int(np.searchsorted(cumulative, 0.5) + 1) if len(cumulative) else 0,
    }


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_28_temperature_proxy_innovation_concentration_v1"
    archive = Path(r"D:\Users\JUNZ\Desktop\文件整理\1 论文\work1\compared_result_all")
    recovery = project / "validation_data/results/paleo_surface_validation_v0/enkf_reconstruction_source_recovery_audit_v2"
    paths = {
        "prior": recovery / "replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy",
        "indices": archive / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat",
        "proxy6": archive / "python_data/proxy_data_6.mat",
        "proxy9": archive / "python_data/proxy_data_9.mat",
        "regem6": recovery / "regem_proxy6_exact_v1/regem_output.mat",
        "regem9": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_regem_replay_v1/regem_output.mat",
        "reference_netcdf": loop / "round_21_temperature_reference_field_v1/temperature_reference_family_balanced_allowed_v1.nc",
        "hma_mask": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "proxy9_identity": loop / "round_07_proxy_network_suitability_inventory_v2/proxy9_suitability_inventory.csv",
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}

    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as archive_file:
        indices = np.asarray(archive_file["all_Xb_indices"], dtype=int) - 1
    target = target_prior(prior, paths["reference_netcdf"], paths["hma_mask"])
    identity9 = pd.read_csv(paths["proxy9_identity"])
    record_frames = []
    window_rows = []
    source_rows = []
    summaries = []
    max_closure = 0.0
    for network in (6, 9):
        hxb, observations, locked_reads = read_proxy_allowed(paths[f"proxy{network}"], YEARS)
        if locked_reads != 0:
            raise RuntimeError("locked-values-read")
        meta = metadata(paths[f"proxy{network}"])
        residuals = np.asarray(scipy.io.loadmat(paths[f"regem{network}"])["R0v_filled"], dtype=float)
        error_variance = np.nanvar(residuals, axis=1, ddof=1)
        runs = []
        for experiment in REPRESENTATIVE_EXPERIMENTS:
            result = run_experiment(target, hxb, observations, error_variance, meta.b1_temperature.to_numpy(float), indices[experiment])
            runs.append(result)
            max_closure = max(max_closure, float(result["closure"]))
        contribution = np.mean(np.stack([np.asarray(result["contributions"]) for result in runs]), axis=0)
        proxy_rows = []
        for window_index, start in enumerate(range(1400, 1900, 50)):
            selected = (YEARS >= start) & (YEARS <= start + 49)
            values = np.mean(contribution[:, selected], axis=1)
            active_fraction = np.mean(np.isfinite(observations[:, selected]), axis=1)
            metrics = shares(values)
            metrics.update({"network": network, "start_year": start, "end_year": start + 49, "net_increment_degC": float(np.sum(values))})
            window_rows.append(metrics)
            for index in range(len(values)):
                proxy_rows.append({
                    "network": network,
                    "record_index_1based": index + 1,
                    "start_year": start,
                    "end_year": start + 49,
                    "mean_contribution_degC": float(values[index]),
                    "active_fraction": float(active_fraction[index]),
                })
        proxy_table = pd.DataFrame(proxy_rows).merge(meta, on="record_index_1based", how="left", validate="many_to_one")
        if network == 9:
            identity_columns = [
                "record_index_1based",
                "metadata_sources",
                "archive_type",
                "measurement",
                "documented_climate_variable",
                "documented_seasonality",
                "network_role",
            ]
            proxy_table = proxy_table.merge(identity9[identity_columns], on="record_index_1based", how="left", validate="many_to_one")
            grouped = proxy_table.groupby(["start_year", "end_year", "metadata_sources"], dropna=False).mean_contribution_degC.sum().reset_index()
            grouped["network"] = 9
            source_rows.append(grouped)
        record_frames.append(proxy_table)
        matrix = proxy_table.pivot(index="record_index_1based", columns="start_year", values="mean_contribution_degC")
        warm_frequency = (matrix > 0).mean(axis=1)
        mean_contribution = matrix.mean(axis=1)
        ranking = pd.DataFrame({
            "record_index_1based": matrix.index,
            "warm_window_fraction": warm_frequency.values,
            "mean_lia_contribution_degC": mean_contribution.values,
            "absolute_mean_lia_contribution_degC": np.abs(mean_contribution.values),
        }).merge(meta, on="record_index_1based", how="left", validate="one_to_one")
        if network == 9:
            ranking = ranking.merge(identity9[identity_columns], on="record_index_1based", how="left", validate="one_to_one")
        ranking.sort_values("absolute_mean_lia_contribution_degC", ascending=False).to_csv(output / f"proxy{network}_record_contribution_ranking.csv", index=False)

    record_table = pd.concat(record_frames, ignore_index=True)
    record_table.to_csv(output / "proxy_window_contributions.csv", index=False)
    window_table = pd.DataFrame(window_rows)
    window_table.to_csv(output / "window_concentration_metrics.csv", index=False)
    if source_rows:
        pd.concat(source_rows, ignore_index=True).to_csv(output / "proxy9_source_group_contributions.csv", index=False)

    for network in (6, 9):
        selected = window_table[window_table.network.eq(network)]
        concentrated = bool(
            selected.top10_absolute_share.median() >= 0.50
            or selected.top10_positive_share.median() >= 0.60
            or selected.effective_absolute_contributor_count.median() <= 20.0
        )
        broad_mixed = bool(not concentrated and selected.positive_fraction.median() >= 0.40)
        summaries.append({
            "network": network,
            "median_positive_fraction": float(selected.positive_fraction.median()),
            "median_top10_positive_share": float(selected.top10_positive_share.median()),
            "median_top10_absolute_share": float(selected.top10_absolute_share.median()),
            "median_effective_absolute_contributor_count": float(selected.effective_absolute_contributor_count.median()),
            "median_positive_records_for_50percent": float(selected.positive_records_for_50percent.median()),
            "concentrated": concentrated,
            "broad_mixed_common_mode": broad_mixed,
        })
    summary_table = pd.DataFrame(summaries)
    summary_table.to_csv(output / "network_concentration_summary.csv", index=False)
    both_concentrated = bool(summary_table.concentrated.all())
    both_broad = bool(summary_table.broad_mixed_common_mode.all())
    decision = (
        "route_to_top_record_source_audit_without_deletion"
        if both_concentrated
        else "close_small_subset_explanation_and_redesign_network_wide_psm_chronology"
        if both_broad
        else "retain_concentration_mechanism_mixed_by_network"
    )
    summary = {
        "status": "pass" if max_closure <= 1e-12 else "fail",
        "locked_1921_1939_values_read": False,
        "closure_max_abs_degC": max_closure,
        "network_metrics": summaries,
        "decision": decision,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), sharex=True, constrained_layout=True)
    centers = (window_table.start_year + window_table.end_year) / 2
    for network, color in ((6, "#0072b2"), (9, "#d55e00")):
        selected = window_table.network.eq(network)
        axes[0].plot(centers[selected], window_table.loc[selected, "top10_absolute_share"], marker="o", color=color, lw=2, label=f"proxy{network}")
        axes[1].plot(centers[selected], window_table.loc[selected, "effective_absolute_contributor_count"], marker="s", color=color, lw=2, label=f"proxy{network}")
    axes[0].axhline(0.5, color="#555555", lw=0.9, ls="--")
    axes[0].set(ylabel="Top-10 absolute share", title="Contribution concentration")
    axes[1].axhline(20, color="#555555", lw=0.9, ls="--")
    axes[1].set(xlabel="50-year window center", ylabel="Effective contributor count", title="Effective number of contributing records")
    for axis in axes:
        axis.legend(frameon=False)
    figure.savefig(output / "temperature_proxy_innovation_concentration.png", dpi=190)
    plt.close(figure)

    (output / "FINAL_RESULT.md").write_text(
        "# Temperature Proxy-Innovation Concentration\n\n"
        + "\n".join(
            f"- proxy{row['network']}: median top-10 absolute share `{row['median_top10_absolute_share']:.3f}`, "
            f"effective contributors `{row['median_effective_absolute_contributor_count']:.1f}`, "
            f"concentrated `{str(row['concentrated']).lower()}`."
            for row in summaries
        )
        + f"\n\nDecision: `{decision}`.\n",
        encoding="utf-8",
    )
    after = {name: state(path) for name, path in paths.items()}
    pure = before == after
    (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": pure}, indent=2), encoding="utf-8")
    if not pure:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
