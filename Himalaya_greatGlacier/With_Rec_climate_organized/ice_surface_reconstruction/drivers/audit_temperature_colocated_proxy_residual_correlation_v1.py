from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import audit_temperature_psm_modern_stability_v1 as modern


ROOT = modern.ROOT
OUTPUT = ROOT / (
    r"validation_data\results\paleo_surface_validation_v0"
    r"\climate_reconstruction_temperature_colocated_proxy_residual_"
    r"correlation_audit_v1"
)
ATTRIBUTION = ROOT / (
    r"validation_data\results\paleo_surface_validation_v0"
    r"\climate_reconstruction_temperature_west_increment_attribution_1861_1900_v2"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(left) & np.isfinite(right)
    count = int(np.count_nonzero(valid))
    if count < 50:
        return {"n": count, "correlation": np.nan, "affine_nrmse": np.nan}
    x = left[valid]
    y = right[valid]
    correlation = float(np.corrcoef(x, y)[0, 1])
    design = np.column_stack([np.ones(count), x])
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    scale = float(np.std(y, ddof=0))
    affine_nrmse = float(np.sqrt(np.mean(residual**2)) / scale)
    return {
        "n": count,
        "correlation": correlation,
        "affine_nrmse": affine_nrmse,
    }


def residual_correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(np.count_nonzero(valid)) < 50:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def main() -> None:
    output_files = [
        OUTPUT / "modern_proxy_psm_residuals_1940_2000.csv",
        OUTPUT / "colocated_group_audit.csv",
        OUTPUT / "exact_affine_duplicate_pairs.csv",
        OUTPUT / "network_summary.csv",
        OUTPUT / "temperature_colocated_proxy_residual_correlation.png",
        OUTPUT / "summary.json",
        OUTPUT / "source_manifest.json",
        OUTPUT / "FINAL_RESULT.md",
    ]
    existing = [path for path in output_files if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite outputs: {existing}")

    manifest = json.loads(modern.MANIFEST.read_text(encoding="utf-8"))[
        "sources_before"
    ]
    paths = {
        "proxy6": Path(manifest["proxy6"]["path"]),
        "proxy9": Path(manifest["proxy9"]["path"]),
        "climate": modern.CLIMATE,
        "attribution6": ATTRIBUTION / "proxy6_west_trend_contribution_ranking.csv",
        "attribution9": ATTRIBUTION / "proxy9_west_trend_contribution_ranking.csv",
        "manifest": modern.MANIFEST,
        "contract": OUTPUT / "ROUND_CONTRACT.md",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: sha256(path) for name, path in paths.items()}

    with h5py.File(modern.CLIMATE, "r") as archive:
        all_years = np.asarray(archive["time"], dtype=int)
        indices = np.flatnonzero(
            (all_years >= modern.YEARS[0]) & (all_years <= modern.YEARS[-1])
        )
        years = all_years[indices]
        if not np.array_equal(years, modern.YEARS):
            raise RuntimeError("Modern climate year axis mismatch")
        latitudes = np.asarray(archive["lat"], dtype=float)
        longitudes = np.asarray(archive["lon"], dtype=float)
        temperature = np.asarray(
            archive["tas_obs_bea_05deg"][indices], dtype=float
        )
        precipitation = np.asarray(
            archive["pr_obs_gpcp_05deg"][indices], dtype=float
        )
    temperature[temperature <= -9000] = np.nan
    precipitation[precipitation <= -9000] = np.nan

    record_rows = []
    series: dict[tuple[int, int], np.ndarray] = {}
    residuals: dict[tuple[int, int], np.ndarray] = {}
    for network in (6, 9):
        with h5py.File(paths[f"proxy{network}"], "r") as archive:
            references = np.asarray(archive["proxy_data/full_proxy"]).reshape(
                -1, order="F"
            )
            for index, reference in enumerate(references):
                proxy = archive[reference]
                coefficients = np.asarray(proxy["b"], dtype=float).reshape(
                    -1, order="F"
                )
                if len(coefficients) != 3 or modern.decode(proxy["psm_mthd"]) != "bivariate":
                    continue
                location = np.asarray(proxy["location"], dtype=float).reshape(
                    -1, order="F"
                )
                latitude = float(location[0])
                longitude = float(location[1])
                lat_index = int(np.argmin(np.abs(latitudes - latitude)))
                lon_index = int(np.argmin(np.abs(longitudes - longitude)))
                actual = modern.chronology(proxy)
                predicted = (
                    coefficients[0]
                    + coefficients[1] * temperature[:, lat_index, lon_index]
                    + coefficients[2] * precipitation[:, lat_index, lon_index]
                )
                residual = actual - predicted
                key = (network, index + 1)
                series[key] = actual
                residuals[key] = residual
                row = {
                    "network": network,
                    "record_index_1based": index + 1,
                    "latitude": latitude,
                    "longitude": longitude,
                    "site_key": f"{latitude:.8f},{longitude:.8f}",
                    "production_b1_temperature": float(coefficients[1]),
                    "production_b2_precipitation": float(coefficients[2]),
                }
                row.update(
                    {
                        f"proxy_{year}": float(value)
                        for year, value in zip(modern.YEARS, actual)
                    }
                )
                row.update(
                    {
                        f"residual_{year}": float(value)
                        for year, value in zip(modern.YEARS, residual)
                    }
                )
                record_rows.append(row)
    records = pd.DataFrame(record_rows)
    records.to_csv(output_files[0], index=False)

    attribution = pd.concat(
        [pd.read_csv(paths["attribution6"]), pd.read_csv(paths["attribution9"])],
        ignore_index=True,
    )
    attribution_lookup = attribution.set_index(["network", "record_index_1based"])[
        "trend_contribution_degC_per_century"
    ]
    group_rows = []
    duplicate_rows = []
    flagged_records: dict[int, set[int]] = {6: set(), 9: set()}
    duplicate_records: dict[int, set[int]] = {6: set(), 9: set()}
    for (network, site_key), group in records.groupby(["network", "site_key"]):
        if len(group) < 2:
            continue
        indices_here = group["record_index_1based"].astype(int).tolist()
        residual_correlations = []
        exact_pairs = 0
        for left_index, right_index in combinations(indices_here, 2):
            residual_r = residual_correlation(
                residuals[(int(network), left_index)],
                residuals[(int(network), right_index)],
            )
            residual_correlations.append(residual_r)
            sequence = pair_metrics(
                series[(int(network), left_index)],
                series[(int(network), right_index)],
            )
            exact = bool(
                sequence["n"] >= 50
                and abs(sequence["correlation"]) >= 0.999999
                and sequence["affine_nrmse"] <= 1e-6
            )
            if exact:
                exact_pairs += 1
                duplicate_records[int(network)].update([left_index, right_index])
            duplicate_rows.append(
                {
                    "network": int(network),
                    "site_key": site_key,
                    "left_record_index_1based": left_index,
                    "right_record_index_1based": right_index,
                    "overlap_n": sequence["n"],
                    "sequence_correlation": sequence["correlation"],
                    "affine_nrmse": sequence["affine_nrmse"],
                    "residual_correlation": residual_r,
                    "exact_affine_duplicate": exact,
                }
            )
        median_residual = float(np.nanmedian(residual_correlations))
        correlated = bool(median_residual >= 0.5)
        if correlated:
            flagged_records[int(network)].update(indices_here)
        contributions = np.asarray(
            [
                attribution_lookup.loc[(int(network), record_index)]
                for record_index in indices_here
            ],
            dtype=float,
        )
        group_rows.append(
            {
                "network": int(network),
                "site_key": site_key,
                "latitude": float(group["latitude"].iloc[0]),
                "longitude": float(group["longitude"].iloc[0]),
                "record_count": int(len(group)),
                "pair_count": int(len(residual_correlations)),
                "median_offdiagonal_residual_correlation": median_residual,
                "maximum_offdiagonal_residual_correlation": float(
                    np.nanmax(residual_correlations)
                ),
                "exact_affine_duplicate_pair_count": exact_pairs,
                "correlated_error_group": correlated,
                "signed_baseline_attribution_degC_per_century": float(
                    np.sum(contributions)
                ),
                "absolute_baseline_attribution_degC_per_century": float(
                    np.sum(np.abs(contributions))
                ),
            }
        )
    groups = pd.DataFrame(group_rows)
    groups.to_csv(output_files[1], index=False)
    duplicates = pd.DataFrame(duplicate_rows)
    duplicates.to_csv(output_files[2], index=False)

    network_rows = []
    for network in (6, 9):
        network_attribution = attribution.loc[attribution["network"].eq(network)]
        total_absolute = float(
            network_attribution["trend_contribution_degC_per_century"].abs().sum()
        )
        flagged = flagged_records[network]
        duplicate = duplicate_records[network]
        flagged_values = np.asarray(
            [attribution_lookup.loc[(network, record)] for record in flagged],
            dtype=float,
        )
        duplicate_values = np.asarray(
            [attribution_lookup.loc[(network, record)] for record in duplicate],
            dtype=float,
        )
        network_rows.append(
            {
                "network": network,
                "bivariate_record_count": int(
                    records.loc[records["network"].eq(network)].shape[0]
                ),
                "colocated_group_count": int(
                    groups.loc[groups["network"].eq(network)].shape[0]
                ),
                "correlated_error_group_count": int(
                    groups.loc[
                        groups["network"].eq(network)
                        & groups["correlated_error_group"]
                    ].shape[0]
                ),
                "correlated_error_record_count": len(flagged),
                "correlated_error_signed_attribution": float(
                    flagged_values.sum() if len(flagged_values) else 0.0
                ),
                "correlated_error_absolute_attribution_share": float(
                    np.abs(flagged_values).sum() / total_absolute
                    if total_absolute
                    else np.nan
                ),
                "exact_affine_duplicate_record_count": len(duplicate),
                "exact_affine_duplicate_absolute_attribution_share": float(
                    np.abs(duplicate_values).sum() / total_absolute
                    if total_absolute
                    else np.nan
                ),
            }
        )
    network_summary = pd.DataFrame(network_rows)
    network_summary.to_csv(output_files[3], index=False)
    authorize = bool(
        np.any(
            (network_summary["correlated_error_record_count"] >= 10)
            & (
                network_summary["correlated_error_absolute_attribution_share"]
                >= 0.20
            )
        )
    )

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)
    for network, subset in groups.groupby("network"):
        axes[0].scatter(
            subset["record_count"],
            subset["median_offdiagonal_residual_correlation"],
            s=30
            + 180
            * subset["absolute_baseline_attribution_degC_per_century"]
            / max(groups["absolute_baseline_attribution_degC_per_century"].max(), 1e-12),
            alpha=0.75,
            label=f"proxy{network}",
        )
    axes[0].axhline(0.5, color="#C8503E", linestyle="--", linewidth=1.0)
    axes[0].set(
        xlabel="Records at exact coordinate",
        ylabel="Median PSM-residual correlation",
        title="Colocated residual dependence",
    )
    axes[0].legend(frameon=False)

    x = np.arange(len(network_summary))
    axes[1].bar(
        x,
        network_summary["correlated_error_absolute_attribution_share"],
        color=["#147D73", "#2F6FA3"],
    )
    axes[1].axhline(0.20, color="#C8503E", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(x, [f"proxy{value}" for value in network_summary["network"]])
    axes[1].set(
        ylabel="Share of absolute baseline attribution",
        title="Contribution covered by correlated-error groups",
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output_files[4], dpi=240)
    plt.close(figure)

    after = {name: sha256(path) for name, path in paths.items()}
    if before != after:
        raise RuntimeError("An input changed during the audit")
    summary = {
        "status": "complete",
        "read_period": [1940, 2000],
        "network_summary": network_rows,
        "block_r_sensitivity_authorized": authorize,
        "authorization_rule": (
            "at_least_10_correlated_error_records_and_at_least_20_percent_"
            "absolute_baseline_attribution_in_either_network"
        ),
        "years_1901_1939_accessed": False,
        "input_hashes_unchanged": before == after,
        "production_modified": False,
    }
    output_files[5].write_text(json.dumps(summary, indent=2), encoding="ascii")
    output_files[6].write_text(
        json.dumps({"before": before, "after": after}, indent=2), encoding="ascii"
    )
    lines = ["# Final result", ""]
    for row in network_rows:
        lines.append(
            f"- proxy{row['network']}: {row['correlated_error_record_count']} records in correlated-error groups; "
            f"{100 * row['correlated_error_absolute_attribution_share']:.2f}% of absolute attribution; "
            f"{row['exact_affine_duplicate_record_count']} exact affine-duplicate records."
        )
    lines.extend(
        [
            "",
            f"Block-correlated R sensitivity authorized: `{str(authorize).lower()}`.",
            "No proxy was removed and no production or source file was modified.",
            "",
        ]
    )
    output_files[7].write_text("\n".join(lines), encoding="ascii")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
