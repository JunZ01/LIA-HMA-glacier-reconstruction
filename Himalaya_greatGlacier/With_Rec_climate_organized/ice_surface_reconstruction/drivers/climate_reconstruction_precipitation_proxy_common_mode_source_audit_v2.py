"""Run the corrected proxy common-mode source audit with metadata joining."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PERMUTATIONS = 20000
RNG_SEED = 20260814


def load_base() -> object:
    path = Path(__file__).with_name(
        "climate_reconstruction_precipitation_proxy_common_mode_source_audit_v1.py"
    )
    spec = importlib.util.spec_from_file_location("common_mode_source_audit_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("base-driver-import-failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    base = load_base()
    root = base.project_root()
    attribution_root = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2"
    )
    stability_root = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_psm_temporal_stability_v2"
    )
    proxy_path = base.archive_root() / "python_data" / "proxy_data_6.mat"
    paths = {
        "combined_desktop": base.COMBINED_DESKTOP,
        "combined_archive": base.COMBINED_ARCHIVE,
        "proxy6": proxy_path,
        "metadata": attribution_root / "proxy6_metadata.csv",
        "trend_attribution": attribution_root / "proxy_trend_contribution_1901_1920.csv",
        "annual_attribution": attribution_root / "annual_proxy_contributions_1901_1920.csv",
        "stability_proxy_table": stability_root / "psm_temporal_stability_proxy_level.csv",
    }
    if missing := [str(path) for path in paths.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: sha256(path) for name, path in paths.items()}

    stability = pd.read_csv(paths["stability_proxy_table"])
    selected_ids = sorted(stability["proxy_index"].astype(int).unique().tolist())
    if len(selected_ids) != 147:
        raise RuntimeError(f"unexpected-selected-proxy-count:{len(selected_ids)}")
    records = base.read_proxy_records(proxy_path, selected_ids)
    desktop = base.map_records(
        records, base.load_combined(base.COMBINED_DESKTOP), "desktop"
    )
    archive = base.map_records(
        records, base.load_combined(base.COMBINED_ARCHIVE), "archive"
    )
    joined = desktop.merge(
        archive,
        on="proxy_index",
        suffixes=("_desktop", "_archive"),
        validate="one_to_one",
    )
    identity_matches = (
        joined["source_flag_desktop"].eq(joined["source_flag_archive"])
        & joined["data_name_desktop"].eq(joined["data_name_archive"])
        & joined["archive_desktop"].eq(joined["archive_archive"])
    )
    if not bool(identity_matches.all()):
        raise RuntimeError("combined-source-identity-disagreement")
    mapping = joined[
        [
            "proxy_index",
            "desktop_column_1based",
            "desktop_mapping_correlation",
            "desktop_mapping_max_abs_residual",
            "archive_column_1based",
            "archive_mapping_correlation",
            "archive_mapping_max_abs_residual",
            "source_flag_desktop",
            "data_name_desktop",
            "archive_desktop",
        ]
    ].rename(
        columns={
            "source_flag_desktop": "source_flag",
            "data_name_desktop": "data_name",
            "archive_desktop": "archive",
        }
    )
    if len(mapping) != 147:
        raise RuntimeError("mapping-row-count-mismatch")
    mapping["source_family"] = [
        base.source_family(int(flag), str(name))
        for flag, name in zip(mapping["source_flag"], mapping["data_name"])
    ]

    trend_table = pd.read_csv(paths["trend_attribution"])
    metadata = pd.read_csv(paths["metadata"])[
        [
            "proxy_index",
            "coefficient_b0",
            "coefficient_b1",
            "coefficient_b2",
            "first_observed_year",
            "last_observed_year",
            "observed_year_count",
        ]
    ]
    proxies = (
        mapping.merge(trend_table, on="proxy_index", how="left", validate="one_to_one")
        .merge(metadata, on="proxy_index", how="left", validate="one_to_one")
    )
    required = [
        "contribution_trend_mm_day_per_century",
        "first_observed_year",
        "last_observed_year",
        "observed_year_count",
        "coefficient_b1",
        "coefficient_b2",
    ]
    if proxies[required].isna().any().any():
        raise RuntimeError("proxy-metadata-join-failed")
    proxies["first_year_quartile"] = pd.qcut(
        proxies["first_observed_year"], 4, duplicates="drop"
    ).astype(str)
    proxies["record_length_quartile"] = pd.qcut(
        proxies["observed_year_count"], 4, duplicates="drop"
    ).astype(str)
    proxies["last_year_bin"] = pd.cut(
        proxies["last_observed_year"],
        [-np.inf, 1980, 1990, 2000, np.inf],
        labels=["through_1980", "1981_1990", "1991_2000", "after_2000"],
    ).astype(str)
    proxies["b1_temperature_sign"] = np.where(
        proxies["coefficient_b1"] >= 0, "nonnegative", "negative"
    )
    proxies["b2_precipitation_sign"] = np.where(
        proxies["coefficient_b2"] >= 0, "nonnegative", "negative"
    )
    proxies["site_multiplicity"] = np.where(
        proxies["site_record_count"] > 1, "multi_record_site", "single_record_site"
    )

    total_trend = float(proxies["contribution_trend_mm_day_per_century"].sum())
    total_absolute = float(proxies["contribution_trend_mm_day_per_century"].abs().sum())
    dimensions = [
        "source_flag",
        "source_family",
        "longitude_band",
        "site_class",
        "site_multiplicity",
        "psm_significance",
        "first_year_quartile",
        "record_length_quartile",
        "last_year_bin",
        "b1_temperature_sign",
        "b2_precipitation_sign",
    ]
    group_rows: list[dict[str, object]] = []
    for dimension in dimensions:
        for group, subset in proxies.groupby(dimension, observed=True):
            group_trend = float(subset["contribution_trend_mm_day_per_century"].sum())
            remaining = total_trend - group_trend
            group_rows.append(
                {
                    "dimension": dimension,
                    "group": str(group),
                    "proxy_count": int(len(subset)),
                    "group_hma_trend_mm_day_per_century": group_trend,
                    "signed_fraction_of_total": group_trend / total_trend,
                    "absolute_contribution_share": float(
                        subset["contribution_trend_mm_day_per_century"].abs().sum()
                        / total_absolute
                    ),
                    "mean_innovation_trend_proxy_units_per_century": float(
                        subset["innovation_trend_proxy_units_per_century"].mean()
                    ),
                    "median_innovation_trend_proxy_units_per_century": float(
                        subset["innovation_trend_proxy_units_per_century"].median()
                    ),
                    "mean_hma_gain_weight": float(subset["mean_hma_gain_weight"].mean()),
                    "leave_group_out_hma_trend_mm_day_per_century": remaining,
                    "leave_group_out_magnitude_reduction_fraction": 1.0
                    - abs(remaining) / abs(total_trend),
                }
            )
    group_summary = pd.DataFrame(group_rows)

    rng = np.random.default_rng(RNG_SEED)
    explanatory_rows: list[dict[str, object]] = []
    for dimension in dimensions:
        for variable in (
            "innovation_trend_proxy_units_per_century",
            "contribution_trend_mm_day_per_century",
        ):
            observed, p_value, group_count = base.permutation_eta(
                proxies[variable].to_numpy(dtype=float), proxies[dimension], rng
            )
            explanatory_rows.append(
                {
                    "dimension": dimension,
                    "variable": variable,
                    "group_count": group_count,
                    "eta_squared": observed,
                    "permutation_p_value": p_value,
                    "permutations": PERMUTATIONS,
                    "rng_seed": RNG_SEED,
                }
            )
    explanatory = pd.DataFrame(explanatory_rows)

    annual = pd.read_csv(paths["annual_attribution"])
    annual = annual[annual["proxy_index"].isin(selected_ids)].merge(
        mapping[["proxy_index", "source_flag", "source_family"]],
        on="proxy_index",
        how="left",
        validate="many_to_one",
    )
    if len(annual) != 20 * 147:
        raise RuntimeError(f"unexpected-annual-row-count:{len(annual)}")
    total_annual = annual.groupby("year", as_index=False)[
        "hma_increment_contribution_mm_day"
    ].sum()
    annual_total_trend = base.linear_trend(
        total_annual["year"].to_numpy(),
        total_annual["hma_increment_contribution_mm_day"].to_numpy(),
    )
    if abs(annual_total_trend - total_trend) > 1e-10:
        raise RuntimeError("annual-trend-closure-failed")
    annual_source = annual.groupby(
        ["year", "source_family"], as_index=False, observed=True
    )["hma_increment_contribution_mm_day"].sum()

    innovation_matrix = annual.pivot(
        index="year", columns="proxy_index", values="innovation"
    ).sort_index()
    if innovation_matrix.shape != (20, 147) or innovation_matrix.isna().any().any():
        raise RuntimeError("innovation-matrix-gate-failed")
    standardized = (innovation_matrix - innovation_matrix.mean(axis=0)).divide(
        innovation_matrix.std(axis=0, ddof=1).replace(0, np.nan), axis=1
    )
    if standardized.isna().any().any():
        raise RuntimeError("innovation-standardization-failed")
    u, singular_values, vt = np.linalg.svd(standardized.to_numpy(), full_matrices=False)
    scores = u[:, 0] * singular_values[0]
    loadings = vt[0].copy()
    network_mean = standardized.mean(axis=1).to_numpy()
    if np.corrcoef(scores, network_mean)[0, 1] < 0:
        scores *= -1
        loadings *= -1
    explained = singular_values**2 / np.sum(singular_values**2)
    loading_lookup = dict(zip(innovation_matrix.columns.astype(int), loadings))
    loading_table = mapping[["proxy_index", "source_flag", "source_family"]].copy()
    loading_table["pc1_loading"] = loading_table["proxy_index"].map(loading_lookup)
    pc1_yearly = pd.DataFrame(
        {
            "year": innovation_matrix.index.to_numpy(dtype=int),
            "pc1_score": scores,
            "standardized_network_mean": network_mean,
        }
    )

    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_proxy_common_mode_source_audit_v2"
    )
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    output.mkdir(parents=True)
    mapping.to_csv(output / "proxy_source_mapping.csv", index=False)
    proxies.to_csv(output / "proxy_source_trend_table.csv", index=False)
    group_summary.to_csv(output / "group_contribution_summary.csv", index=False)
    explanatory.to_csv(output / "grouping_explanatory_power.csv", index=False)
    annual_source.to_csv(output / "annual_source_family_contributions.csv", index=False)
    total_annual.to_csv(output / "annual_total_hma_contribution.csv", index=False)
    loading_table.to_csv(output / "innovation_pc1_loadings.csv", index=False)
    pc1_yearly.to_csv(output / "innovation_pc1_yearly.csv", index=False)

    source_groups = group_summary[group_summary["dimension"].eq("source_family")].sort_values(
        "group_hma_trend_mm_day_per_century"
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes[0, 0].barh(
        source_groups["group"],
        source_groups["group_hma_trend_mm_day_per_century"],
        color="#4c78a8",
    )
    axes[0, 0].axvline(0, color="#333333", lw=0.8)
    axes[0, 0].set_xlabel("HMA trend contribution (mm/day/century)")
    axes[0, 0].set_title("Contribution by mapped source family")
    axes[0, 0].grid(axis="x", alpha=0.2)
    for family, subset in annual_source.groupby("source_family", observed=True):
        axes[0, 1].plot(
            subset["year"],
            subset["hma_increment_contribution_mm_day"],
            marker="o",
            ms=2.5,
            lw=1.0,
            label=family,
        )
    axes[0, 1].set_title("Annual HMA contribution by source family")
    axes[0, 1].set_ylabel("mm/day")
    axes[0, 1].grid(alpha=0.2)
    axes[0, 1].legend(fontsize=7, frameon=False)
    axes[1, 0].plot(pc1_yearly["year"], pc1_yearly["pc1_score"], marker="o", label="Innovation PC1")
    axes[1, 0].plot(pc1_yearly["year"], pc1_yearly["standardized_network_mean"], marker="s", label="Network mean")
    axes[1, 0].set_title("Proxy-network common mode")
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend(frameon=False)
    power = explanatory[
        explanatory["variable"].eq("contribution_trend_mm_day_per_century")
    ].sort_values("eta_squared")
    axes[1, 1].barh(
        power["dimension"],
        power["eta_squared"],
        color=np.where(power["permutation_p_value"] < 0.05, "#d95f02", "#1b9e77"),
    )
    axes[1, 1].set_xlabel("Eta-squared")
    axes[1, 1].set_title("Grouping explanation of contribution trend")
    axes[1, 1].grid(axis="x", alpha=0.2)
    figure.savefig(output / "figure_proxy_common_mode_source_audit.png", dpi=220)
    plt.close(figure)

    top_source = source_groups.iloc[0]
    maximum_source_reduction = float(
        source_groups["leave_group_out_magnitude_reduction_fraction"].max()
    )
    source_test = explanatory[
        explanatory["dimension"].eq("source_family")
        & explanatory["variable"].eq("contribution_trend_mm_day_per_century")
    ].iloc[0]
    if maximum_source_reduction >= 0.5:
        decision = "one_source_family_can_remove_at_least_half_the_network_trend"
    elif float(source_test["permutation_p_value"]) < 0.05:
        decision = "source_families_differ_but_no_single_family_explains_half"
    else:
        decision = "source_family_does_not_explain_the_common_trend"
    image = np.asarray(Image.open(output / "figure_proxy_common_mode_source_audit.png").convert("RGB"))
    after = {name: sha256(path) for name, path in paths.items()}
    summary = {
        "experiment_id": "climate_reconstruction_precipitation_proxy_common_mode_source_audit_v2",
        "date": "2026-08-14",
        "period": [1901, 1920],
        "locked_final_period_accessed": False,
        "assessed_proxy_count": 147,
        "mapping_unique_in_both_combined_sources": True,
        "combined_source_identity_matches": True,
        "maximum_mapping_residual": float(
            max(
                mapping["desktop_mapping_max_abs_residual"].max(),
                mapping["archive_mapping_max_abs_residual"].max(),
            )
        ),
        "total_hma_trend_mm_day_per_century": total_trend,
        "annual_trend_closure_error": annual_total_trend - total_trend,
        "pc1_explained_variance_fraction": float(explained[0]),
        "pc1_loading_positive_fraction": float(np.mean(loadings > 0)),
        "pc1_trend_per_century": base.linear_trend(pc1_yearly["year"].to_numpy(), scores),
        "largest_negative_source_family": str(top_source["group"]),
        "largest_negative_source_family_trend": float(top_source["group_hma_trend_mm_day_per_century"]),
        "maximum_source_leave_out_magnitude_reduction_fraction": maximum_source_reduction,
        "source_family_contribution_eta_squared": float(source_test["eta_squared"]),
        "source_family_contribution_permutation_p_value": float(source_test["permutation_p_value"]),
        "source_concentration_decision": decision,
        "source_hashes_unchanged": before == after,
        "figure_nonwhite_fraction": float(np.mean(np.any(image < 250, axis=2))),
        "decision_boundary": "Source and leave-group-out results are diagnostics only. They do not authorize proxy deletion, PSM replacement, or an EnKF production rerun.",
    }
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8"
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "EXPERIMENT.md").write_text(
        "\n".join(
            [
                "# Precipitation Proxy Common-Mode Source Audit V2",
                "",
                "- Uniquely map all 147 assessed proxies in both combined sources.",
                "- Join proxy6 metadata explicitly before chronology grouping.",
                "- Diagnose source, spatial, chronology, and PSM group structure.",
                "- Use 20,000 deterministic permutations per grouping test.",
                "- Do not read 1921-1939 or alter production files.",
                "",
                summary["decision_boundary"],
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# Precipitation Proxy Common-Mode Source Audit V2",
                "",
                f"- Mapped proxies: `{len(mapping)}`",
                f"- Maximum mapping residual: `{summary['maximum_mapping_residual']:.3e}`",
                f"- Total HMA trend: `{total_trend:.6g} mm/day/century`",
                f"- PC1 explained variance: `{summary['pc1_explained_variance_fraction']:.3f}`",
                f"- Largest negative source family: `{summary['largest_negative_source_family']}`",
                f"- Maximum source leave-out reduction: `{maximum_source_reduction:.3f}`",
                f"- Source-family permutation p-value: `{summary['source_family_contribution_permutation_p_value']:.6g}`",
                f"- Decision: `{decision}`",
                f"- Source hashes unchanged: `{before == after}`",
                "",
                summary["decision_boundary"],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
