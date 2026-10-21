"""Audit the source and network structure of the 1901-1920 proxy common mode."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COMBINED_DESKTOP = Path(r"D:\Users\JUNZ\Desktop\select\PHYDA\combined_proxydata6.mat")
COMBINED_ARCHIVE = Path(
    r"E:\2Data\2Rec_ref\a_new_two_proxy\process_past_20000_proxy\combined_proxydata6.mat"
)
PERMUTATIONS = 20000
RNG_SEED = 20260814
MINIMUM_MAPPING_CORRELATION = 0.999999999
MAXIMUM_MAPPING_RESIDUAL = 1e-10


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def archive_root() -> Path:
    archive_folder = "".join(chr(value) for value in (25991, 20214, 25972, 29702))
    paper_folder = "1 " + "".join(chr(value) for value in (35770, 25991))
    return Path(r"D:\Users\JUNZ\Desktop") / archive_folder / paper_folder / "work1" / "compared_result_all"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_char(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset).reshape(-1, order="F")
    return "".join(chr(int(value)) for value in values if int(value)).strip()


def read_proxy_records(path: Path, record_ids: list[int]) -> dict[int, dict[str, object]]:
    records: dict[int, dict[str, object]] = {}
    with h5py.File(path, "r") as handle:
        references = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
        for record_id in record_ids:
            proxy = handle[references[record_id - 1]]
            years = np.asarray(proxy["age_yng"], dtype=float).reshape(-1, order="F")
            values = np.asarray(proxy["data"], dtype=float).reshape(-1, order="F")
            finite = np.isfinite(years) & np.isfinite(values)
            records[record_id] = {
                "years": years[finite].astype(int),
                "values": values[finite],
                "location": np.asarray(proxy["location"], dtype=float).reshape(-1, order="F")[:2],
                "ptype": decode_char(proxy["ptype"]),
                "psm_method": decode_char(proxy["psm_mthd"]),
            }
    return records


def scalar_text(value: object) -> str:
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    return str(value).strip()


def load_combined(path: Path) -> dict[str, object]:
    source = scipy.io.loadmat(path, simplify_cells=True)
    return {
        "years": np.asarray(source["year"], dtype=float).reshape(-1).astype(int),
        "data": np.asarray(source["combined6_data"], dtype=float),
        "latitudes": np.asarray(source["p_lat"], dtype=float).reshape(-1),
        "longitudes": np.asarray(source["p_lon"], dtype=float).reshape(-1),
        "archives": np.atleast_1d(source["archive"]),
        "names": np.asarray(source["dataName"]).reshape(-1),
        "flags": np.asarray(source["source_flag"], dtype=int).reshape(-1),
    }


def align(years: np.ndarray, values: np.ndarray, target_years: np.ndarray) -> np.ndarray:
    lookup = {
        int(year): float(value)
        for year, value in zip(years, values)
        if np.isfinite(year) and np.isfinite(value)
    }
    return np.array([lookup.get(int(year), np.nan) for year in target_years], dtype=float)


def affine_closure(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 10:
        return {"n": int(valid.sum()), "correlation": np.nan, "maximum_abs_residual": np.inf}
    x = right[valid]
    y = left[valid]
    design = np.column_stack([x, np.ones(len(x))])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ np.array([slope, intercept])
    return {
        "n": int(valid.sum()),
        "correlation": float(np.corrcoef(y, x)[0, 1]),
        "maximum_abs_residual": float(np.max(np.abs(residual))),
    }


def map_records(
    records: dict[int, dict[str, object]], combined: dict[str, object], source_name: str
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    latitudes = np.asarray(combined["latitudes"])
    longitudes = np.asarray(combined["longitudes"])
    data = np.asarray(combined["data"])
    years = np.asarray(combined["years"])
    for record_id, record in records.items():
        location = np.asarray(record["location"])
        candidates = np.flatnonzero(
            np.isclose(latitudes, location[0]) & np.isclose(longitudes, location[1])
        )
        matches = []
        for column in candidates:
            candidate = align(years, data[:, column], np.asarray(record["years"]))
            closure = affine_closure(np.asarray(record["values"]), candidate)
            if (
                closure["correlation"] >= MINIMUM_MAPPING_CORRELATION
                and closure["maximum_abs_residual"] <= MAXIMUM_MAPPING_RESIDUAL
            ):
                matches.append((int(column), closure))
        if len(matches) != 1:
            raise RuntimeError(
                f"mapping-not-unique:{source_name}:proxy-{record_id}:"
                f"coordinate-candidates-{len(candidates)}:closure-matches-{len(matches)}"
            )
        column, closure = matches[0]
        rows.append(
            {
                "proxy_index": record_id,
                f"{source_name}_column_1based": column + 1,
                f"{source_name}_mapping_correlation": closure["correlation"],
                f"{source_name}_mapping_max_abs_residual": closure["maximum_abs_residual"],
                "source_flag": int(np.asarray(combined["flags"])[column]),
                "data_name": scalar_text(np.asarray(combined["names"])[column]),
                "archive": scalar_text(np.asarray(combined["archives"])[column]),
            }
        )
    return pd.DataFrame(rows).sort_values("proxy_index")


def source_family(flag: int, name: str) -> str:
    if name in {"PAGES2k2017", "LMRdb1.0", "Mann_2008", "Iso2k"}:
        return name
    if name.endswith(":trsgi_breit"):
        return "source_flag_3_breitenmoser_trsgi"
    if flag == 2:
        return "source_flag_2_named_tree"
    if ":d18O" in name or ":delD" in name:
        return "source_flag_3_isotope"
    return f"source_flag_{flag}_other"


def linear_trend(years: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(values)
    x = years[valid].astype(float)
    return float(100.0 * np.polyfit(x - x.mean(), values[valid], 1)[0])


def eta_squared(values: np.ndarray, codes: np.ndarray, group_count: int) -> float:
    mean = float(np.mean(values))
    total = float(np.sum((values - mean) ** 2))
    if total == 0:
        return 0.0
    counts = np.bincount(codes, minlength=group_count)
    sums = np.bincount(codes, weights=values, minlength=group_count)
    valid = counts > 0
    group_means = sums[valid] / counts[valid]
    between = float(np.sum(counts[valid] * (group_means - mean) ** 2))
    return between / total


def permutation_eta(
    values: np.ndarray, labels: pd.Series, rng: np.random.Generator
) -> tuple[float, float, int]:
    codes, unique = pd.factorize(labels.astype(str), sort=True)
    group_count = len(unique)
    observed = eta_squared(values, codes, group_count)
    exceed = 0
    for _ in range(PERMUTATIONS):
        permuted = rng.permutation(values)
        exceed += eta_squared(permuted, codes, group_count) >= observed
    return observed, float((exceed + 1) / (PERMUTATIONS + 1)), group_count


def main() -> int:
    root = project_root()
    attribution_root = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2"
    )
    stability_root = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_psm_temporal_stability_v2"
    )
    proxy_path = archive_root() / "python_data" / "proxy_data_6.mat"
    paths = {
        "combined_desktop": COMBINED_DESKTOP,
        "combined_archive": COMBINED_ARCHIVE,
        "proxy6": proxy_path,
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
    records = read_proxy_records(proxy_path, selected_ids)
    desktop = map_records(records, load_combined(COMBINED_DESKTOP), "desktop")
    archive = map_records(records, load_combined(COMBINED_ARCHIVE), "archive")
    mapping = desktop.merge(
        archive.drop(columns=["source_flag", "data_name", "archive"]),
        on="proxy_index",
        how="inner",
        validate="one_to_one",
    )
    archive_identity = map_records(records, load_combined(COMBINED_ARCHIVE), "archive_identity")
    identity = desktop[["proxy_index", "source_flag", "data_name", "archive"]].merge(
        archive_identity[["proxy_index", "source_flag", "data_name", "archive"]],
        on="proxy_index",
        suffixes=("_desktop", "_archive"),
        validate="one_to_one",
    )
    identity_matches = (
        identity["source_flag_desktop"].eq(identity["source_flag_archive"])
        & identity["data_name_desktop"].eq(identity["data_name_archive"])
        & identity["archive_desktop"].eq(identity["archive_archive"])
    )
    if not bool(identity_matches.all()):
        raise RuntimeError("combined-source-identity-disagreement")
    if len(mapping) != 147:
        raise RuntimeError("mapping-row-count-mismatch")
    mapping["source_family"] = [
        source_family(int(flag), str(name))
        for flag, name in zip(mapping["source_flag"], mapping["data_name"])
    ]

    trend_table = pd.read_csv(paths["trend_attribution"])
    proxies = mapping.merge(
        trend_table,
        on="proxy_index",
        how="left",
        validate="one_to_one",
    )
    if proxies["contribution_trend_mm_day_per_century"].isna().any():
        raise RuntimeError("trend-attribution-join-failed")
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
    group_dimensions = [
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
    for dimension in group_dimensions:
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
    for dimension in group_dimensions:
        for variable in (
            "innovation_trend_proxy_units_per_century",
            "contribution_trend_mm_day_per_century",
        ):
            observed, p_value, group_count = permutation_eta(
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
    annual_total_trend = linear_trend(
        total_annual["year"].to_numpy(),
        total_annual["hma_increment_contribution_mm_day"].to_numpy(),
    )
    if abs(annual_total_trend - total_trend) > 1e-10:
        raise RuntimeError(
            f"annual-trend-closure-failed:{annual_total_trend}:{total_trend}"
        )
    annual_source = annual.groupby(
        ["year", "source_family"], as_index=False, observed=True
    )["hma_increment_contribution_mm_day"].sum()

    innovation_matrix = annual.pivot(
        index="year", columns="proxy_index", values="innovation"
    ).sort_index()
    if innovation_matrix.isna().any().any():
        raise RuntimeError("innovation-matrix-incomplete")
    centered = innovation_matrix - innovation_matrix.mean(axis=0)
    scales = innovation_matrix.std(axis=0, ddof=1).replace(0, np.nan)
    standardized = centered.divide(scales, axis=1)
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
    loading_table = mapping[["proxy_index", "source_flag", "source_family"]].copy()
    loading_table["pc1_loading"] = [
        loadings[list(innovation_matrix.columns).index(proxy_index)]
        for proxy_index in loading_table["proxy_index"]
    ]
    pc1_yearly = pd.DataFrame(
        {
            "year": innovation_matrix.index.to_numpy(dtype=int),
            "pc1_score": scores,
            "standardized_network_mean": network_mean,
        }
    )

    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_proxy_common_mode_source_audit_v1"
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
    contribution_power = explanatory[
        explanatory["variable"].eq("contribution_trend_mm_day_per_century")
    ].sort_values("eta_squared")
    axes[1, 1].barh(
        contribution_power["dimension"],
        contribution_power["eta_squared"],
        color=np.where(contribution_power["permutation_p_value"] < 0.05, "#d95f02", "#1b9e77"),
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
    source_explanation = explanatory[
        explanatory["dimension"].eq("source_family")
        & explanatory["variable"].eq("contribution_trend_mm_day_per_century")
    ].iloc[0]
    if maximum_source_reduction >= 0.5:
        source_concentration = "one_source_family_can_remove_at_least_half_the_network_trend"
    elif float(source_explanation["permutation_p_value"]) < 0.05:
        source_concentration = "source_families_differ_but_no_single_family_explains_half"
    else:
        source_concentration = "source_family_does_not_explain_the_common_trend"
    image = np.asarray(Image.open(output / "figure_proxy_common_mode_source_audit.png").convert("RGB"))
    after = {name: sha256(path) for name, path in paths.items()}
    summary = {
        "experiment_id": "climate_reconstruction_precipitation_proxy_common_mode_source_audit_v1",
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
        "pc1_trend_per_century": linear_trend(pc1_yearly["year"].to_numpy(), scores),
        "largest_negative_source_family": str(top_source["group"]),
        "largest_negative_source_family_trend": float(
            top_source["group_hma_trend_mm_day_per_century"]
        ),
        "maximum_source_leave_out_magnitude_reduction_fraction": maximum_source_reduction,
        "source_family_contribution_eta_squared": float(source_explanation["eta_squared"]),
        "source_family_contribution_permutation_p_value": float(
            source_explanation["permutation_p_value"]
        ),
        "source_concentration_decision": source_concentration,
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
                "# Precipitation Proxy Common-Mode Source Audit V1",
                "",
                "- Uniquely map all 147 assessed proxies to both combined_proxydata6 sources.",
                "- Diagnose source, spatial, chronology, and PSM group structure.",
                "- Decompose annual 1901-1920 innovations and HMA contributions.",
                "- Use deterministic permutation tests and leave-group-out diagnostics.",
                "- Do not read 1921-1939 or alter any production source.",
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
                "# Precipitation Proxy Common-Mode Source Audit V1",
                "",
                f"- Mapped proxies: `{len(mapping)}`",
                f"- Maximum mapping residual: `{summary['maximum_mapping_residual']:.3e}`",
                f"- Total HMA trend: `{total_trend:.6g} mm/day/century`",
                f"- PC1 explained variance: `{summary['pc1_explained_variance_fraction']:.3f}`",
                f"- Largest negative source family: `{summary['largest_negative_source_family']}`",
                f"- Maximum source leave-out reduction: `{maximum_source_reduction:.3f}`",
                f"- Source-family permutation p-value: `{summary['source_family_contribution_permutation_p_value']:.6g}`",
                f"- Decision: `{source_concentration}`",
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
