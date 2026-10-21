"""Test documented-season temperature in western bivariate proxy PSMs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy.io

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_climate_reconstruction_loop_v2_round118_documented_moisture_seasonal_incremental_skill_v1 as common


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
LOOP_V1 = RESULTS / "climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection"
LOOP_V2 = RESULTS / "climate_reconstruction_improvement_loop_v2"
OUTPUT = RESULTS / "climate_reconstruction_temperature_west_bivariate_documented_season_oof_v1"
PHYDA = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")

SEASON_MONTHS = {
    "JJA": (6, 7, 8),
    "4": (4,),
    "APR": (4,),
}
ANNUAL_LABELS = {"ANN", "annual_months_01_12", "1 2 3 4 5 6 7 8 9 10 11 12"}
PATHS = {
    "bea_self": ("BEA_GPCC", "BEA_GPCC"),
    "cru_self": ("CRU", "CRU"),
    "bea_to_cru": ("BEA_GPCC", "CRU"),
    "cru_to_bea": ("CRU", "BEA_GPCC"),
}


def contract_text() -> str:
    return """# Western bivariate documented-season OOF contract

- Purpose: test whether documented-season temperature, with the original annual precipitation covariate retained, improves western bivariate proxy prediction over annual temperature.
- Selection: proxy6 and proxy9 records already classified as west of 80 E, bivariate, formally documented temperature proxies, and nonannual-season operators.
- Modern period: 1940-2000 only, using the five frozen Round-08 blocked folds.
- Products: Berkeley Earth temperature plus GPCC precipitation, and CRU temperature plus CRU precipitation; score both self-product and cross-product paths.
- Control: annual temperature plus annual precipitation. Candidate: documented-season temperature plus the identical annual precipitation covariate.
- Gates: every network/product self-path median RMSE change must be nonnegative; at least half of records in each network must improve on both self paths; every pooled self/cross path median must be nonnegative; paired BEA/CRU temperature-coefficient signs must agree at least 70 percent.
- This western subset was identified from a paleo anomaly. A pass is mechanism support only and cannot promote a production operator without independent, prospectively selected evidence.
- Do not read 1901-1939 climate/proxy values, run EnKF, modify production files, or run OGGM.
"""


def load_network(network: int, ledger: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    ranking_path = RESULTS / (
        "climate_reconstruction_temperature_west_increment_attribution_1861_1900_v2/"
        f"proxy{network}_west_trend_contribution_ranking.csv"
    )
    ranking = pd.read_csv(ranking_path)
    west = ranking[
        ranking.psm_method.eq("bivariate") & ranking.proxy_location_group.eq("west_lt80")
    ].copy()
    west_ids = set(west.record_index_1based.astype(int))
    selected = ledger[
        ledger.network.eq(f"proxy{network}")
        & ledger.network_record_1based.astype(int).isin(west_ids)
        & ledger.documented_network_role_v4.eq("documented_temperature")
    ].copy()
    selected["season_code"] = selected.canonical_seasonality_v4.astype(str).str.strip()
    selected = selected[~selected.season_code.isin(ANNUAL_LABELS)].copy()
    selected = selected[selected.season_code.isin(SEASON_MONTHS)].sort_values("network_record_1based")
    expected = {6: 37, 9: 49}[network]
    if len(selected) != expected:
        raise RuntimeError(f"proxy{network}-selection-count:{len(selected)}-expected:{expected}")

    proxy_path = LOOP_V1 / f"proxy{network}_allowed_1940_2000_official_baseline_v2.mat"
    data = scipy.io.loadmat(proxy_path, variable_names=["years", "observations", "locations", "model_type"])
    years = np.asarray(data["years"], dtype=int).reshape(-1)
    observations = np.asarray(data["observations"], dtype=float)
    locations = np.asarray(data["locations"], dtype=float)
    model_type = np.asarray(data["model_type"], dtype=int).reshape(-1)
    expected_columns = {6: 202, 9: 297}[network]
    if not np.array_equal(years, common.YEARS) or observations.shape != (61, expected_columns):
        raise RuntimeError(f"proxy{network}-modern-input-contract")
    indices = selected.network_record_1based.to_numpy(int) - 1
    if np.any(model_type[indices] != 2):
        raise RuntimeError(f"proxy{network}-nonbivariate-record-selected")
    return selected, observations[:, indices], locations[indices]


def climate_at_locations(locations: np.ndarray, seasons: list[str], paths: dict[str, Path]) -> dict[str, dict[str, np.ndarray]]:
    latitudes = locations[:, 0]
    longitudes = locations[:, 1]
    season_months = [SEASON_MONTHS[value] for value in seasons]
    annual_months = [tuple(range(1, 13)) for _ in seasons]

    bea = common.read_berkeley_points(paths["berkeley_temperature"], latitudes, longitudes)
    cru_t = common.read_netcdf3_points(paths["cru_temperature"], "tmp", 1901, latitudes, longitudes)
    gpcc = common.read_netcdf3_points(paths["gpcc_precipitation"], "precip", 1891, latitudes, longitudes)
    cru_p = common.read_netcdf3_points(paths["cru_precipitation"], "pre", 1901, latitudes, longitudes)

    return {
        "BEA_GPCC": {
            "annual_temperature": common.center(common.aggregate_temperature(bea, annual_months)),
            "seasonal_temperature": common.center(common.aggregate_temperature(bea, season_months)),
            "annual_precipitation": common.center(common.aggregate_precipitation(gpcc, annual_months)),
        },
        "CRU": {
            "annual_temperature": common.center(common.aggregate_temperature(cru_t, annual_months)),
            "seasonal_temperature": common.center(common.aggregate_temperature(cru_t, season_months)),
            "annual_precipitation": common.center(common.aggregate_precipitation(cru_p, annual_months)),
        },
    }


def evaluate_network(
    network: int,
    selected: pd.DataFrame,
    response: np.ndarray,
    climate: dict[str, dict[str, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    record_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    for column, row in enumerate(selected.itertuples(index=False)):
        predictions = {
            path: {branch: np.full(len(common.YEARS), np.nan) for branch in ("annual", "seasonal")}
            for path in PATHS
        }
        for fold, (start, end) in enumerate(common.BLOCKS, start=1):
            test = (common.YEARS >= start) & (common.YEARS <= end)
            train = ~test
            fitted: dict[tuple[str, str], np.ndarray | None] = {}
            for product in ("BEA_GPCC", "CRU"):
                precipitation = climate[product]["annual_precipitation"][:, column]
                for branch, temperature_name in (
                    ("annual", "annual_temperature"),
                    ("seasonal", "seasonal_temperature"),
                ):
                    temperature = climate[product][temperature_name][:, column]
                    fitted[(product, branch)] = common.fit(
                        response[:, column], temperature, precipitation, train
                    )
                    coefficients = fitted[(product, branch)]
                    coefficient_rows.append(
                        {
                            "network": f"proxy{network}",
                            "record_index_1based": int(row.network_record_1based),
                            "season_code": row.season_code,
                            "fold": fold,
                            "fit_product": product,
                            "branch": branch,
                            "intercept": float(coefficients[0]) if coefficients is not None else np.nan,
                            "temperature_coefficient": float(coefficients[1]) if coefficients is not None else np.nan,
                            "precipitation_coefficient": float(coefficients[2]) if coefficients is not None else np.nan,
                        }
                    )
            for path_name, (fit_product, eval_product) in PATHS.items():
                precipitation = climate[eval_product]["annual_precipitation"][:, column]
                for branch, temperature_name in (
                    ("annual", "annual_temperature"),
                    ("seasonal", "seasonal_temperature"),
                ):
                    temperature = climate[eval_product][temperature_name][:, column]
                    predictions[path_name][branch][test] = common.predict(
                        fitted[(fit_product, branch)], temperature, precipitation, test
                    )[test]

        result: dict[str, object] = {
            "network": f"proxy{network}",
            "record_index_1based": int(row.network_record_1based),
            "evidence_group_id": row.evidence_group_id,
            "season_code": row.season_code,
            "latitude": float(row.latitude),
            "longitude": float(row.longitude),
        }
        self_improved = True
        all_improved = True
        for path_name in PATHS:
            annual_rmse = common.rmse(response[:, column], predictions[path_name]["annual"])
            seasonal_rmse = common.rmse(response[:, column], predictions[path_name]["seasonal"])
            improvement = 1.0 - seasonal_rmse / annual_rmse
            result[f"{path_name}_annual_rmse"] = annual_rmse
            result[f"{path_name}_seasonal_rmse"] = seasonal_rmse
            result[f"{path_name}_improvement"] = improvement
            result[f"{path_name}_annual_correlation"] = common.correlation(
                response[:, column], predictions[path_name]["annual"]
            )
            result[f"{path_name}_seasonal_correlation"] = common.correlation(
                response[:, column], predictions[path_name]["seasonal"]
            )
            all_improved &= bool(np.isfinite(improvement) and improvement > 0)
            if path_name in {"bea_self", "cru_self"}:
                self_improved &= bool(np.isfinite(improvement) and improvement > 0)
        result["both_self_paths_improve"] = self_improved
        result["all_four_paths_improve"] = all_improved
        record_rows.append(result)
    return pd.DataFrame(record_rows), pd.DataFrame(coefficient_rows)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "ROUND_CONTRACT.md").write_text(contract_text(), encoding="utf-8")

    paths = {
        "facts": ROOT / "LIA_PROJECT_FACTS.md",
        "ledger": LOOP_V2 / "round_110_unresolved_proxy_metadata_recovery_v4/proxy_evidence_master_ledger_metadata_recovered.csv",
        "proxy6": LOOP_V1 / "proxy6_allowed_1940_2000_official_baseline_v2.mat",
        "proxy9": LOOP_V1 / "proxy9_allowed_1940_2000_official_baseline_v2.mat",
        "west_proxy6": RESULTS / "climate_reconstruction_temperature_west_increment_attribution_1861_1900_v2/proxy6_west_trend_contribution_ranking.csv",
        "west_proxy9": RESULTS / "climate_reconstruction_temperature_west_increment_attribution_1861_1900_v2/proxy9_west_trend_contribution_ranking.csv",
        "berkeley_temperature": PHYDA / "input-recon/Land_and_Ocean_LatLong1_1850_2016.nc",
        "cru_temperature": PHYDA / "input-recon/cruts.2004151855.v4.04/tmp/cru_ts4.04.1901.2019.tmp.dat.nc",
        "cru_precipitation": PHYDA / "input-recon/cruts.2004151855.v4.04/pre/cru_ts4.04.1901.2019.pre.dat.nc",
        "gpcc_precipitation": Path(r"D:\VM\VMshare\data\analyses\GPCC\GPCC05\gpcc_1891_2020_05.nc"),
        "common_driver": Path(common.__file__).resolve(),
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: common.state(path) for name, path in paths.items()}

    ledger = pd.read_csv(paths["ledger"], low_memory=False)
    all_records = []
    all_coefficients = []
    selected_tables = []
    for network in (6, 9):
        selected, response, locations = load_network(network, ledger)
        climate = climate_at_locations(locations, selected.season_code.tolist(), paths)
        records, coefficients = evaluate_network(network, selected, response, climate)
        all_records.append(records)
        all_coefficients.append(coefficients)
        selected_tables.append(selected.assign(selected_network=f"proxy{network}"))

    records = pd.concat(all_records, ignore_index=True)
    coefficients = pd.concat(all_coefficients, ignore_index=True)
    selected = pd.concat(selected_tables, ignore_index=True)
    records.to_csv(OUTPUT / "record_oof_metrics.csv", index=False)
    coefficients.to_csv(OUTPUT / "fold_coefficients.csv", index=False)
    selected.to_csv(OUTPUT / "selected_proxy_records.csv", index=False)

    path_medians = {
        path: float(np.nanmedian(records[f"{path}_improvement"])) for path in PATHS
    }
    network_summary: dict[str, dict[str, object]] = {}
    for network in ("proxy6", "proxy9"):
        subset = records[records.network.eq(network)]
        network_summary[network] = {
            "record_count": int(len(subset)),
            "season_counts": subset.season_code.value_counts().to_dict(),
            "median_improvements": {
                path: float(np.nanmedian(subset[f"{path}_improvement"])) for path in PATHS
            },
            "fraction_both_self_paths_improve": float(np.mean(subset.both_self_paths_improve)),
            "fraction_all_four_paths_improve": float(np.mean(subset.all_four_paths_improve)),
        }

    seasonal_coefficients = coefficients[coefficients.branch.eq("seasonal")]
    paired = seasonal_coefficients.pivot_table(
        index=["network", "record_index_1based", "fold"],
        columns="fit_product",
        values="temperature_coefficient",
    ).dropna()
    sign_agreement = float(
        np.mean(np.sign(paired.BEA_GPCC.to_numpy()) == np.sign(paired.CRU.to_numpy()))
    )
    gates = {
        **{
            f"{network}_{path}_median_nonnegative": bool(
                network_summary[network]["median_improvements"][path] >= 0
            )
            for network in ("proxy6", "proxy9")
            for path in ("bea_self", "cru_self")
        },
        **{f"pooled_{path}_median_nonnegative": bool(value >= 0) for path, value in path_medians.items()},
        "proxy6_half_improve_on_both_self_paths": bool(
            network_summary["proxy6"]["fraction_both_self_paths_improve"] >= 0.50
        ),
        "proxy9_half_improve_on_both_self_paths": bool(
            network_summary["proxy9"]["fraction_both_self_paths_improve"] >= 0.50
        ),
        "seasonal_temperature_coefficient_product_sign_agreement_at_least_70pct": bool(
            sign_agreement >= 0.70
        ),
    }
    passes = bool(all(gates.values()))
    decision = (
        "seasonal_mismatch_remains_plausible_but_requires_independent_prospective_validation"
        if passes
        else "reject_documented_season_temperature_mismatch_as_primary_western_bivariate_psm_cause"
    )
    summary = {
        "experiment_id": OUTPUT.name,
        "status": "complete",
        "modern_years": [1940, 2000],
        "blocks": list(common.BLOCKS),
        "network_summary": network_summary,
        "pooled_path_median_improvements": path_medians,
        "seasonal_temperature_coefficient_product_sign_agreement": sign_agreement,
        "gates": gates,
        "passes_all_mechanism_support_gates": passes,
        "decision": decision,
        "posthoc_western_subset_cannot_authorize_production_promotion": True,
        "early_1901_1920_accessed": False,
        "locked_1921_1939_accessed": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    labels = []
    values = []
    colors = []
    palette = {"proxy6": "#276FBF", "proxy9": "#B43A35"}
    for network in ("proxy6", "proxy9"):
        subset = records[records.network.eq(network)]
        for path in PATHS:
            labels.append(f"{network}\n{path}")
            values.append(subset[f"{path}_improvement"].to_numpy(float) * 100)
            colors.append(palette[network])
    boxes = axes[0].boxplot(values, labels=labels, showfliers=False, patch_artist=True)
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    axes[0].axhline(0, color="#333333", linewidth=1)
    axes[0].set(ylabel="Seasonal vs annual RMSE improvement (%)", title="Blocked OOF skill")
    axes[0].tick_params(axis="x", rotation=35, labelsize=8)

    for network, marker in (("proxy6", "o"), ("proxy9", "s")):
        subset = records[records.network.eq(network)]
        annual = subset[["bea_self_annual_rmse", "cru_self_annual_rmse"]].mean(axis=1)
        seasonal = subset[["bea_self_seasonal_rmse", "cru_self_seasonal_rmse"]].mean(axis=1)
        axes[1].scatter(annual, seasonal, s=27, alpha=0.75, marker=marker, color=palette[network], label=network)
    limit = float(
        np.nanmax(
            records[[
                "bea_self_annual_rmse", "cru_self_annual_rmse",
                "bea_self_seasonal_rmse", "cru_self_seasonal_rmse",
            ]].to_numpy(float)
        )
    )
    axes[1].plot([0, limit], [0, limit], color="#333333", linewidth=1)
    axes[1].set(
        xlim=(0, limit), ylim=(0, limit), aspect="equal",
        xlabel="Annual-temperature PSM RMSE", ylabel="Seasonal-temperature PSM RMSE",
        title="Self-product mean RMSE",
    )
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "western_bivariate_documented_season_oof.png", dpi=220)
    plt.close(fig)

    after = {name: common.state(path) for name, path in paths.items()}
    unchanged = before == after
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "source_hashes_unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("source-state-changed")
    summary["source_hashes_unchanged"] = True
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# Western Bivariate Documented-Season OOF",
        "",
        f"- proxy6 records: `{network_summary['proxy6']['record_count']}`; proxy9 records: `{network_summary['proxy9']['record_count']}`.",
        f"- Pooled median RMSE changes: `{json.dumps(path_medians, sort_keys=True)}`.",
        f"- Fractions improving on both self-product paths: proxy6 `{network_summary['proxy6']['fraction_both_self_paths_improve']:.3%}`, proxy9 `{network_summary['proxy9']['fraction_both_self_paths_improve']:.3%}`.",
        f"- Seasonal temperature coefficient BEA/CRU sign agreement: `{sign_agreement:.3%}`.",
        f"- All mechanism-support gates pass: `{str(passes).lower()}`.",
        "",
        f"Decision: `{decision}`.",
        "",
        "This is a modern blocked-OOF mechanism test of a paleo-outcome-defined western subset. It cannot authorize production promotion. No 1901-1939 value, EnKF, or OGGM run was used.",
    ]
    (OUTPUT / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
