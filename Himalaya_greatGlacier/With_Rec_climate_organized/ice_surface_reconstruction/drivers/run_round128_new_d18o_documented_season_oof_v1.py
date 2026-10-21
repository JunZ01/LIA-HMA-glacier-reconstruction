"""Compare annual and published-season PSMs for four new HMA d18O proxies."""

from __future__ import annotations

import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SITES = {
    "mimei": {"latitude": 29.45, "longitude": 96.43, "temperature_months": (6, 7, 8), "precipitation_mode": "JJA"},
    "xinpu": {"latitude": 30.42, "longitude": 95.07, "temperature_months": (6, 7, 8), "precipitation_mode": "JJA"},
    "ganesh": {"latitude": 28.183, "longitude": 85.183, "temperature_months": (6, 7, 8, 9), "precipitation_mode": "JJAS"},
    "keylong": {"latitude": 32.75, "longitude": 76.62, "temperature_months": (6, 7, 8), "precipitation_mode": "ONDJFMA"},
}
PATHS = {
    "bea_self": ("BEA_GPCC", "BEA_GPCC"),
    "cru_self": ("CRU", "CRU"),
    "bea_to_cru": ("BEA_GPCC", "CRU"),
    "cru_to_bea": ("CRU", "BEA_GPCC"),
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module-spec-failed:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def aggregate_cross_year_winter(monthly: np.ndarray, years: np.ndarray) -> np.ndarray:
    cube = monthly.reshape(len(years), 12, monthly.shape[1])
    result = np.full((len(years), monthly.shape[1]), np.nan)
    for row in range(1, len(years)):
        values = np.concatenate([cube[row - 1, 9:12], cube[row, 0:4]], axis=0)
        complete = np.all(np.isfinite(values), axis=0)
        days = 31 + 30 + 31 + 31 + (29 if int(years[row]) % 4 == 0 else 28) + 31 + 30
        result[row, complete] = np.sum(values[:, complete], axis=0) / days
    return result


def oof_predictions(response, products, fit_model, predictor_model, model, r118):
    prediction = np.full(len(r118.YEARS), np.nan)
    baseline = np.full(len(r118.YEARS), np.nan)
    coefficients = []
    for fold, (start, end) in enumerate(r118.BLOCKS, start=1):
        test = (r118.YEARS >= start) & (r118.YEARS <= end)
        train = ~test
        fit_t = products[fit_model][model]["temperature"]
        fit_p = products[fit_model][model]["precipitation"]
        coeff = r118.fit(response, fit_t, fit_p, train)
        pred_t = products[predictor_model][model]["temperature"]
        pred_p = products[predictor_model][model]["precipitation"]
        prediction[test] = r118.predict(coeff, pred_t, pred_p, test)[test]
        valid_train = train & np.isfinite(response) & np.isfinite(fit_t) & np.isfinite(fit_p)
        valid_test = test & np.isfinite(pred_t) & np.isfinite(pred_p)
        if int(valid_train.sum()) >= 30:
            baseline[valid_test] = np.mean(response[valid_train])
        coefficients.append(
            {
                "fold": fold,
                "test_period": f"{start}-{end}",
                "temperature_coefficient": float(coeff[1]) if coeff is not None else np.nan,
                "precipitation_coefficient": float(coeff[2]) if coeff is not None else np.nan,
            }
        )
    return prediction, baseline, coefficients


def pooled(response: np.ndarray, prediction: np.ndarray, baseline: np.ndarray, r118):
    valid = np.isfinite(response) & np.isfinite(prediction) & np.isfinite(baseline)
    if int(valid.sum()) < 30:
        return {"count": int(valid.sum()), "rmse": np.nan, "correlation": np.nan, "skill": np.nan}
    model_sse = float(np.sum((response[valid] - prediction[valid]) ** 2))
    baseline_sse = float(np.sum((response[valid] - baseline[valid]) ** 2))
    return {
        "count": int(valid.sum()),
        "rmse": float(np.sqrt(model_sse / valid.sum())),
        "correlation": r118.correlation(response[valid], prediction[valid]),
        "skill": float(1.0 - model_sse / baseline_sse),
    }


def main() -> int:
    driver = Path(__file__).resolve().parent
    r118 = load_module(
        "round128_r118",
        driver / "run_climate_reconstruction_loop_v2_round118_documented_moisture_seasonal_incremental_skill_v1.py",
    )
    r123 = load_module("round128_r123", driver / "qualify_round123_southeast_tibet_d18o_v1.py")
    r125 = load_module("round128_r125", driver / "qualify_round125_ganesh_keylong_d18o_v1.py")
    root = r118.project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_128_new_d18o_season_oof_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    phyda = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
    r123_dir = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_southeast_tibet_d18o_proxy_qualification_20260818_v1"
    )
    r125_dir = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_ganesh_keylong_d18o_proxy_qualification_20260818_v1"
    )
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "berkeley_temperature": phyda / "input-recon/Land_and_Ocean_LatLong1_1850_2016.nc",
        "cru_temperature": phyda / "input-recon/cruts.2004151855.v4.04/tmp/cru_ts4.04.1901.2019.tmp.dat.nc",
        "cru_precipitation": phyda / "input-recon/cruts.2004151855.v4.04/pre/cru_ts4.04.1901.2019.pre.dat.nc",
        "gpcc_precipitation": Path(r"D:\VM\VMshare\data\analyses\GPCC\GPCC05\gpcc_1891_2020_05.nc"),
        "mimei": r123_dir / "raw_unopened/mimei_tree_ring_d18o.txt",
        "xinpu": r123_dir / "raw_unopened/xinpu_tree_ring_d18o.txt",
        "ganesh": r125_dir / "raw_unopened/ganesh2018d18o-noaa.txt",
        "keylong": r125_dir / "raw_unopened/keylong2020d18o-noaa.txt",
        "r123_summary": r123_dir / "summary.json",
        "r125_summary": r125_dir / "summary.json",
        "contract": driver / "round128_new_d18o_documented_season_oof_v1_contract.md",
        "driver": Path(__file__).resolve(),
        "r118_driver": driver / "run_climate_reconstruction_loop_v2_round118_documented_moisture_seasonal_incremental_skill_v1.py",
        "r123_driver": driver / "qualify_round123_southeast_tibet_d18o_v1.py",
        "r125_driver": driver / "qualify_round125_ganesh_keylong_d18o_v1.py",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    shutil.copy2(paths["contract"], output / "ROUND_CONTRACT.md")
    before = {name: r118.state(path) for name, path in paths.items()}
    for key in ("r123_summary", "r125_summary"):
        prior = json.loads(paths[key].read_text(encoding="utf-8"))
        if prior["status"] != "complete" or not prior["source_hashes_unchanged"]:
            raise RuntimeError(f"prior-source-contract-failed:{key}")
        if prior.get("locked_1921_1939_accessed", False) or prior.get("external_validation_reserve_opened", False):
            raise RuntimeError(f"prior-boundary-contract-failed:{key}")

    responses = {}
    for site in ("mimei", "xinpu"):
        responses[site], audit = r123.parse_candidate(paths[site])
        if audit["locked_1921_1939_values_decoded"]:
            raise RuntimeError(f"locked-candidate-value-decoded:{site}")
    for site in ("ganesh", "keylong"):
        responses[site], audit = r125.parse_noaa_candidate(paths[site], r118.YEARS)
        if audit["locked_1921_1939_values_decoded"]:
            raise RuntimeError(f"locked-candidate-value-decoded:{site}")
    site_names = list(SITES)
    latitudes = np.asarray([SITES[site]["latitude"] for site in site_names], dtype=float)
    longitudes = np.asarray([SITES[site]["longitude"] for site in site_names], dtype=float)
    response = np.column_stack([responses[site] for site in site_names])

    bea_monthly = r118.read_berkeley_points(paths["berkeley_temperature"], latitudes, longitudes)
    cru_t_monthly = r118.read_netcdf3_points(paths["cru_temperature"], "tmp", 1901, latitudes, longitudes)
    cru_p_monthly = r118.read_netcdf3_points(paths["cru_precipitation"], "pre", 1901, latitudes, longitudes)
    gpcc_monthly = r118.read_netcdf3_points(paths["gpcc_precipitation"], "precip", 1891, latitudes, longitudes)
    annual_months = [(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)] * len(site_names)
    temperature_months = [SITES[site]["temperature_months"] for site in site_names]
    precipitation_months = [
        (6, 7, 8) if SITES[site]["precipitation_mode"] == "JJA" else (6, 7, 8, 9)
        for site in site_names
    ]
    annual = {
        "BEA_GPCC": {
            "temperature": r118.center(r118.aggregate_temperature(bea_monthly, annual_months)),
            "precipitation": r118.center(r118.aggregate_precipitation(gpcc_monthly, annual_months)),
        },
        "CRU": {
            "temperature": r118.center(r118.aggregate_temperature(cru_t_monthly, annual_months)),
            "precipitation": r118.center(r118.aggregate_precipitation(cru_p_monthly, annual_months)),
        },
    }
    seasonal = {
        "BEA_GPCC": {
            "temperature": r118.center(r118.aggregate_temperature(bea_monthly, temperature_months)),
            "precipitation": r118.center(r118.aggregate_precipitation(gpcc_monthly, precipitation_months)),
        },
        "CRU": {
            "temperature": r118.center(r118.aggregate_temperature(cru_t_monthly, temperature_months)),
            "precipitation": r118.center(r118.aggregate_precipitation(cru_p_monthly, precipitation_months)),
        },
    }
    keylong_column = site_names.index("keylong")
    seasonal["BEA_GPCC"]["precipitation"][:, keylong_column] = r118.center(
        aggregate_cross_year_winter(gpcc_monthly[:, [keylong_column]], r118.YEARS)
    )[:, 0]
    seasonal["CRU"]["precipitation"][:, keylong_column] = r118.center(
        aggregate_cross_year_winter(cru_p_monthly[:, [keylong_column]], r118.YEARS)
    )[:, 0]
    products = {
        name: {
            "annual": {kind: values[:, column] for kind, values in annual[name].items()},
            "seasonal": {kind: values[:, column] for kind, values in seasonal[name].items()},
        }
        for column, name in []
    }

    metric_rows = []
    coefficient_rows = []
    curve_rows = []
    decisions = {}
    for column, site in enumerate(site_names):
        site_products = {
            product: {
                model: {kind: climate[product][kind][:, column] for kind in ("temperature", "precipitation")}
                for model, climate in (("annual", annual), ("seasonal", seasonal))
            }
            for product in ("BEA_GPCC", "CRU")
        }
        site_metrics = []
        for path_name, (fit_product, predictor_product) in PATHS.items():
            model_outputs = {}
            for model in ("annual", "seasonal"):
                prediction, baseline, coefficients = oof_predictions(
                    response[:, column], site_products, fit_product, predictor_product, model, r118
                )
                current = pooled(response[:, column], prediction, baseline, r118)
                model_outputs[model] = {"prediction": prediction, **current}
                for item in coefficients:
                    coefficient_rows.append(
                        {"site": site, "path": path_name, "fit_product": fit_product, "model": model, **item}
                    )
                for year, observed, predicted in zip(r118.YEARS, response[:, column], prediction):
                    curve_rows.append(
                        {"site": site, "path": path_name, "model": model, "year": int(year), "proxy_d18o": observed, "oof_prediction": predicted}
                    )
            improvement = 1.0 - model_outputs["seasonal"]["rmse"] / model_outputs["annual"]["rmse"]
            row = {
                "site": site,
                "path": path_name,
                "annual_count": model_outputs["annual"]["count"],
                "annual_rmse": model_outputs["annual"]["rmse"],
                "annual_correlation": model_outputs["annual"]["correlation"],
                "annual_skill": model_outputs["annual"]["skill"],
                "seasonal_count": model_outputs["seasonal"]["count"],
                "seasonal_rmse": model_outputs["seasonal"]["rmse"],
                "seasonal_correlation": model_outputs["seasonal"]["correlation"],
                "seasonal_skill": model_outputs["seasonal"]["skill"],
                "seasonal_improvement_over_annual": improvement,
            }
            metric_rows.append(row)
            site_metrics.append(row)
        frame = pd.DataFrame(site_metrics)
        median_improvement = float(np.nanmedian(frame.seasonal_improvement_over_annual))
        median_skill = float(np.nanmedian(frame.seasonal_skill))
        median_correlation = float(np.nanmedian(frame.seasonal_correlation))
        advances = bool(median_improvement >= 0.01 and median_skill > 0 and median_correlation > 0)
        decisions[site] = {
            "decision": "advance_documented_season_operator" if advances else "do_not_advance_documented_season_operator",
            "median_seasonal_improvement_over_annual": median_improvement,
            "median_seasonal_skill": median_skill,
            "median_seasonal_correlation": median_correlation,
            "positive_improvement_path_fraction": float(np.mean(frame.seasonal_improvement_over_annual > 0)),
        }

    metrics = pd.DataFrame(metric_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    curves = pd.DataFrame(curve_rows)
    metrics.to_csv(output / "documented_season_oof_metrics.csv", index=False)
    coefficients.to_csv(output / "fold_coefficients.csv", index=False)
    curves.to_csv(output / "documented_season_oof_curves.csv", index=False)
    pd.DataFrame(
        [{"site": site, **metadata, **decisions[site]} for site, metadata in SITES.items()]
    ).to_csv(output / "candidate_decisions.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for axis, site in zip(axes.flat, site_names):
        subset = curves[(curves.site.eq(site)) & (curves.path.eq("bea_self"))]
        observed = subset[subset.model.eq("annual")]
        annual_curve = subset[subset.model.eq("annual")]
        seasonal_curve = subset[subset.model.eq("seasonal")]
        axis.plot(observed.year, observed.proxy_d18o, color="#111111", linewidth=1.8, label="observed d18O")
        axis.plot(annual_curve.year, annual_curve.oof_prediction, color="#777777", linewidth=1.3, label="annual OOF")
        axis.plot(seasonal_curve.year, seasonal_curve.oof_prediction, color="#0072B2", linewidth=1.5, label="documented-season OOF")
        axis.set_title(f"{site}: {SITES[site]['precipitation_mode']}")
        axis.set_ylabel("tree-ring d18O")
        axis.legend(frameon=False, fontsize=8)
    fig.savefig(output / "round128_new_d18o_season_oof.png", dpi=220)
    plt.close(fig)

    after = {name: r118.state(path) for name, path in paths.items()}
    unchanged = bool(before == after)
    if not unchanged:
        raise RuntimeError("source-changed-during-round128")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "source_hashes_unchanged": unchanged}, indent=2), encoding="utf-8"
    )
    summary = {
        "experiment_id": "round128_new_d18o_documented_season_oof_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "locked_1921_1939_candidate_values_decoded": 0,
        "keylong_1940_predictor_missing_by_contract": True,
        "wache_excluded": True,
        "external_validation_reserve_opened": False,
        "source_hashes_unchanged": unchanged,
        "decisions": decisions,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Round 128 New d18O Documented-Season OOF",
        "",
        "- Locked 1921-1939 accessed: `false`.",
        "- Keylong 1940 predictor omitted: `true`.",
        f"- Source hashes unchanged: `{str(unchanged).lower()}`.",
        "",
    ]
    for site, result in decisions.items():
        lines.extend(
            [
                f"## {site}",
                "",
                f"- Decision: `{result['decision']}`.",
                f"- Median seasonal improvement: `{result['median_seasonal_improvement_over_annual']:.3%}`.",
                f"- Median seasonal OOF skill: `{result['median_seasonal_skill']:.3f}`.",
                f"- Median seasonal OOF correlation: `{result['median_seasonal_correlation']:.3f}`.",
                f"- Positive paths: `{result['positive_improvement_path_fraction']:.1%}`.",
                "",
            ]
        )
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
