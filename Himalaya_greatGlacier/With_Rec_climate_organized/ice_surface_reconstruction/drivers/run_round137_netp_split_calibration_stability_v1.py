"""Test NETP paleo trajectory stability under disjoint calibration halves."""

from __future__ import annotations

import calendar
import hashlib
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


CAL_YEARS = np.arange(1940, 2001, dtype=int)
PALEO_YEARS = np.arange(1000, 1921, dtype=int)
PRODUCTS = ("GPCC", "CRU")
SPLITS = (("early", 1940, 1969), ("late", 1970, 2000))
PERIODS = ((1515, 1900), (1800, 1900))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    result: dict[str, object] = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if stat.st_size <= 100_000_000:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
    return result


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return float("nan")
    if np.std(left[valid]) <= 1e-12 or np.std(right[valid]) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def trend_sign(years: np.ndarray, values: np.ndarray) -> int:
    valid = np.isfinite(values)
    if int(valid.sum()) < 3:
        return 0
    return int(np.sign(np.polyfit(years[valid], values[valid], 1)[0]))


def zscore(values: np.ndarray) -> np.ndarray:
    output = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if int(valid.sum()) < 2:
        return output
    scale = float(np.std(values[valid], ddof=1))
    if scale <= 1e-12:
        return output
    output[valid] = (values[valid] - float(np.mean(values[valid]))) / scale
    return output


def aggregate_mjj(monthly: np.ndarray) -> np.ndarray:
    result = np.full((len(CAL_YEARS), monthly.shape[1]), np.nan)
    reshaped = monthly.reshape(len(CAL_YEARS), 12, monthly.shape[1])
    for row, year in enumerate(CAL_YEARS):
        values = reshaped[row, 4:7]
        valid = np.all(np.isfinite(values), axis=0)
        days = np.asarray(
            [calendar.monthrange(int(year), month)[1] for month in (5, 6, 7)],
            dtype=float,
        )
        result[row, valid] = np.sum(values[:, valid], axis=0) / days.sum()
    return result


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    loop = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2"
    )
    output = loop / "round_137_netp_split_calibration_stability_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round137_netp_split_calibration_stability_v1_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")

    base = load_module(
        "round137_base",
        driver / "run_round134_netp_mjj_low_frequency_reconstruction_v2.py",
    )
    r118 = load_module(
        "round137_r118",
        driver
        / "run_climate_reconstruction_loop_v2_round118_documented_moisture_seasonal_incremental_skill_v1.py",
    )
    archive = load_module(
        "round137_archive",
        driver / "audit_positive_b2_event_seasonal_psm_v1.py",
    )
    phyda = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "proxy6": archive.archive_proxy_path(),
        "metadata": root
        / "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/"
        "proxy6_metadata.csv",
        "gpcc": Path(r"D:\VM\VMshare\data\analyses\GPCC\GPCC05\gpcc_1891_2020_05.nc"),
        "cru": phyda
        / "input-recon/cruts.2004151855.v4.04/pre/cru_ts4.04.1901.2019.pre.dat.nc",
        "round134_curves": loop
        / "round_134_netp_mjj_low_frequency_reconstruction_v2/"
        "netp_mjj_reconstruction_curves.csv",
        "round135_curves": loop
        / "round_135_withheld_proxy15_arbitration_v1/"
        "proxy15_round134_qt_curves.csv",
        "round136_summary": loop
        / "round_136_netp_spatial_group_leave_one_out_v1/summary.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path) for name, path in paths.items()}
    round136 = json.loads(paths["round136_summary"].read_text(encoding="utf-8"))
    if not round136["passes_all_gates"] or round136["locked_1921_1939_accessed"]:
        raise RuntimeError("round136-contract-failed")

    metadata = pd.read_csv(paths["metadata"])
    _, groups = base.groups_from_metadata(metadata)
    if len(groups) != 15:
        raise RuntimeError("group-contract-failed")
    proxy_modern = base.read_proxy_values(paths["proxy6"], base.TRAIN_IDS, CAL_YEARS)
    proxy_paleo = base.read_proxy_values(paths["proxy6"], base.TRAIN_IDS, PALEO_YEARS)
    proxy_paleo_smooth = base.centered_rolling(proxy_paleo, 11)

    lat = np.asarray([35, 37, 39, 41] * 8, dtype=float)
    lon = np.repeat(np.arange(93, 109, 2), 4).astype(float)
    weights = np.cos(np.deg2rad(lat))
    targets: dict[str, np.ndarray] = {}
    for product, path, variable, source_start in (
        ("GPCC", paths["gpcc"], "precip", 1891),
        ("CRU", paths["cru"], "pre", 1901),
    ):
        monthly = r118.read_netcdf3_points(path, variable, source_start, lat, lon)
        cells = aggregate_mjj(monthly)
        if cells.shape != (61, 32) or not np.isfinite(cells).all():
            raise RuntimeError(f"modern-target-failed:{product}")
        targets[product] = np.average(cells, axis=1, weights=weights)

    branch_curves: dict[str, np.ndarray] = {}
    calibration_rows = []
    for product in PRODUCTS:
        for split, start, end in SPLITS:
            fit_mask = (CAL_YEARS >= start) & (CAL_YEARS <= end)
            proxy_fit = base.smooth_inside_mask(proxy_modern, fit_mask)
            target_fit = base.smooth_inside_mask(targets[product], fit_mask)
            fit = fit_mask & np.isfinite(target_fit)
            comp_modern, center, scale, signs = base.composite(
                proxy_fit, target_fit, fit, groups
            )
            prediction, coefficients = base.linear_fit_predict(
                comp_modern, target_fit, fit, fit
            )
            comp_paleo, _, _, _ = base.composite(
                proxy_paleo_smooth,
                np.full(len(PALEO_YEARS), np.nan),
                np.zeros(len(PALEO_YEARS), dtype=bool),
                groups,
                center,
                scale,
                signs,
            )
            branch = coefficients[0] + coefficients[1] * comp_paleo
            name = f"{product}_{split}"
            branch_curves[name] = base.centered_rolling(branch, 31)
            calibration_rows.append(
                {
                    "branch": name,
                    "product": product,
                    "calibration_half": split,
                    "fit_years": f"{start}-{end}",
                    "effective_fit_years": f"{start + 5}-{end - 5}",
                    "fit_count": int(fit.sum()),
                    "fit_correlation": correlation(target_fit, prediction),
                    "active_proxy_count": int(np.isfinite(signs).sum()),
                    "intercept": float(coefficients[0]),
                    "slope": float(coefficients[1]),
                }
            )
    calibration = pd.DataFrame(calibration_rows)

    round134 = pd.read_csv(paths["round134_curves"]).set_index("year").reindex(PALEO_YEARS)
    full_product = {
        "GPCC": round134.GPCC_31yr.to_numpy(dtype=float),
        "CRU": round134.CRU_31yr.to_numpy(dtype=float),
        "marginalized": round134.marginalized_31yr.to_numpy(dtype=float),
    }
    validation = pd.read_csv(paths["round135_curves"]).set_index("year").reindex(PALEO_YEARS)
    proxy15 = validation.proxy15_oriented_11_then_31yr.to_numpy(dtype=float)
    qt31 = validation.qt_prec_31yr.to_numpy(dtype=float)

    metric_rows = []
    for name, branch in branch_curves.items():
        product = name.split("_", 1)[0]
        for start, end in PERIODS:
            use = (
                (PALEO_YEARS >= start)
                & (PALEO_YEARS <= end)
                & np.isfinite(branch)
                & np.isfinite(full_product[product])
                & np.isfinite(proxy15)
                & np.isfinite(qt31)
            )
            proxy_qt_r = correlation(proxy15[use], qt31[use])
            proxy_branch_r = correlation(proxy15[use], branch[use])
            metric_rows.append(
                {
                    "branch": name,
                    "product": product,
                    "period": f"{start}_{end}",
                    "common_year_count": int(use.sum()),
                    "branch_same_product_full_correlation": correlation(
                        branch[use], full_product[product][use]
                    ),
                    "branch_full_marginalized_correlation": correlation(
                        branch[use], full_product["marginalized"][use]
                    ),
                    "branch_trend_sign": trend_sign(PALEO_YEARS[use], branch[use]),
                    "proxy15_branch_correlation": proxy_branch_r,
                    "proxy15_qt_correlation": proxy_qt_r,
                    "qt_minus_branch_correlation_delta": proxy_qt_r - proxy_branch_r,
                    "withheld_proxy_favors_branch": proxy_qt_r - proxy_branch_r <= -0.10,
                }
            )
    metrics = pd.DataFrame(metric_rows)
    pair_rows = []
    for product in PRODUCTS:
        for start, end in PERIODS:
            early = branch_curves[f"{product}_early"]
            late = branch_curves[f"{product}_late"]
            use = (
                (PALEO_YEARS >= start)
                & (PALEO_YEARS <= end)
                & np.isfinite(early)
                & np.isfinite(late)
            )
            pair_rows.append(
                {
                    "product": product,
                    "period": f"{start}_{end}",
                    "common_year_count": int(use.sum()),
                    "early_late_correlation": correlation(early[use], late[use]),
                }
            )
    pairs = pd.DataFrame(pair_rows)

    conflict = metrics.loc[metrics.period == "1800_1900"]
    gates = {
        "all_fit_correlations_ge_0_70": bool((calibration.fit_correlation >= 0.70).all()),
        "all_split_same_product_full_correlations_ge_0_70": bool(
            (metrics.branch_same_product_full_correlation >= 0.70).all()
        ),
        "all_early_late_correlations_ge_0_70": bool(
            (pairs.early_late_correlation >= 0.70).all()
        ),
        "all_1800_1900_trends_positive": bool((conflict.branch_trend_sign == 1).all()),
        "all_1800_1900_branches_supported_by_withheld_proxy": bool(
            conflict.withheld_proxy_favors_branch.all()
        ),
    }
    passes = all(gates.values())
    decision = (
        "advance_to_structural_uncertainty_envelope"
        if passes
        else "stop_temporal_candidate_at_split_calibration_instability"
    )

    curves = pd.DataFrame({"year": PALEO_YEARS, **branch_curves})
    curves.to_csv(output / "split_calibration_paleo_curves.csv", index=False)
    calibration.to_csv(output / "split_calibration_fit_metrics.csv", index=False)
    metrics.to_csv(output / "split_calibration_validation_metrics.csv", index=False)
    pairs.to_csv(output / "split_calibration_early_late_pairs.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)
    colors = {
        "GPCC_early": "#0072B2",
        "GPCC_late": "#56B4E9",
        "CRU_early": "#009E73",
        "CRU_late": "#E69F00",
    }
    for axis, start, title in (
        (axes[0], 1515, "NETP split-calibration paleo curves"),
        (axes[1], 1800, "Conflict-period detail"),
    ):
        mask = (PALEO_YEARS >= start) & (PALEO_YEARS <= 1900)
        for name, branch in branch_curves.items():
            axis.plot(
                PALEO_YEARS[mask],
                zscore(branch[mask]),
                color=colors[name],
                linewidth=1.4,
                label=name.replace("_", " "),
            )
        axis.plot(
            PALEO_YEARS[mask],
            zscore(full_product["marginalized"][mask]),
            color="#6A3D9A",
            linewidth=2.2,
            label="full calibration",
        )
        axis.plot(
            PALEO_YEARS[mask],
            zscore(proxy15[mask]),
            color="#222222",
            linewidth=1.7,
            label="withheld proxy 15",
        )
        axis.plot(
            PALEO_YEARS[mask],
            zscore(qt31[mask]),
            color="#CC79A7",
            linewidth=1.5,
            label="QT-Prec",
        )
        axis.axhline(0.0, color="#AAAAAA", linewidth=0.7)
        axis.set(title=title, ylabel="within-curve standardized anomaly")
        axis.legend(ncol=4, frameon=False)
    axes[1].set_xlabel("year")
    fig.savefig(output / "round137_netp_split_calibration_curves.png", dpi=200)
    plt.close(fig)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-file-changed")
    summary = {
        "experiment_id": "round_137_netp_split_calibration_stability_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "minimum_fit_correlation": float(calibration.fit_correlation.min()),
        "minimum_split_same_product_full_correlation": {
            period: float(frame.branch_same_product_full_correlation.min())
            for period, frame in metrics.groupby("period")
        },
        "minimum_early_late_correlation": {
            period: float(frame.early_late_correlation.min())
            for period, frame in pairs.groupby("period")
        },
        "gates": gates,
        "passes_all_gates": passes,
        "decision": decision,
        "locked_1921_1939_accessed": False,
        "frozen_external_validation_reserve_opened": False,
        "enkf_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8"
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Round 137 Final Result\n\n"
        f"- Minimum split fit correlation: `{calibration.fit_correlation.min():.3f}`.\n"
        f"- Minimum split/full correlations: "
        f"`{metrics.groupby('period').branch_same_product_full_correlation.min().to_dict()}`.\n"
        f"- Minimum early/late correlations: "
        f"`{pairs.groupby('period').early_late_correlation.min().to_dict()}`.\n"
        f"- Gates: `{gates}`.\n"
        f"- Final decision: `{decision}`.\n\n"
        "No 1921-1939 value or frozen-reserve value was read, and no EnKF or OGGM run occurred.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
