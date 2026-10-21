"""Leave out each Round 134 NETP proxy spatial group in turn."""

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


def reconstruct(
    base,
    proxy_modern_smooth: np.ndarray,
    proxy_paleo_smooth: np.ndarray,
    target: np.ndarray,
    groups: list[np.ndarray],
) -> tuple[np.ndarray, float]:
    target_smooth = base.centered_rolling(target, 11)
    fit = np.isfinite(target_smooth)
    comp_modern, center, scale, signs = base.composite(
        proxy_modern_smooth, target_smooth, fit, groups
    )
    full_prediction, coefficients = base.linear_fit_predict(
        comp_modern,
        target_smooth,
        fit,
        np.ones(len(CAL_YEARS), dtype=bool),
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
    paleo = coefficients[0] + coefficients[1] * comp_paleo
    return paleo, correlation(target_smooth, full_prediction)


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    loop = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2"
    )
    output = loop / "round_136_netp_spatial_group_leave_one_out_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round136_netp_spatial_group_leave_one_out_v1_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")

    base = load_module(
        "round136_base",
        driver / "run_round134_netp_mjj_low_frequency_reconstruction_v2.py",
    )
    r118 = load_module(
        "round136_r118",
        driver
        / "run_climate_reconstruction_loop_v2_round118_documented_moisture_seasonal_incremental_skill_v1.py",
    )
    archive = load_module(
        "round136_archive",
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
        "round135_summary": loop
        / "round_135_withheld_proxy15_arbitration_v1/summary.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path) for name, path in paths.items()}

    round135_summary = json.loads(paths["round135_summary"].read_text(encoding="utf-8"))
    if not round135_summary["informative_for_arbitration"]:
        raise RuntimeError("round135-informativeness-contract-failed")
    if round135_summary["locked_1921_1939_accessed"]:
        raise RuntimeError("round135-boundary-failed")

    metadata = pd.read_csv(paths["metadata"])
    group_keys, groups = base.groups_from_metadata(metadata)
    if len(group_keys) != 15 or len(groups) != 15:
        raise RuntimeError("round134-group-identity-failed")
    proxy_modern = base.read_proxy_values(paths["proxy6"], base.TRAIN_IDS, CAL_YEARS)
    proxy_paleo = base.read_proxy_values(paths["proxy6"], base.TRAIN_IDS, PALEO_YEARS)
    proxy_modern_smooth = base.centered_rolling(proxy_modern, 11)
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

    full_predictions: dict[str, np.ndarray] = {}
    full_fit_correlations: dict[str, float] = {}
    for product in PRODUCTS:
        full_predictions[product], full_fit_correlations[product] = reconstruct(
            base,
            proxy_modern_smooth,
            proxy_paleo_smooth,
            targets[product],
            groups,
        )
    full_predictions["marginalized"] = np.nanmean(
        np.column_stack((full_predictions["GPCC"], full_predictions["CRU"])), axis=1
    )
    full31 = {
        name: base.centered_rolling(values, 31)
        for name, values in full_predictions.items()
    }

    round134 = pd.read_csv(paths["round134_curves"]).set_index("year").reindex(PALEO_YEARS)
    closure = {
        "GPCC": float(
            np.nanmax(np.abs(full_predictions["GPCC"] - round134.GPCC_mjj_mm_day.to_numpy()))
        ),
        "CRU": float(
            np.nanmax(np.abs(full_predictions["CRU"] - round134.CRU_mjj_mm_day.to_numpy()))
        ),
        "marginalized": float(
            np.nanmax(
                np.abs(
                    full_predictions["marginalized"]
                    - round134.marginalized_mjj_mm_day.to_numpy()
                )
            )
        ),
    }

    validation = pd.read_csv(paths["round135_curves"]).set_index("year").reindex(PALEO_YEARS)
    proxy15 = validation.proxy15_oriented_11_then_31yr.to_numpy(dtype=float)
    qt31 = validation.qt_prec_31yr.to_numpy(dtype=float)

    branch_curves = {"year": PALEO_YEARS}
    metric_rows = []
    fit_rows = []
    group_rows = []
    for omit_index, (key, omitted_members) in enumerate(zip(group_keys, groups)):
        kept_groups = [group for index, group in enumerate(groups) if index != omit_index]
        branch_product: dict[str, np.ndarray] = {}
        for product in PRODUCTS:
            branch_product[product], fit_r = reconstruct(
                base,
                proxy_modern_smooth,
                proxy_paleo_smooth,
                targets[product],
                kept_groups,
            )
            fit_rows.append(
                {
                    "omitted_group_index": omit_index + 1,
                    "omitted_group_latitude": key[0],
                    "omitted_group_longitude": key[1],
                    "product": product,
                    "modern_full_fit_correlation": fit_r,
                }
            )
        marginalized = np.nanmean(
            np.column_stack((branch_product["GPCC"], branch_product["CRU"])), axis=1
        )
        branch31 = base.centered_rolling(marginalized, 31)
        branch_name = f"omit_group_{omit_index + 1:02d}"
        branch_curves[branch_name] = branch31
        member_ids = [base.TRAIN_IDS[index] for index in omitted_members]
        group_rows.append(
            {
                "omitted_group_index": omit_index + 1,
                "group_latitude": key[0],
                "group_longitude": key[1],
                "proxy_indices": ",".join(str(value) for value in member_ids),
                "proxy_count": len(member_ids),
            }
        )
        for start, end in PERIODS:
            use = (
                (PALEO_YEARS >= start)
                & (PALEO_YEARS <= end)
                & np.isfinite(branch31)
                & np.isfinite(full31["marginalized"])
                & np.isfinite(proxy15)
                & np.isfinite(qt31)
            )
            proxy_qt_r = correlation(proxy15[use], qt31[use])
            proxy_branch_r = correlation(proxy15[use], branch31[use])
            metric_rows.append(
                {
                    "omitted_group_index": omit_index + 1,
                    "omitted_group_latitude": key[0],
                    "omitted_group_longitude": key[1],
                    "period": f"{start}_{end}",
                    "common_year_count": int(use.sum()),
                    "branch_full_correlation": correlation(
                        branch31[use], full31["marginalized"][use]
                    ),
                    "branch_trend_sign": trend_sign(PALEO_YEARS[use], branch31[use]),
                    "full_trend_sign": trend_sign(
                        PALEO_YEARS[use], full31["marginalized"][use]
                    ),
                    "proxy15_branch_correlation": proxy_branch_r,
                    "proxy15_qt_correlation": proxy_qt_r,
                    "qt_minus_branch_correlation_delta": proxy_qt_r - proxy_branch_r,
                    "withheld_proxy_favors_branch": proxy_qt_r - proxy_branch_r <= -0.10,
                }
            )

    metrics = pd.DataFrame(metric_rows)
    fits = pd.DataFrame(fit_rows)
    groups_frame = pd.DataFrame(group_rows)
    curves = pd.DataFrame(branch_curves)
    closure_pass = max(closure.values()) <= 1e-12
    modern_fit_pass = bool((fits.modern_full_fit_correlation >= 0.70).all())
    branch_correlation_pass = bool((metrics.branch_full_correlation >= 0.90).all())
    conflict = metrics.loc[metrics.period == "1800_1900"]
    trend_pass = bool(
        (conflict.branch_trend_sign == 1).all()
        and (conflict.branch_trend_sign == conflict.full_trend_sign).all()
    )
    withheld_support_count = int(conflict.withheld_proxy_favors_branch.sum())
    withheld_pass = withheld_support_count >= 13
    gates = {
        "exact_round134_full_curve_closure": closure_pass,
        "all_15_branches_completed": len(conflict) == 15,
        "all_modern_full_fit_correlations_ge_0_70": modern_fit_pass,
        "all_branch_full_correlations_ge_0_90": branch_correlation_pass,
        "all_1800_1900_trends_positive_and_match_full": trend_pass,
        "at_least_13_branches_supported_by_withheld_proxy": withheld_pass,
    }
    passes = all(gates.values())
    decision = (
        "advance_fixed_temporal_candidate_to_method_uncertainty_validation"
        if passes
        else "do_not_advance_network_sensitive_temporal_candidate"
    )

    metrics.to_csv(output / "leave_one_group_out_metrics.csv", index=False)
    fits.to_csv(output / "leave_one_group_out_modern_fit.csv", index=False)
    groups_frame.to_csv(output / "omitted_group_inventory.csv", index=False)
    curves.to_csv(output / "leave_one_group_out_curves.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)
    for axis, start, title in (
        (axes[0], 1515, "NETP spatial-group leave-one-out curves"),
        (axes[1], 1800, "Conflict-period detail"),
    ):
        mask = (PALEO_YEARS >= start) & (PALEO_YEARS <= 1900)
        for column in curves.columns[1:]:
            axis.plot(
                PALEO_YEARS[mask],
                zscore(curves[column].to_numpy(dtype=float)[mask]),
                color="#999999",
                alpha=0.35,
                linewidth=0.9,
            )
        axis.plot(
            PALEO_YEARS[mask],
            zscore(full31["marginalized"][mask]),
            color="#0072B2",
            linewidth=2.2,
            label="full Round 134",
        )
        axis.plot(
            PALEO_YEARS[mask],
            zscore(proxy15[mask]),
            color="#222222",
            linewidth=1.8,
            label="withheld proxy 15",
        )
        axis.plot(
            PALEO_YEARS[mask],
            zscore(qt31[mask]),
            color="#CC79A7",
            linewidth=1.6,
            label="QT-Prec",
        )
        axis.axhline(0.0, color="#AAAAAA", linewidth=0.7)
        axis.set(title=title, ylabel="within-curve standardized anomaly")
        axis.legend(ncol=3, frameon=False)
    axes[1].set_xlabel("year")
    fig.savefig(output / "round136_netp_spatial_group_leave_one_out_curves.png", dpi=200)
    plt.close(fig)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-file-changed")
    summary = {
        "experiment_id": "round_136_netp_spatial_group_leave_one_out_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "round134_full_curve_max_abs_closure": closure,
        "minimum_modern_full_fit_correlation": float(
            fits.modern_full_fit_correlation.min()
        ),
        "minimum_branch_full_correlation": {
            period: float(frame.branch_full_correlation.min())
            for period, frame in metrics.groupby("period")
        },
        "withheld_support_count_1800_1900": withheld_support_count,
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
        "# Round 136 Final Result\n\n"
        f"- Exact Round 134 closure: `{closure_pass}`; max differences `{closure}`.\n"
        f"- Minimum modern full-fit correlation: `{fits.modern_full_fit_correlation.min():.3f}`.\n"
        f"- Minimum leave-one-group-out/full correlation: "
        f"`{metrics.groupby('period').branch_full_correlation.min().to_dict()}`.\n"
        f"- Positive conflict-period trends retained: `{trend_pass}`.\n"
        f"- Withheld proxy supports `{withheld_support_count}/15` branches over 1800-1900.\n"
        f"- Final decision: `{decision}`.\n\n"
        "No 1921-1939 value or frozen-reserve value was read, and no EnKF or OGGM run occurred.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
