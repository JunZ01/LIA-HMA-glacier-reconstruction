"""Arbitrate the Round 134/QT-Prec conflict with withheld NETP proxy 15."""

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
from scipy.io import loadmat

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODERN_YEARS = np.arange(1940, 2001, dtype=int)
PALEO_YEARS = np.arange(1000, 1921, dtype=int)
PROXY_ID = 15
EXPECTED_PUBLIC_ID = "Asi-GHEGAN.PAGES2k.2013"
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


def trend_sign(years: np.ndarray, values: np.ndarray) -> int:
    valid = np.isfinite(values)
    if int(valid.sum()) < 3:
        return 0
    return int(np.sign(np.polyfit(years[valid], values[valid], 1)[0]))


def aggregate_mjj(monthly: np.ndarray) -> np.ndarray:
    result = np.full((len(MODERN_YEARS), monthly.shape[1]), np.nan)
    reshaped = monthly.reshape(len(MODERN_YEARS), 12, monthly.shape[1])
    for row, year in enumerate(MODERN_YEARS):
        values = reshaped[row, 4:7]
        valid = np.all(np.isfinite(values), axis=0)
        days = np.asarray(
            [calendar.monthrange(int(year), month)[1] for month in (5, 6, 7)],
            dtype=float,
        )
        result[row, valid] = np.sum(values[:, valid], axis=0) / days.sum()
    return result


def period_label(delta: float) -> str:
    if delta >= 0.10:
        return "favors_qt_prec"
    if delta <= -0.10:
        return "favors_round134_composite"
    return "nondiscriminating"


def overall_decision(informative: bool, labels: dict[str, str]) -> str:
    if not informative:
        return "withheld_proxy_not_stable_enough_to_arbitrate"
    long_label = labels["1515_1900"]
    conflict_label = labels["1800_1900"]
    if long_label == conflict_label and long_label != "nondiscriminating":
        return "withheld_proxy_supports_" + long_label.removeprefix("favors_")
    if long_label == "nondiscriminating" and conflict_label != "nondiscriminating":
        return "withheld_proxy_supports_" + conflict_label.removeprefix("favors_")
    if conflict_label == "nondiscriminating" and long_label != "nondiscriminating":
        return "withheld_proxy_supports_" + long_label.removeprefix("favors_")
    if long_label == conflict_label == "nondiscriminating":
        return "withheld_proxy_is_nondiscriminating"
    return "withheld_proxy_gives_mixed_period_evidence"


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    loop = (
        root
        / "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2"
    )
    output = loop / "round_135_withheld_proxy15_arbitration_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    contract = driver / "round135_withheld_proxy15_arbitration_v1_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")
    base = load_module(
        "round135_base",
        driver / "run_round134_netp_mjj_low_frequency_reconstruction_v2.py",
    )
    r118 = load_module(
        "round135_r118",
        driver
        / "run_climate_reconstruction_loop_v2_round118_documented_moisture_seasonal_incremental_skill_v1.py",
    )
    archive = load_module(
        "round135_archive",
        driver / "audit_positive_b2_event_seasonal_psm_v1.py",
    )
    phyda = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "proxy6": archive.archive_proxy_path(),
        "proxy_metadata": root
        / "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/"
        "proxy6_metadata.csv",
        "pages_metadata": Path(
            r"E:\2Data\2Rec_ref\a_new_two_proxy\process_past_20000_proxy\pages2k_dataset_p.mat"
        ),
        "gpcc": Path(r"D:\VM\VMshare\data\analyses\GPCC\GPCC05\gpcc_1891_2020_05.nc"),
        "cru": phyda
        / "input-recon/cruts.2004151855.v4.04/pre/cru_ts4.04.1901.2019.pre.dat.nc",
        "round134_curves": loop
        / "round_134_netp_mjj_low_frequency_reconstruction_v2/"
        "netp_mjj_reconstruction_curves.csv",
        "round134_summary": loop
        / "round_134_netp_mjj_low_frequency_reconstruction_v4_final_report/summary.json",
        "external": root
        / "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_external_low_frequency_benchmark_20260818_v2/"
        "external_benchmark_allowed_values.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path) for name, path in paths.items()}

    round134_summary = json.loads(paths["round134_summary"].read_text(encoding="utf-8"))
    if round134_summary["locked_1921_1939_accessed"]:
        raise RuntimeError("round134-boundary-failed")

    metadata = pd.read_csv(paths["proxy_metadata"])
    record = metadata.loc[metadata.proxy_index == PROXY_ID]
    if len(record) != 1:
        raise RuntimeError("proxy15-metadata-identity-failed")
    record = record.iloc[0]
    if abs(float(record.latitude) - 37.93) > 0.01 or abs(float(record.longitude) - 101.53) > 0.01:
        raise RuntimeError("proxy15-coordinate-failed")

    pages = loadmat(
        paths["pages_metadata"],
        variable_names=["p_lat", "p_lon", "pages2k_dataName"],
        squeeze_me=True,
        struct_as_record=False,
    )
    pages_lat = np.asarray(pages["p_lat"], dtype=float).reshape(-1)
    pages_lon = np.asarray(pages["p_lon"], dtype=float).reshape(-1)
    pages_name = np.asarray(pages["pages2k_dataName"]).reshape(-1)
    identity_indices = np.flatnonzero(
        np.isclose(pages_lat, float(record.latitude), atol=1e-6)
        & np.isclose(pages_lon, float(record.longitude), atol=1e-6)
    )
    identity_names = [str(pages_name[index]).strip() for index in identity_indices]
    if identity_names != [EXPECTED_PUBLIC_ID]:
        raise RuntimeError(f"proxy15-public-identity-failed:{identity_names}")

    proxy_modern = base.read_proxy_values(paths["proxy6"], (PROXY_ID,), MODERN_YEARS)[:, 0]
    proxy_paleo = base.read_proxy_values(paths["proxy6"], (PROXY_ID,), PALEO_YEARS)[:, 0]
    if not np.isfinite(proxy_modern).all():
        raise RuntimeError("proxy15-modern-support-failed")
    if np.isfinite(proxy_paleo[(PALEO_YEARS >= 1921) & (PALEO_YEARS <= 1939)]).any():
        raise RuntimeError("locked-years-read")

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

    proxy_modern_smooth = base.centered_rolling(proxy_modern, 11)
    proxy_paleo_smooth = base.centered_rolling(proxy_paleo, 11)
    modern_center = float(np.nanmean(proxy_modern_smooth))
    modern_scale = float(np.nanstd(proxy_modern_smooth, ddof=1))
    if modern_scale <= 1e-12:
        raise RuntimeError("proxy15-modern-scale-failed")

    calibration_rows = []
    full_signs = []
    matching_block_signs = 0
    target_smooth: dict[str, np.ndarray] = {}
    for product in PRODUCTS:
        target_smooth[product] = base.centered_rolling(targets[product], 11)
        full_r = correlation(proxy_modern_smooth, target_smooth[product])
        full_sign = int(np.sign(full_r))
        full_signs.append(full_sign)
        calibration_rows.append(
            {
                "product": product,
                "window": "1940-2000_full",
                "first_effective_year": 1945,
                "last_effective_year": 1995,
                "correlation": full_r,
                "sign": full_sign,
            }
        )
        for label, start, end in (("1940-1969", 1940, 1969), ("1970-2000", 1970, 2000)):
            mask = (MODERN_YEARS >= start) & (MODERN_YEARS <= end)
            proxy_block = base.smooth_inside_mask(proxy_modern, mask)
            target_block = base.smooth_inside_mask(targets[product], mask)
            block_r = correlation(proxy_block, target_block)
            block_sign = int(np.sign(block_r))
            if block_sign == full_sign and full_sign != 0:
                matching_block_signs += 1
            calibration_rows.append(
                {
                    "product": product,
                    "window": label,
                    "first_effective_year": start + 5,
                    "last_effective_year": end - 5,
                    "correlation": block_r,
                    "sign": block_sign,
                }
            )
    calibration = pd.DataFrame(calibration_rows)
    full_rows = calibration.loc[calibration.window == "1940-2000_full"]
    orientation_agreement = len(set(full_signs)) == 1 and full_signs[0] != 0
    minimum_full_abs_r = float(full_rows.correlation.abs().min())
    informative = bool(
        orientation_agreement
        and minimum_full_abs_r >= 0.20
        and matching_block_signs >= 3
    )
    orientation = full_signs[0] if orientation_agreement else 0

    proxy_modern_oriented = (
        (proxy_modern_smooth - modern_center) / modern_scale * orientation
        if orientation
        else np.full(len(MODERN_YEARS), np.nan)
    )
    proxy_paleo_oriented = (
        (proxy_paleo_smooth - modern_center) / modern_scale * orientation
        if orientation
        else np.full(len(PALEO_YEARS), np.nan)
    )
    proxy15_31 = base.centered_rolling(proxy_paleo_oriented, 31)

    round134 = pd.read_csv(paths["round134_curves"])
    if ((round134.year >= 1921) & (round134.year <= 1939)).any():
        raise RuntimeError("locked-years-in-round134-curves")
    round134 = round134.set_index("year").reindex(PALEO_YEARS)
    composite = round134["marginalized_31yr"].to_numpy(dtype=float)

    external = pd.read_csv(paths["external"])
    if ((external.year >= 1921) & (external.year <= 1939)).any():
        raise RuntimeError("locked-years-in-external")
    qt = external.loc[
        external.benchmark.eq("northwestern_qtp_mjj_precipitation")
        & external.year.between(1000, 1920),
        ["year", "value"],
    ].copy()
    qt["year"] = qt.year.astype(int)
    qt = qt.drop_duplicates("year").set_index("year").reindex(PALEO_YEARS)
    qt31 = base.centered_rolling(qt.value.to_numpy(dtype=float), 31)

    metric_rows = []
    labels: dict[str, str] = {}
    for start, end in PERIODS:
        use = (
            (PALEO_YEARS >= start)
            & (PALEO_YEARS <= end)
            & np.isfinite(proxy15_31)
            & np.isfinite(composite)
            & np.isfinite(qt31)
        )
        if int(use.sum()) < 30:
            raise RuntimeError(f"insufficient-common-period:{start}-{end}")
        proxy_qt_r = correlation(proxy15_31[use], qt31[use])
        proxy_composite_r = correlation(proxy15_31[use], composite[use])
        delta = proxy_qt_r - proxy_composite_r
        label = period_label(delta)
        key = f"{start}_{end}"
        labels[key] = label
        metric_rows.append(
            {
                "period": key,
                "first_effective_year": int(PALEO_YEARS[use][0]),
                "last_effective_year": int(PALEO_YEARS[use][-1]),
                "common_year_count": int(use.sum()),
                "proxy15_qt_correlation": proxy_qt_r,
                "proxy15_round134_correlation": proxy_composite_r,
                "qt_minus_round134_correlation_delta": delta,
                "round134_qt_correlation": correlation(composite[use], qt31[use]),
                "proxy15_trend_sign": trend_sign(PALEO_YEARS[use], proxy15_31[use]),
                "qt_trend_sign": trend_sign(PALEO_YEARS[use], qt31[use]),
                "round134_trend_sign": trend_sign(PALEO_YEARS[use], composite[use]),
                "proxy15_qt_trend_agreement": trend_sign(PALEO_YEARS[use], proxy15_31[use])
                == trend_sign(PALEO_YEARS[use], qt31[use]),
                "proxy15_round134_trend_agreement": trend_sign(PALEO_YEARS[use], proxy15_31[use])
                == trend_sign(PALEO_YEARS[use], composite[use]),
                "period_decision": label,
            }
        )
    metrics = pd.DataFrame(metric_rows)
    decision = overall_decision(informative, labels)

    curves = pd.DataFrame(
        {
            "year": PALEO_YEARS,
            "proxy15_oriented_11yr": proxy_paleo_oriented,
            "proxy15_oriented_11_then_31yr": proxy15_31,
            "round134_marginalized_31yr": composite,
            "qt_prec_31yr": qt31,
        }
    )
    calibration.to_csv(output / "proxy15_modern_orientation_metrics.csv", index=False)
    metrics.to_csv(output / "proxy15_arbitration_metrics.csv", index=False)
    curves.to_csv(output / "proxy15_round134_qt_curves.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)
    axes[0].plot(MODERN_YEARS, proxy_modern_oriented, color="#222222", linewidth=2.0, label="withheld proxy 15")
    for product, color in (("GPCC", "#0072B2"), ("CRU", "#009E73")):
        axes[0].plot(MODERN_YEARS, zscore(target_smooth[product]), color=color, linewidth=1.4, label=f"{product} NETP MJJ")
    axes[0].set(title="Modern-only orientation check", ylabel="standardized 11-year anomaly")
    axes[0].legend(ncol=3, frameon=False)

    full = (PALEO_YEARS >= 1500) & (PALEO_YEARS <= 1900)
    common = full & np.isfinite(proxy15_31) & np.isfinite(composite) & np.isfinite(qt31)
    for axis, mask, title in (
        (axes[1], common, "Independent low-frequency arbitration"),
        (axes[2], common & (PALEO_YEARS >= 1800), "Conflict-period detail"),
    ):
        axis.plot(PALEO_YEARS[mask], zscore(proxy15_31[mask]), color="#222222", linewidth=2.0, label="withheld proxy 15")
        axis.plot(PALEO_YEARS[mask], zscore(composite[mask]), color="#0072B2", linewidth=1.7, label="Round 134 composite")
        axis.plot(PALEO_YEARS[mask], zscore(qt31[mask]), color="#CC79A7", linewidth=1.7, label="QT-Prec")
        axis.axhline(0.0, color="#999999", linewidth=0.7)
        axis.set(title=title, ylabel="within-panel standardized anomaly")
        axis.legend(ncol=3, frameon=False)
    axes[2].set_xlabel("year")
    fig.savefig(output / "round135_withheld_proxy15_arbitration_curves.png", dpi=200)
    plt.close(fig)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-file-changed")
    summary = {
        "experiment_id": "round_135_withheld_proxy15_arbitration_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "proxy_index": PROXY_ID,
        "proxy_location": [float(record.latitude), float(record.longitude)],
        "public_identity": EXPECTED_PUBLIC_ID,
        "modern_orientation_agreement": orientation_agreement,
        "modern_orientation_sign": orientation,
        "minimum_full_modern_abs_correlation": minimum_full_abs_r,
        "matching_block_sign_count_of_4": matching_block_signs,
        "informative_for_arbitration": informative,
        "period_decisions": labels,
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
    period_lines = "\n".join(
        f"- {row.period}: proxy-QT r={row.proxy15_qt_correlation:.3f}, "
        f"proxy-Round134 r={row.proxy15_round134_correlation:.3f}, "
        f"delta={row.qt_minus_round134_correlation_delta:+.3f}, "
        f"decision={row.period_decision}."
        for row in metrics.itertuples(index=False)
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Round 135 Final Result\n\n"
        f"- Public identity: `{EXPECTED_PUBLIC_ID}`.\n"
        f"- Modern orientation informative: `{informative}`; minimum |r|="
        f"`{minimum_full_abs_r:.3f}`, matching block signs `{matching_block_signs}/4`.\n"
        f"{period_lines}\n"
        f"- Final decision: `{decision}`.\n\n"
        "This diagnostic did not fit to QT-Prec, read 1921-1939, open the frozen "
        "reserve, run EnKF, or run OGGM.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
