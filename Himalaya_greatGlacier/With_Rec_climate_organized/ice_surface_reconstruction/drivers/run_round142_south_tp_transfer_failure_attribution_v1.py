"""Attribute Round 141 transfer failure to target disagreement or PSM instability."""

from __future__ import annotations

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


PERIODS = (("early", 1941, 1970), ("late", 1971, 2000))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    result: dict[str, object] = {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if stat.st_size <= 100_000_000:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
    return result


def trend(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    return float(np.polyfit(np.flatnonzero(valid), values[valid], 1)[0]) if int(valid.sum()) >= 3 else float("nan")


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_142_south_tp_transfer_failure_attribution_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round142_south_tp_transfer_failure_attribution_v1_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")

    base = load_module("round142_base", driver / "run_round134_netp_mjj_low_frequency_reconstruction_v2.py")
    r141 = load_module("round142_r141", driver / "run_round141_south_tp_crossyear_precipitation_external_validation_v1.py")
    r118 = load_module(
        "round142_r118",
        driver / "run_climate_reconstruction_loop_v2_round118_documented_moisture_seasonal_incremental_skill_v1.py",
    )
    audit = load_module("round142_archive", driver / "audit_positive_b2_event_seasonal_psm_v1.py")
    phyda = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
    r141_output = loop / "round_141_south_tp_crossyear_precipitation_external_validation_v1"
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "r141_driver": driver / "run_round141_south_tp_crossyear_precipitation_external_validation_v1.py",
        "r141_summary": r141_output / "summary.json",
        "r141_inventory": r141_output / "proxy_inventory.csv",
        "proxy6": audit.archive_proxy_path(),
        "gpcc": Path(r"D:\VM\VMshare\data\analyses\GPCC\GPCC05\gpcc_1891_2020_05.nc"),
        "cru": phyda / "input-recon/cruts.2004151855.v4.04/pre/cru_ts4.04.1901.2019.pre.dat.nc",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path) for name, path in paths.items()}
    r141_summary = json.loads(paths["r141_summary"].read_text(encoding="utf-8"))
    if r141_summary.get("decision") != "close_south_tp_transfer_modern_skill_failed":
        raise RuntimeError("round141-decision-mismatch")

    inventory = pd.read_csv(paths["r141_inventory"])
    proxy_ids = tuple(inventory.proxy_index.astype(int))
    if len(proxy_ids) != 38:
        raise RuntimeError("round141-inventory-mismatch")
    proxy = base.read_proxy_values(paths["proxy6"], proxy_ids, r141.CAL_YEARS)

    lat = np.tile(np.asarray([27, 29, 31, 33, 35], dtype=float), 6)
    lon = np.repeat(np.asarray([91, 93, 95, 97, 99, 101], dtype=float), 5)
    weights = np.cos(np.deg2rad(lat))
    targets: dict[tuple[str, str], np.ndarray] = {}
    for product, path, variable, source_start in (
        ("GPCC", paths["gpcc"], "precip", 1891),
        ("CRU", paths["cru"], "pre", 1901),
    ):
        monthly = r118.read_netcdf3_points(path, variable, source_start, lat, lon)
        for season, (previous, current) in r141.SEASONS.items():
            cells = r141.aggregate_crossyear(monthly, previous, current)
            targets[(season, product)] = np.average(cells, axis=1, weights=weights)

    target_rows = []
    proxy_rows = []
    smoothed_targets: dict[tuple[str, str, str], np.ndarray] = {}
    for season in r141.SEASONS:
        for period, start, end in PERIODS:
            mask = (r141.CAL_YEARS >= start) & (r141.CAL_YEARS <= end)
            smooth_proxy = base.smooth_inside_mask(proxy, mask)
            for product in r141.PRODUCTS:
                smoothed_targets[(season, product, period)] = base.smooth_inside_mask(targets[(season, product)], mask)
            gpcc = smoothed_targets[(season, "GPCC", period)]
            cru = smoothed_targets[(season, "CRU", period)]
            use = mask & np.isfinite(gpcc) & np.isfinite(cru)
            gpcc_trend = trend(gpcc[use])
            cru_trend = trend(cru[use])
            target_rows.append({
                "season": season,
                "period": period,
                "effective_count": int(use.sum()),
                "gpcc_cru_correlation": base.correlation(gpcc[use], cru[use]),
                "gpcc_trend": gpcc_trend,
                "cru_trend": cru_trend,
                "trend_sign_agreement": bool(np.sign(gpcc_trend) == np.sign(cru_trend)),
            })
            for product in r141.PRODUCTS:
                target = smoothed_targets[(season, product, period)]
                for column, proxy_id in enumerate(proxy_ids):
                    proxy_rows.append({
                        "proxy_index": proxy_id,
                        "season": season,
                        "product": product,
                        "period": period,
                        "correlation": base.correlation(smooth_proxy[:, column], target),
                    })

    target_metrics = pd.DataFrame(target_rows)
    proxy_long = pd.DataFrame(proxy_rows)
    proxy_wide = proxy_long.pivot(index=["proxy_index", "season", "product"], columns="period", values="correlation").reset_index()
    proxy_wide["sign_agreement"] = np.sign(proxy_wide.early) == np.sign(proxy_wide.late)
    proxy_wide["absolute_correlation_change"] = np.abs(proxy_wide.late - proxy_wide.early)

    stability_rows = []
    for (season, product), group in proxy_wide.groupby(["season", "product"], sort=False):
        finite = np.isfinite(group.early) & np.isfinite(group.late)
        stability_rows.append({
            "season": season,
            "product": product,
            "finite_proxy_count": int(finite.sum()),
            "sign_agreement_fraction": float(group.loc[finite, "sign_agreement"].mean()),
            "median_absolute_correlation_change": float(group.loc[finite, "absolute_correlation_change"].median()),
            "early_late_correlation_across_proxies": base.correlation(group.loc[finite, "early"].to_numpy(), group.loc[finite, "late"].to_numpy()),
        })
    stability = pd.DataFrame(stability_rows)
    stability["passes_temporal_stability"] = (
        (stability.sign_agreement_fraction >= 0.60)
        & (stability.median_absolute_correlation_change <= 0.40)
    )

    product_proxy_rows = []
    for season in r141.SEASONS:
        for period in ("early", "late"):
            gpcc = proxy_wide[(proxy_wide.season == season) & (proxy_wide.product == "GPCC")].set_index("proxy_index")[period]
            cru = proxy_wide[(proxy_wide.season == season) & (proxy_wide.product == "CRU")].set_index("proxy_index")[period]
            product_proxy_rows.append({
                "season": season,
                "period": period,
                "proxy_correlation_vector_gpcc_cru_correlation": base.correlation(gpcc.to_numpy(), cru.to_numpy()),
                "proxy_sign_agreement_fraction_across_products": float((np.sign(gpcc) == np.sign(cru)).mean()),
            })
    product_proxy = pd.DataFrame(product_proxy_rows)

    target_coherent = bool(
        (target_metrics.gpcc_cru_correlation >= 0.80).all()
        and target_metrics.trend_sign_agreement.all()
    )
    both_product_failure_by_season = {
        season: bool(
            (~stability[stability.season == season].passes_temporal_stability).all()
        )
        for season in r141.SEASONS
    }
    proxy_instability = bool(any(both_product_failure_by_season.values()))
    if not target_coherent:
        decision = "resolve_instrumental_target_disagreement_before_psm_redesign"
    elif proxy_instability:
        decision = "advance_to_proxy_specific_cross_validated_seasonal_psm_redesign"
    else:
        decision = "failure_not_attributed_stop_before_method_change"

    target_metrics.to_csv(output / "instrumental_target_coherence.csv", index=False)
    proxy_wide.to_csv(output / "proxy_early_late_correlations.csv", index=False)
    stability.to_csv(output / "proxy_temporal_stability_summary.csv", index=False)
    product_proxy.to_csv(output / "proxy_product_agreement.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = {"GPCC": "#0072B2", "CRU": "#D55E00"}
    for column, season in enumerate(r141.SEASONS):
        for product in r141.PRODUCTS:
            joined = np.fmax(
                smoothed_targets[(season, product, "early")],
                smoothed_targets[(season, product, "late")],
            )
            valid = np.isfinite(joined)
            z = np.full(len(joined), np.nan)
            z[valid] = (joined[valid] - np.mean(joined[valid])) / np.std(joined[valid], ddof=1)
            axes[0, column].plot(r141.CAL_YEARS, z, color=colors[product], linewidth=1.5, label=product)
        axes[0, column].axvline(1970.5, color="#777777", linestyle=":", linewidth=1)
        axes[0, column].set_title(f"{season}: GPCC and CRU targets")
        axes[0, column].set_ylabel("within-block standardized 11-year target")
        for product in r141.PRODUCTS:
            rows = proxy_wide[(proxy_wide.season == season) & (proxy_wide.product == product)]
            axes[1, column].scatter(rows.early, rows.late, s=28, alpha=0.75, color=colors[product], label=product)
        axes[1, column].axhline(0, color="#999999", linewidth=0.7)
        axes[1, column].axvline(0, color="#999999", linewidth=0.7)
        axes[1, column].plot([-1, 1], [-1, 1], color="#555555", linestyle="--", linewidth=0.8)
        axes[1, column].set(xlim=(-1, 1), ylim=(-1, 1), xlabel="proxy-target r, 1941-1970", ylabel="proxy-target r, 1971-2000")
        axes[1, column].set_title(f"{season}: individual proxy stability")
    for axis in axes.flat:
        axis.grid(color="#e2e2e2", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    fig.savefig(output / "round142_transfer_failure_attribution.png", dpi=220)
    plt.close(fig)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    summary = {
        "experiment_id": "round_142_south_tp_transfer_failure_attribution_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "target_metrics": target_metrics.to_dict(orient="records"),
        "proxy_stability": stability.to_dict(orient="records"),
        "proxy_product_agreement": product_proxy.to_dict(orient="records"),
        "instrumental_targets_coherent": target_coherent,
        "both_products_fail_proxy_stability_by_season": both_product_failure_by_season,
        "proxy_temporal_instability_diagnosed": proxy_instability,
        "decision": decision,
        "locked_1921_1939_accessed": False,
        "external_validation_reserve_opened": False,
        "enkf_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 142 Transfer-Failure Attribution\n\n"
        + "\n".join(
            f"- {row.season} {row.period}: GPCC/CRU target r=`{row.gpcc_cru_correlation:.3f}`, trend agreement=`{row.trend_sign_agreement}`."
            for row in target_metrics.itertuples()
        )
        + "\n"
        + "\n".join(
            f"- {row.season} {row.product}: proxy sign agreement=`{row.sign_agreement_fraction:.1%}`, median |delta r|=`{row.median_absolute_correlation_change:.3f}`, stable=`{row.passes_temporal_stability}`."
            for row in stability.itertuples()
        )
        + f"\n- Decision: `{decision}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
