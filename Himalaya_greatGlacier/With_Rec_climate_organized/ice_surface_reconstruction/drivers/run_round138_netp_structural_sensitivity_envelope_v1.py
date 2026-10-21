"""Build the NETP temporal structural sensitivity envelope."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


YEARS = np.arange(1000, 1921, dtype=int)
PERIODS = ((1515, 1900), (1800, 1900))


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


def trend_per_century(years: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if int(valid.sum()) < 3:
        return float("nan")
    return float(np.polyfit(years[valid], values[valid], 1)[0] * 100.0)


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


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    loop = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2"
    )
    output = loop / "round_138_netp_structural_sensitivity_envelope_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round138_netp_structural_sensitivity_envelope_v1_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")

    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "round134_curves": loop
        / "round_134_netp_mjj_low_frequency_reconstruction_v2/"
        "netp_mjj_reconstruction_curves.csv",
        "round135_curves": loop
        / "round_135_withheld_proxy15_arbitration_v1/"
        "proxy15_round134_qt_curves.csv",
        "round136_curves": loop
        / "round_136_netp_spatial_group_leave_one_out_v1/"
        "leave_one_group_out_curves.csv",
        "round137_curves": loop
        / "round_137_netp_split_calibration_stability_v1/"
        "split_calibration_paleo_curves.csv",
        "round136_summary": loop
        / "round_136_netp_spatial_group_leave_one_out_v1/summary.json",
        "round137_summary": loop
        / "round_137_netp_split_calibration_stability_v1/summary.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path) for name, path in paths.items()}
    for key in ("round136_summary", "round137_summary"):
        upstream = json.loads(paths[key].read_text(encoding="utf-8"))
        if not upstream["passes_all_gates"] or upstream["locked_1921_1939_accessed"]:
            raise RuntimeError(f"upstream-contract-failed:{key}")

    round134 = pd.read_csv(paths["round134_curves"]).set_index("year").reindex(YEARS)
    round136 = pd.read_csv(paths["round136_curves"]).set_index("year").reindex(YEARS)
    round137 = pd.read_csv(paths["round137_curves"]).set_index("year").reindex(YEARS)
    validation = pd.read_csv(paths["round135_curves"]).set_index("year").reindex(YEARS)
    for frame_name, frame in (
        ("round134", round134),
        ("round136", round136),
        ("round137", round137),
        ("validation", validation),
    ):
        locked = frame.index.to_series().between(1921, 1939)
        if bool(locked.any()):
            raise RuntimeError(f"locked-years-in-derived-input:{frame_name}")

    members: dict[str, np.ndarray] = {
        "full_round134": round134.marginalized_31yr.to_numpy(dtype=float)
    }
    loo_columns = [column for column in round136.columns if column.startswith("omit_group_")]
    if len(loo_columns) != 15:
        raise RuntimeError("leave-one-out-member-count-failed")
    for column in loo_columns:
        members[column] = round136[column].to_numpy(dtype=float)
    members["split_early_product_mean"] = np.nanmean(
        round137[["GPCC_early", "CRU_early"]].to_numpy(dtype=float), axis=1
    )
    members["split_late_product_mean"] = np.nanmean(
        round137[["GPCC_late", "CRU_late"]].to_numpy(dtype=float), axis=1
    )
    if len(members) != 18:
        raise RuntimeError("structural-member-count-failed")
    matrix = np.column_stack(list(members.values()))

    quantiles = np.full((len(YEARS), 7), np.nan)
    member_count = np.isfinite(matrix).sum(axis=1)
    for row in np.flatnonzero(member_count > 0):
        values = matrix[row, np.isfinite(matrix[row])]
        quantiles[row] = np.quantile(values, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    envelope = pd.DataFrame(
        {
            "year": YEARS,
            "member_count": member_count,
            "minimum_mm_day": quantiles[:, 0],
            "q05_mm_day": quantiles[:, 1],
            "q25_mm_day": quantiles[:, 2],
            "median_mm_day": quantiles[:, 3],
            "q75_mm_day": quantiles[:, 4],
            "q95_mm_day": quantiles[:, 5],
            "maximum_mm_day": quantiles[:, 6],
            "full_round134_mm_day": members["full_round134"],
        }
    )

    proxy15 = validation.proxy15_oriented_11_then_31yr.to_numpy(dtype=float)
    qt31 = validation.qt_prec_31yr.to_numpy(dtype=float)
    full = members["full_round134"]
    median = quantiles[:, 3]
    metric_rows = []
    for name, values in members.items():
        family = (
            "full"
            if name == "full_round134"
            else "leave_one_group_out"
            if name.startswith("omit_group_")
            else "split_calibration"
        )
        for start, end in PERIODS:
            use = (
                (YEARS >= start)
                & (YEARS <= end)
                & np.isfinite(values)
                & np.isfinite(full)
            )
            metric_rows.append(
                {
                    "member": name,
                    "family": family,
                    "period": f"{start}_{end}",
                    "common_year_count": int(use.sum()),
                    "member_full_correlation": correlation(values[use], full[use]),
                    "trend_mm_day_per_century": trend_per_century(YEARS[use], values[use]),
                    "mean_bias_vs_full_mm_day": float(np.mean(values[use] - full[use])),
                    "rmse_vs_full_mm_day": float(
                        np.sqrt(np.mean((values[use] - full[use]) ** 2))
                    ),
                }
            )
    metrics = pd.DataFrame(metric_rows)

    validation_rows = []
    for start, end in PERIODS:
        use = (
            (YEARS >= start)
            & (YEARS <= end)
            & np.isfinite(median)
            & np.isfinite(proxy15)
            & np.isfinite(qt31)
        )
        proxy_qt_r = correlation(proxy15[use], qt31[use])
        proxy_median_r = correlation(proxy15[use], median[use])
        validation_rows.append(
            {
                "period": f"{start}_{end}",
                "common_year_count": int(use.sum()),
                "proxy15_median_correlation": proxy_median_r,
                "proxy15_qt_correlation": proxy_qt_r,
                "qt_minus_median_correlation_delta": proxy_qt_r - proxy_median_r,
                "withheld_proxy_favors_median": proxy_qt_r - proxy_median_r <= -0.10,
            }
        )
    validation_metrics = pd.DataFrame(validation_rows)

    common = (YEARS >= 1515) & (YEARS <= 1900)
    conflict = metrics.loc[metrics.period == "1800_1900"]
    conflict_validation = validation_metrics.loc[
        validation_metrics.period == "1800_1900"
    ].iloc[0]
    gates = {
        "all_18_members_finite_1515_1900": bool((member_count[common] == 18).all()),
        "all_member_full_correlations_ge_0_90": bool(
            (metrics.member_full_correlation >= 0.90).all()
        ),
        "all_1800_1900_trends_positive": bool(
            (conflict.trend_mm_day_per_century > 0).all()
        ),
        "withheld_proxy_supports_envelope_median": bool(
            conflict_validation.withheld_proxy_favors_median
        ),
    }
    passes = all(gates.values())
    decision = (
        "structurally_robust_netp_mjj_temporal_candidate_with_sensitivity_envelope"
        if passes
        else "structural_envelope_exposes_unresolved_temporal_instability"
    )

    envelope.to_csv(output / "netp_mjj_structural_sensitivity_envelope.csv", index=False)
    pd.DataFrame({"year": YEARS, **members}).to_csv(
        output / "netp_mjj_structural_members.csv", index=False
    )
    metrics.to_csv(output / "structural_member_metrics.csv", index=False)
    validation_metrics.to_csv(output / "envelope_external_validation_metrics.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), constrained_layout=True)
    full_mask = (YEARS >= 1515) & (YEARS <= 1900) & (member_count == 18)
    axes[0].fill_between(
        YEARS[full_mask], quantiles[full_mask, 1], quantiles[full_mask, 5],
        color="#B3CDE3", alpha=0.65, label="5-95% structural range"
    )
    axes[0].fill_between(
        YEARS[full_mask], quantiles[full_mask, 2], quantiles[full_mask, 4],
        color="#6497B1", alpha=0.55, label="25-75% structural range"
    )
    axes[0].plot(YEARS[full_mask], median[full_mask], color="#005B96", linewidth=2.0, label="structural median")
    axes[0].plot(YEARS[full_mask], full[full_mask], color="#222222", linewidth=1.2, label="full Round 134")
    axes[0].set(title="NETP MJJ structural sensitivity envelope", ylabel="31-year MJJ precipitation (mm/day)")
    axes[0].legend(ncol=4, frameon=False)

    for axis, start, title in (
        (axes[1], 1515, "Independent evidence comparison"),
        (axes[2], 1800, "Conflict-period detail"),
    ):
        mask = (
            (YEARS >= start)
            & (YEARS <= 1900)
            & np.isfinite(median)
            & np.isfinite(proxy15)
            & np.isfinite(qt31)
        )
        axis.plot(YEARS[mask], zscore(median[mask]), color="#005B96", linewidth=2.0, label="envelope median")
        axis.plot(YEARS[mask], zscore(proxy15[mask]), color="#222222", linewidth=1.7, label="withheld proxy 15")
        axis.plot(YEARS[mask], zscore(qt31[mask]), color="#CC79A7", linewidth=1.5, label="QT-Prec")
        axis.axhline(0.0, color="#AAAAAA", linewidth=0.7)
        axis.set(title=title, ylabel="within-curve standardized anomaly")
        axis.legend(ncol=3, frameon=False)
    axes[2].set_xlabel("year")
    fig.savefig(output / "round138_netp_structural_sensitivity_envelope.png", dpi=200)
    plt.close(fig)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-file-changed")
    conflict_slopes = conflict.trend_mm_day_per_century.to_numpy(dtype=float)
    conflict_width = quantiles[conflict.index if False else np.zeros(0, dtype=int)]
    conflict_year_mask = (YEARS >= 1800) & (YEARS <= 1900) & (member_count == 18)
    summary = {
        "experiment_id": "round_138_netp_structural_sensitivity_envelope_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "member_count": len(members),
        "member_family_counts": metrics.drop_duplicates("member").family.value_counts().to_dict(),
        "minimum_member_full_correlation": {
            period: float(frame.member_full_correlation.min())
            for period, frame in metrics.groupby("period")
        },
        "trend_1800_1900_mm_day_per_century": {
            "minimum": float(np.min(conflict_slopes)),
            "median": float(np.median(conflict_slopes)),
            "maximum": float(np.max(conflict_slopes)),
        },
        "envelope_width_1800_1900_mm_day": {
            "mean_q05_q95": float(
                np.mean(quantiles[conflict_year_mask, 5] - quantiles[conflict_year_mask, 1])
            ),
            "maximum_full_range": float(
                np.max(quantiles[conflict_year_mask, 6] - quantiles[conflict_year_mask, 0])
            ),
        },
        "gates": gates,
        "passes_all_gates": passes,
        "decision": decision,
        "uncertainty_interpretation": "structural_sensitivity_not_probability_interval",
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
        "# Round 138 Final Result\n\n"
        f"- Structural members: `{len(members)}`.\n"
        f"- Minimum member/full correlations: `{summary['minimum_member_full_correlation']}`.\n"
        f"- 1800-1900 trend range: `{summary['trend_1800_1900_mm_day_per_century']}` mm/day/century.\n"
        f"- 1800-1900 envelope width: `{summary['envelope_width_1800_1900_mm_day']}`.\n"
        f"- Gates: `{gates}`.\n"
        f"- Final decision: `{decision}`.\n\n"
        "The percentile bands are structural sensitivity summaries, not probability intervals. "
        "No 1921-1939 value or frozen-reserve value was read, and no EnKF or OGGM run occurred.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
