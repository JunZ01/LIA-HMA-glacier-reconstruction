"""Summarize GPCC-versus-CRU blocked proxy9 PSM stability."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path, include_hash: bool = True) -> dict[str, object]:
    stat = path.stat()
    result: dict[str, object] = {
        "path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)
    }
    if include_hash:
        value = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                value.update(block)
        result["sha256"] = value.hexdigest()
    return result


def median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanmedian(values))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    project = root()
    output = project / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2/round_05_cross_product_psm_stability_v1"
    )
    paths = {
        "matlab_output": output / "cross_product_psm_v1.mat",
        "proxy9": project / (
            "validation_data/results/paleo_surface_validation_v0/"
            "climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection/"
            "proxy9_allowed_1940_2000_official_baseline_v2.mat"
        ),
        "gpcc": Path(r"D:\VM\VMshare\data\analyses\GPCC\GPCC05\gpcc_1891_2020_05.nc"),
        "cru": Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master\input-recon\cru_ts4.04.1901.2019.pre.dat.nc"),
        "loader": Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master\load_obs_S.m"),
        "mon2ann": Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master\mon2ann.m"),
        "contract": output / "ROUND_CONTRACT.md",
        "facts": project / "LIA_PROJECT_FACTS.md",
        "matlab_driver": Path(__file__).with_name("fit_climate_reconstruction_loop_v2_round05_cross_product_psm_v1.m"),
        "analysis_driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    large = {"gpcc", "cru"}
    before = {name: state(path, include_hash=name not in large) for name, path in paths.items()}

    with h5py.File(paths["matlab_output"], "r") as handle:
        years = np.asarray(handle["years"], dtype=int).ravel()
        model_type = np.asarray(handle["model_type"], dtype=int).ravel()
        precipitation = np.asarray(handle["precipitation"], dtype=float).transpose(2, 1, 0)
        coefficients = np.asarray(handle["coefficients"], dtype=float).transpose(3, 2, 1, 0)
        standardized_precipitation = np.asarray(
            handle["standardized_precipitation_effect"], dtype=float
        ).transpose(2, 1, 0)
        holdout = np.asarray(handle["holdout_rmse"], dtype=float).transpose(3, 2, 1, 0)
    if not np.array_equal(years, np.arange(1940, 2001)):
        raise RuntimeError("year-contract-failed")
    if np.any((years >= 1921) & (years <= 1939)):
        raise RuntimeError("locked-period-accessed")
    sensitive = model_type == 2
    if sensitive.sum() != 290:
        raise RuntimeError("proxy-contract-failed")

    b2_gpcc = coefficients[sensitive, 1:, 2, 0]
    b2_cru = coefficients[sensitive, 1:, 2, 1]
    paired = np.isfinite(b2_gpcc) & np.isfinite(b2_cru) & (b2_gpcc != 0) & (b2_cru != 0)
    sign_same = np.sign(b2_gpcc) == np.sign(b2_cru)
    paired_sign_agreement = float(np.mean(sign_same[paired]))

    within_gpcc = holdout[sensitive, :, 0, 0]
    gpcc_fit_cru_eval = holdout[sensitive, :, 0, 1]
    within_cru = holdout[sensitive, :, 1, 1]
    cru_fit_gpcc_eval = holdout[sensitive, :, 1, 0]
    penalty_gpcc_to_cru = gpcc_fit_cru_eval / within_gpcc - 1.0
    penalty_cru_to_gpcc = cru_fit_gpcc_eval / within_cru - 1.0

    effect_gpcc = standardized_precipitation[sensitive, 1:, 0]
    effect_cru = standardized_precipitation[sensitive, 1:, 1]
    effect_ratio = np.maximum(np.abs(effect_gpcc), np.abs(effect_cru)) / np.minimum(
        np.abs(effect_gpcc), np.abs(effect_cru)
    )

    record_rows: list[dict[str, object]] = []
    robust = np.zeros(sensitive.sum(), dtype=bool)
    sensitive_indices = np.flatnonzero(sensitive)
    for local, original in enumerate(sensitive_indices):
        valid_sign = paired[local]
        folds = int(np.sum(valid_sign))
        same_count = int(np.sum(sign_same[local] & valid_sign))
        sign_fraction = float(same_count / folds) if folds else float("nan")
        penalty_a = median(penalty_gpcc_to_cru[local])
        penalty_b = median(penalty_cru_to_gpcc[local])
        robust[local] = folds >= 4 and same_count >= 4 and penalty_a <= 0.10 and penalty_b <= 0.10
        record_rows.append({
            "record_index_1based": int(original + 1),
            "paired_fold_count": folds,
            "same_b2_sign_fold_count": same_count,
            "b2_sign_agreement_fraction": sign_fraction,
            "median_gpcc_fit_cru_eval_penalty": penalty_a,
            "median_cru_fit_gpcc_eval_penalty": penalty_b,
            "median_standardized_b2_effect_ratio": median(effect_ratio[local]),
            "robust_record": bool(robust[local]),
        })
    write_csv(output / "record_stability.csv", record_rows)

    fold_rows: list[dict[str, object]] = []
    blocks = [(1940, 1951), (1952, 1963), (1964, 1975), (1976, 1987), (1988, 2000)]
    for fold, (start, end) in enumerate(blocks):
        fold_rows.append({
            "fold": fold + 1, "start_year": start, "end_year": end,
            "paired_b2_count": int(np.sum(paired[:, fold])),
            "b2_sign_agreement": float(np.mean(sign_same[:, fold][paired[:, fold]])),
            "gpcc_within_median_rmse": median(within_gpcc[:, fold]),
            "cru_within_median_rmse": median(within_cru[:, fold]),
            "gpcc_fit_cru_eval_median_penalty": median(penalty_gpcc_to_cru[:, fold]),
            "cru_fit_gpcc_eval_median_penalty": median(penalty_cru_to_gpcc[:, fold]),
        })
    write_csv(output / "fold_stability.csv", fold_rows)

    target_gpcc = precipitation[:, sensitive, 0]
    target_cru = precipitation[:, sensitive, 1]
    target_rows: list[dict[str, object]] = []
    target_correlations = []
    target_rmse = []
    for local, original in enumerate(sensitive_indices):
        left = target_gpcc[:, local]
        right = target_cru[:, local]
        valid = np.isfinite(left) & np.isfinite(right)
        correlation = float(np.corrcoef(left[valid], right[valid])[0, 1])
        rmse = float(np.sqrt(np.mean((left[valid] - right[valid]) ** 2)))
        target_correlations.append(correlation)
        target_rmse.append(rmse)
        target_rows.append({
            "record_index_1based": int(original + 1),
            "valid_year_count": int(valid.sum()),
            "gpcc_cru_anomaly_correlation": correlation,
            "gpcc_cru_anomaly_rmse_mm_day": rmse,
        })
    write_csv(output / "precipitation_target_agreement.csv", target_rows)

    median_penalty_a = median(penalty_gpcc_to_cru)
    median_penalty_b = median(penalty_cru_to_gpcc)
    robust_fraction = float(np.mean(robust))
    gates = {
        "b2_sign_agreement_at_least_70pct": bool(paired_sign_agreement >= 0.70),
        "gpcc_to_cru_penalty_no_more_than_10pct": bool(median_penalty_a <= 0.10),
        "cru_to_gpcc_penalty_no_more_than_10pct": bool(median_penalty_b <= 0.10),
        "robust_record_fraction_at_least_50pct": bool(robust_fraction >= 0.50),
    }
    passes = bool(all(gates.values()))
    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "precipitation_sensitive_records": int(sensitive.sum()),
        "paired_proxy_fold_count": int(np.sum(paired)),
        "paired_b2_sign_agreement": paired_sign_agreement,
        "median_holdout_rmse": {
            "gpcc_fit_gpcc_eval": median(within_gpcc),
            "cru_fit_cru_eval": median(within_cru),
            "gpcc_fit_cru_eval": median(gpcc_fit_cru_eval),
            "cru_fit_gpcc_eval": median(cru_fit_gpcc_eval),
        },
        "median_cross_product_penalty": {
            "gpcc_fit_to_cru_eval": median_penalty_a,
            "cru_fit_to_gpcc_eval": median_penalty_b,
        },
        "standardized_b2_effect_ratio": {
            "median": median(effect_ratio),
            "p90": float(np.nanpercentile(effect_ratio, 90)),
        },
        "target_agreement": {
            "gpcc_cru_anomaly_correlation_median": median(np.asarray(target_correlations)),
            "gpcc_cru_anomaly_correlation_p10": float(np.nanpercentile(target_correlations, 10)),
            "gpcc_cru_anomaly_rmse_mm_day_median": median(np.asarray(target_rmse)),
        },
        "robust_record_count": int(np.sum(robust)),
        "robust_record_fraction": robust_fraction,
        "gates": gates,
        "passes_cross_product_stability": passes,
        "decision": "permit_separate_operator_test" if passes else "close_direct_operator_promotion_and_revise_psm_target_definition",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    axes[0].hist(np.asarray(target_correlations), bins=25, color="#005f73", edgecolor="white")
    axes[0].set(title="GPCC-CRU target agreement", xlabel="1940-2000 correlation", ylabel="Proxy records")
    axes[1].hist(np.concatenate([penalty_gpcc_to_cru.ravel(), penalty_cru_to_gpcc.ravel()]), bins=35, color="#ee9b00", edgecolor="white")
    axes[1].axvline(0.10, color="#9b2226", linewidth=1.5, linestyle="--")
    axes[1].set(title="Cross-product prediction penalty", xlabel="Relative RMSE change", ylabel="Proxy-folds")
    axes[2].scatter(
        effect_gpcc.ravel(), effect_cru.ravel(), s=12, alpha=0.5, color="#0a9396"
    )
    finite_effect = np.isfinite(effect_gpcc) & np.isfinite(effect_cru)
    limit = float(np.nanpercentile(np.abs(np.concatenate([effect_gpcc[finite_effect], effect_cru[finite_effect]])), 98))
    axes[2].plot([-limit, limit], [-limit, limit], color="#444444", linewidth=1)
    axes[2].set(xlim=(-limit, limit), ylim=(-limit, limit), title="Standardized precipitation effect", xlabel="GPCC fit", ylabel="CRU fit")
    fig.suptitle("Proxy9 cross-product PSM stability", fontsize=14)
    fig.savefig(output / "cross_product_psm_stability.png", dpi=180)
    plt.close(fig)

    report = (
        "# Cross-Product PSM Stability\n\n"
        f"- Paired GPCC/CRU `b2` sign agreement: `{paired_sign_agreement:.3%}`.\n"
        f"- Median cross-product RMSE penalty, GPCC fit evaluated with CRU: `{median_penalty_a:.3%}`.\n"
        f"- Median cross-product RMSE penalty, CRU fit evaluated with GPCC: `{median_penalty_b:.3%}`.\n"
        f"- Robust records: `{np.sum(robust)}/{len(robust)}` (`{robust_fraction:.3%}`).\n"
        f"- Median GPCC-CRU precipitation-anomaly correlation: `{np.median(target_correlations):.6f}`.\n"
        f"- All predeclared gates pass: `{str(passes).lower()}`.\n\n"
        f"Decision: `{summary['decision']}`.\n\n"
        "No EnKF or production reconstruction was run, and 1921-1939 was not accessed.\n"
    )
    (output / "FINAL_RESULT.md").write_text(report, encoding="utf-8")

    after = {name: state(path, include_hash=name not in large) for name, path in paths.items()}
    manifest = {"sources_before": before, "sources_after": after, "source_states_unchanged": before == after}
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
