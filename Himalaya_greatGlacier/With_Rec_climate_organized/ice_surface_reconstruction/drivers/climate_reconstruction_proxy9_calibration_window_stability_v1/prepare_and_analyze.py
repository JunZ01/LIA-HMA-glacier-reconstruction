"""Prepare and analyze the proxy9 equal-length calibration-window replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_v73(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {name: np.asarray(handle[name]) for name in handle.keys() if not name.startswith("#")}


def effective_counts(proxy_path: Path, windows: list[list[int]]) -> np.ndarray:
    result = np.zeros((297, len(windows)), dtype=int)
    with h5py.File(proxy_path, "r") as archive:
        refs = np.asarray(archive["proxy_data/full_proxy"]).reshape(-1, order="F")
        for index, reference in enumerate(refs):
            ages = np.asarray(archive[reference]["age_yng"], dtype=int).reshape(-1, order="F")
            values = np.asarray(archive[reference]["data"], dtype=float).reshape(-1, order="F")
            for column, (start, end) in enumerate(windows):
                result[index, column] = int(np.sum(np.isfinite(values) & (ages >= start) & (ages <= end)))
    return result


def plot_coefficients(frame: pd.DataFrame, output: Path) -> None:
    valid = frame.model_type.eq("bivariate")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.4), constrained_layout=True)
    for axis, column, title in zip(
        axes,
        ("temp_coefficient", "prcp_coefficient"),
        ("Temperature coefficient", "Precipitation coefficient"),
    ):
        pivot = frame[valid].pivot(index="proxy9_record_1based", columns="window", values=column)
        baseline = pivot["1940-2000"]
        for window, color in (("1901-1961", "#2166ac"), ("1920-1980", "#b2182b")):
            axis.scatter(baseline, pivot[window], s=16, alpha=0.55, color=color, label=window)
        limits = np.nanpercentile(np.abs(pivot.to_numpy()), 98)
        axis.plot([-limits, limits], [-limits, limits], color="#555555", lw=0.8)
        axis.axhline(0, color="#888888", lw=0.6)
        axis.axvline(0, color="#888888", lw=0.6)
        axis.set(xlabel="1940-2000 coefficient", ylabel="Alternative-window coefficient", title=title)
        axis.grid(alpha=0.16)
    axes[0].legend(frameon=False)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_stability(frame: pd.DataFrame, output: Path) -> None:
    summary = frame.groupby("window").agg(
        valid_records=("fit_valid", "sum"),
        median_pcr=("pcr", "median"),
        median_hxb_correlation=("hxb_correlation_to_baseline", "median"),
        sign_change_count=("temperature_sign_changed_from_baseline", "sum"),
    ).reindex(["1901-1961", "1920-1980", "1940-2000"])
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), constrained_layout=True)
    axes[0].bar(summary.index, summary.valid_records, color=["#2166ac", "#b2182b", "#555555"])
    axes[0].set(title="Valid PSM fits", ylabel="Records")
    axes[1].bar(summary.index, summary.median_pcr, color=["#2166ac", "#b2182b", "#555555"])
    axes[1].set(title="Median PSM correlation", ylabel="Correlation")
    axes[2].bar(summary.index, summary.median_hxb_correlation, color=["#2166ac", "#b2182b", "#555555"])
    axes[2].set(title="Median HXb correlation to baseline", ylabel="Correlation")
    for axis in axes:
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.16)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def prepare(cfg: dict, root: Path, output: Path) -> int:
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    output.mkdir(parents=True)
    audit = pd.read_csv(root / cfg["sources"]["network_audit"])
    if len(audit) != int(cfg["gates"]["proxy_count"]):
        raise ValueError("proxy-count-changed")
    audit.to_csv(output / "selected_records.csv", index=False)
    source_paths = {
        "proxy9": Path(cfg["archive_root"]) / cfg["sources"]["proxy9"],
        "combined_proxy": Path(cfg["sources"]["combined_proxy"]),
        "network_audit": root / cfg["sources"]["network_audit"],
        "early_contributions": root / cfg["sources"]["early_contributions"],
    }
    manifest = {name: {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size} for name, path in source_paths.items()}
    (output / "input_manifest_before.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "prepared", "output": str(output), "sources": manifest}, indent=2))
    return 0


def analyze(cfg: dict, root: Path, output: Path) -> int:
    result_path = output / "calibration_windows.mat"
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    data = load_v73(result_path)
    windows = [list(map(int, item)) for item in cfg["windows"]]
    labels = [f"{start}-{end}" for start, end in windows]
    # MATLAB v7.3 stores dimensions in reverse order when read by h5py.
    hxb = np.asarray(data["hxb"], dtype=float).transpose(2, 1, 0)
    coefficients = np.asarray(data["b"], dtype=float).transpose(2, 1, 0)
    pcr = np.asarray(data["pcr"], dtype=float).T
    pcr_t = np.asarray(data["pcr_t"], dtype=float).T
    pcr_p = np.asarray(data["pcr_p"], dtype=float).T
    pval = np.asarray(data["pval"], dtype=float).T
    residual_variance = np.asarray(data["residual_variance"], dtype=float).T
    data_max_abs = np.asarray(data["data_max_abs"], dtype=float).T
    repeat = np.asarray(data["baseline_repeat_hxb"], dtype=float).T
    baseline_index = labels.index("1940-2000")
    repeat_closure = float(np.nanmax(np.abs(hxb[:, baseline_index] - repeat)))
    source_data_closure = float(np.nanmax(data_max_abs))
    gates = cfg["gates"]
    if repeat_closure > float(gates["baseline_repeat_hxb_max_abs"]):
        raise RuntimeError(f"common-rng-repeat-failed:{repeat_closure}")
    if source_data_closure > float(gates["source_data_max_abs"]):
        raise RuntimeError(f"proxy-data-closure-failed:{source_data_closure}")

    proxy_path = Path(cfg["archive_root"]) / cfg["sources"]["proxy9"]
    counts = effective_counts(proxy_path, windows)
    audit = pd.read_csv(output / "selected_records.csv")
    contribution = pd.read_csv(root / cfg["sources"]["early_contributions"])[[
        "proxy9_record_1based",
        "archived_R_hma_mean_contribution_c",
        "archived_R_hma_trend_contribution_c_per_century",
    ]]
    rows = []
    for index in range(297):
        baseline_temp = coefficients[index, baseline_index, 1]
        baseline_hxb = hxb[index, baseline_index]
        for window_index, label in enumerate(labels):
            current = hxb[index, window_index]
            valid_hxb = np.isfinite(current) & np.isfinite(baseline_hxb)
            correlation = float(np.corrcoef(current[valid_hxb], baseline_hxb[valid_hxb])[0, 1]) if valid_hxb.sum() > 2 and np.std(current[valid_hxb]) > 0 and np.std(baseline_hxb[valid_hxb]) > 0 else np.nan
            temp = coefficients[index, window_index, 1]
            fit_valid = bool(np.isfinite(current).all() and np.isfinite(temp))
            rows.append({
                "proxy9_record_1based": index + 1,
                "window": label,
                "model_type": "bivariate" if index < 290 else "univariate",
                "effective_calibration_count": int(counts[index, window_index]),
                "fit_valid": fit_valid,
                "intercept": coefficients[index, window_index, 0],
                "temp_coefficient": temp,
                "prcp_coefficient": coefficients[index, window_index, 2],
                "pcr": pcr[index, window_index],
                "pcr_temperature": pcr_t[index, window_index],
                "pcr_precipitation": pcr_p[index, window_index],
                "p_value": pval[index, window_index],
                "residual_variance": residual_variance[index, window_index],
                "hxb_correlation_to_baseline": correlation,
                "hxb_rms_difference_to_baseline": float(np.sqrt(np.nanmean((current - baseline_hxb) ** 2))),
                "temperature_sign_changed_from_baseline": bool(np.isfinite(temp) and np.isfinite(baseline_temp) and np.sign(temp) != np.sign(baseline_temp)),
            })
    frame = pd.DataFrame(rows).merge(audit, on="proxy9_record_1based", how="left", validate="many_to_one").merge(contribution, on="proxy9_record_1based", how="left", validate="many_to_one")
    frame.to_csv(output / "proxy9_calibration_window_diagnostics.csv", index=False)
    window_summary = frame.groupby("window").agg(
        valid_fit_count=("fit_valid", "sum"),
        records_below_20_effective_years=("effective_calibration_count", lambda x: int((x < 20).sum())),
        median_effective_years=("effective_calibration_count", "median"),
        median_pcr=("pcr", "median"),
        median_p_value=("p_value", "median"),
        significant_p_lt_0p05_count=("p_value", lambda x: int((x < 0.05).sum())),
        median_residual_variance=("residual_variance", "median"),
        median_hxb_correlation_to_baseline=("hxb_correlation_to_baseline", "median"),
        median_hxb_rms_difference=("hxb_rms_difference_to_baseline", "median"),
        temperature_sign_change_count=("temperature_sign_changed_from_baseline", "sum"),
    ).reindex(labels).reset_index()
    window_summary.to_csv(output / "calibration_window_summary.csv", index=False)

    alternatives = frame[~frame.window.eq("1940-2000")].copy()
    instability = alternatives.groupby("proxy9_record_1based").agg(
        minimum_hxb_correlation=("hxb_correlation_to_baseline", "min"),
        maximum_hxb_rms_difference=("hxb_rms_difference_to_baseline", "max"),
        any_temperature_sign_change=("temperature_sign_changed_from_baseline", "max"),
        minimum_effective_calibration_count=("effective_calibration_count", "min"),
    ).reset_index().merge(audit, on="proxy9_record_1based", how="left", validate="one_to_one").merge(contribution, on="proxy9_record_1based", how="left", validate="one_to_one")
    instability["priority_score"] = (
        (1.0 - instability.minimum_hxb_correlation.clip(-1, 1)).fillna(2.0)
        * instability.archived_R_hma_trend_contribution_c_per_century.abs()
    )
    instability.sort_values("priority_score", ascending=False).to_csv(output / "record_calibration_instability_ranking.csv", index=False)
    plot_coefficients(frame, output / "figure_coefficient_stability.png")
    plot_stability(frame, output / "figure_window_stability_summary.png")

    before = json.loads((output / "input_manifest_before.json").read_text(encoding="utf-8"))
    after = {name: {"path": item["path"], "sha256": sha256(Path(item["path"])), "size_bytes": Path(item["path"]).stat().st_size} for name, item in before.items()}
    pure = before == after
    summary = {
        "experiment_id": cfg["experiment_id"],
        "status": "pass" if pure else "fail",
        "baseline_repeat_hxb_max_abs": repeat_closure,
        "proxy_data_max_abs": source_data_closure,
        "source_hashes_unchanged": pure,
        "window_summary": window_summary.to_dict(orient="records"),
        "top_instability_records": instability.nlargest(15, "priority_score")[[
            "proxy9_record_1based", "combined_name", "minimum_hxb_correlation",
            "maximum_hxb_rms_difference", "any_temperature_sign_change",
            "minimum_effective_calibration_count", "archived_R_hma_trend_contribution_c_per_century",
        ]].to_dict(orient="records"),
        "interpretation_boundary": "PSM-only equal-length window diagnosis. No EnKF reconstruction or production-window selection is made here.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "input_manifest_after.json").write_text(json.dumps(after, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Proxy9 Calibration-Window Stability v1\n\n"
        f"Status: **{summary['status']}**. This is a PSM-only equal-length calibration-window diagnosis; see `summary.json` and the record-level tables.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if not pure:
        raise RuntimeError("source-purity-failed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "analyze"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment_config.json"))
    args = parser.parse_args()
    root = project_root()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    output = root / cfg["result_root"]
    return prepare(cfg, root, output) if args.mode == "prepare" else analyze(cfg, root, output)


if __name__ == "__main__":
    raise SystemExit(main())
