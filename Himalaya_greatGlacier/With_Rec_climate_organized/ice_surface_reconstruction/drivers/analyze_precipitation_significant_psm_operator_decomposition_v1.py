"""Summarize the frozen significant-bivariate-PSM operator decomposition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BRANCHES = [
    "baseline",
    "combined_rebuild",
    "temperature_only",
    "precipitation_only",
    "zero_selected_anomaly",
]
PERIOD = (1901, 1920)
TOLERANCE = 1e-12


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


def decode_matlab_string(handle: h5py.File, reference: h5py.Reference) -> str:
    codes = np.asarray(handle[reference]).reshape(-1)
    return "".join(chr(int(code)) for code in codes)


def trend_per_century(years: np.ndarray, values: np.ndarray) -> float:
    slope = np.polyfit(years.astype(float), values.astype(float), 1)[0]
    return float(slope * 100.0)


def main() -> int:
    root = project_root()
    result_dir = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_significant_psm_operator_decomposition_v1"
    output_path = result_dir / "operator_decomposition.mat"
    source_audit_path = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_proxy_common_mode_source_audit_v2/proxy_source_trend_table.csv"
    preparation_path = result_dir / "prepare_summary.json"
    if missing := [str(path) for path in (output_path, source_audit_path, preparation_path) if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))

    with h5py.File(output_path, "r") as handle:
        branch_names = [decode_matlab_string(handle, ref) for ref in handle["branch_names"][()].reshape(-1)]
        if branch_names != BRANCHES:
            raise RuntimeError(f"unexpected-branches:{branch_names}")
        years = np.asarray(handle["run_years"]).reshape(-1).astype(int)
        total_increment = np.asarray(handle["branch_total_increment"]).T
        selected_contribution = np.asarray(handle["branch_selected_contribution"]).T
        nonselected_contribution = np.asarray(handle["branch_nonselected_contribution"]).T
        contribution = np.asarray(handle["branch_contribution"]).transpose(1, 2, 0)
        selected = np.asarray(handle["selected"]).reshape(-1).astype(int)
        closure_values = {
            "combined_operator_max_abs_error": float(np.asarray(handle["combined_closure"]).squeeze()),
            "archived_baseline_max_abs_error": float(np.asarray(handle["baseline_closure"]).squeeze()),
            "innovation_invariance_max_abs_error": float(np.asarray(handle["innovation_invariance"]).squeeze()),
            "branch_increment_max_abs_closure_error": float(np.nanmax(np.abs(np.asarray(handle["branch_closure_error"])))),
            "operator_mean_invariance_max_abs_error": float(np.max(np.abs(np.asarray(handle["mean_invariance"])))),
        }

    if not np.array_equal(years, np.arange(PERIOD[0], PERIOD[1] + 1)):
        raise RuntimeError("period-contract-failed")
    if len(selected) != 65 or len(np.unique(selected)) != 65:
        raise RuntimeError("selected-proxy-contract-failed")
    if any(value > TOLERANCE for value in closure_values.values()):
        raise RuntimeError("closure-contract-failed:" + json.dumps(closure_values, sort_keys=True))

    audit = pd.read_csv(source_audit_path).set_index("proxy_index")
    selected_audit = audit.loc[selected]
    if not selected_audit["psm_significance"].eq("p<=0.05").all():
        raise RuntimeError("significance-contract-failed")
    positive_b2 = selected_audit.index[selected_audit["coefficient_b2"] >= 0].to_numpy(dtype=int)
    negative_b2 = selected_audit.index[selected_audit["coefficient_b2"] < 0].to_numpy(dtype=int)
    if len(positive_b2) + len(negative_b2) != len(selected):
        raise RuntimeError("b2-stratification-contract-failed")

    frame = pd.DataFrame({"year": years})
    for branch_idx, branch in enumerate(BRANCHES):
        frame[f"increment_{branch}"] = total_increment[:, branch_idx]
        frame[f"selected_contribution_{branch}"] = selected_contribution[:, branch_idx]
        frame[f"nonselected_contribution_{branch}"] = nonselected_contribution[:, branch_idx]
        frame[f"selected_b2_nonnegative_contribution_{branch}"] = contribution[:, positive_b2 - 1, branch_idx].sum(axis=1)
        frame[f"selected_b2_negative_contribution_{branch}"] = contribution[:, negative_b2 - 1, branch_idx].sum(axis=1)

    baseline = frame["increment_baseline"].to_numpy()
    zero = frame["increment_zero_selected_anomaly"].to_numpy()
    temperature = frame["increment_temperature_only"].to_numpy()
    precipitation = frame["increment_precipitation_only"].to_numpy()
    frame["effect_total_selected_operator_anomaly"] = baseline - zero
    frame["effect_temperature_operator_anomaly"] = temperature - zero
    frame["effect_precipitation_operator_anomaly"] = precipitation - zero
    frame["effect_nonadditivity"] = (
        frame["effect_total_selected_operator_anomaly"]
        - frame["effect_temperature_operator_anomaly"]
        - frame["effect_precipitation_operator_anomaly"]
    )
    frame["combined_rebuild_minus_baseline"] = frame["increment_combined_rebuild"] - baseline

    trend_rows: list[dict[str, object]] = []
    for column in frame.columns:
        if column == "year":
            continue
        values = frame[column].to_numpy(dtype=float)
        trend_rows.append(
            {
                "series": column,
                "mean_mm_day": float(values.mean()),
                "start_1901_mm_day": float(values[0]),
                "end_1920_mm_day": float(values[-1]),
                "change_1901_to_1920_mm_day": float(values[-1] - values[0]),
                "trend_mm_day_per_century": trend_per_century(years, values),
            }
        )
    trends = pd.DataFrame(trend_rows).sort_values("series")
    trend_lookup = trends.set_index("series")["trend_mm_day_per_century"].to_dict()
    effect_trends = {
        "total_selected_operator_anomaly": trend_lookup["effect_total_selected_operator_anomaly"],
        "temperature_operator_anomaly": trend_lookup["effect_temperature_operator_anomaly"],
        "precipitation_operator_anomaly": trend_lookup["effect_precipitation_operator_anomaly"],
        "nonadditivity": trend_lookup["effect_nonadditivity"],
    }
    dominant = max(
        ("temperature", abs(effect_trends["temperature_operator_anomaly"])),
        ("precipitation", abs(effect_trends["precipitation_operator_anomaly"])),
        key=lambda item: item[1],
    )[0]

    with preparation_path.open(encoding="utf-8") as handle:
        preparation = json.load(handle)
    summary = {
        "experiment_id": "climate_reconstruction_precipitation_significant_psm_operator_decomposition_v1",
        "analysis_driver": Path(__file__).name,
        "period": list(PERIOD),
        "locked_final_period_accessed": False,
        "selected_proxy_definition": preparation["selected_proxy_definition"],
        "selected_proxy_count": int(len(selected)),
        "b2_nonnegative_proxy_count": int(len(positive_b2)),
        "b2_negative_proxy_count": int(len(negative_b2)),
        "closure": closure_values,
        "branch_trend_mm_day_per_century": {branch: trend_lookup[f"increment_{branch}"] for branch in BRANCHES},
        "operator_effect_trend_mm_day_per_century": effect_trends,
        "dominant_single_operator_pathway_by_absolute_trend": dominant,
        "interpretation_limit": "Frozen-background EnKF mechanism sensitivity only; it does not validate, replace, or rerun any PSM or production reconstruction.",
    }

    frame.to_csv(result_dir / "operator_decomposition_annual_curves.csv", index=False)
    trends.to_csv(result_dir / "operator_decomposition_trends.csv", index=False)
    (result_dir / "operator_decomposition_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {
        "operator_decomposition_mat_sha256": sha256(output_path),
        "source_audit_csv_sha256": sha256(source_audit_path),
        "prepare_summary_json_sha256": sha256(preparation_path),
        "analysis_driver_sha256": sha256(Path(__file__)),
    }
    (result_dir / "analysis_source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11, "legend.fontsize": 8})
    figure, axes = plt.subplots(3, 1, figsize=(9.2, 9.0), sharex=True, constrained_layout=True)
    colors = {
        "baseline": "#1b1b1b",
        "combined_rebuild": "#0072B2",
        "temperature_only": "#D55E00",
        "precipitation_only": "#009E73",
        "zero_selected_anomaly": "#7A7A7A",
    }
    labels = {
        "baseline": "archived baseline",
        "combined_rebuild": "combined rebuild",
        "temperature_only": "T anomaly only",
        "precipitation_only": "P anomaly only",
        "zero_selected_anomaly": "selected anomalies removed",
    }
    for branch in BRANCHES:
        axes[0].plot(years, frame[f"increment_{branch}"], label=labels[branch], color=colors[branch], linewidth=2 if branch == "baseline" else 1.5)
    axes[0].axhline(0, color="#B5B5B5", linewidth=0.8)
    axes[0].set_ylabel("HMA increment (mm day$^{-1}$)")
    axes[0].set_title("Frozen significant-PSM branches")
    axes[0].legend(ncol=2, frameon=False, loc="best")

    axes[1].plot(years, frame["effect_total_selected_operator_anomaly"], color="#1b1b1b", linewidth=2, label="total selected anomaly effect")
    axes[1].plot(years, frame["effect_temperature_operator_anomaly"], color="#D55E00", linewidth=1.6, label="temperature pathway")
    axes[1].plot(years, frame["effect_precipitation_operator_anomaly"], color="#009E73", linewidth=1.6, label="precipitation pathway")
    axes[1].plot(years, frame["effect_nonadditivity"], color="#6A3D9A", linewidth=1.4, linestyle="--", label="EnKF nonadditivity")
    axes[1].axhline(0, color="#B5B5B5", linewidth=0.8)
    axes[1].set_ylabel("Increment effect (mm day$^{-1}$)")
    axes[1].set_title("Effect of selected-operator ensemble anomalies")
    axes[1].legend(ncol=2, frameon=False, loc="best")

    axes[2].plot(years, frame["selected_b2_nonnegative_contribution_baseline"], color="#0072B2", linewidth=1.8, label=f"selected b2 >= 0 (n={len(positive_b2)})")
    axes[2].plot(years, frame["selected_b2_negative_contribution_baseline"], color="#CC79A7", linewidth=1.8, label=f"selected b2 < 0 (n={len(negative_b2)})")
    axes[2].plot(years, frame["selected_contribution_baseline"], color="#1b1b1b", linewidth=2, label="all selected")
    axes[2].axhline(0, color="#B5B5B5", linewidth=0.8)
    axes[2].set_xlabel("Year")
    axes[2].set_ylabel("Selected contribution (mm day$^{-1}$)")
    axes[2].set_title("Archived baseline: selected PSM contribution by precipitation-coefficient sign")
    axes[2].legend(ncol=2, frameon=False, loc="best")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    figure.savefig(result_dir / "operator_decomposition_curves.png", dpi=180)
    plt.close(figure)

    result_text = [
        "# Significant PSM Operator Decomposition",
        "",
        "Period: 1901-1920 only. The locked 1921-1939 final test period was not accessed.",
        "",
        "## Validation",
        "",
        *[f"- {key}: `{value:.3e}`" for key, value in closure_values.items()],
        "",
        "## Trend Results",
        "",
        f"- Archived baseline: `{trend_lookup['increment_baseline']:.6f}` mm day^-1 century^-1.",
        f"- Selected-operator anomaly total effect: `{effect_trends['total_selected_operator_anomaly']:.6f}` mm day^-1 century^-1.",
        f"- Temperature-only pathway: `{effect_trends['temperature_operator_anomaly']:.6f}` mm day^-1 century^-1.",
        f"- Precipitation-only pathway: `{effect_trends['precipitation_operator_anomaly']:.6f}` mm day^-1 century^-1.",
        f"- Nonadditivity: `{effect_trends['nonadditivity']:.6f}` mm day^-1 century^-1.",
        f"- Larger single pathway by absolute trend: `{dominant}`.",
        "",
        "This is a frozen-background EnKF mechanism sensitivity. It does not validate or replace PSMs and is not a production reconstruction rerun.",
    ]
    (result_dir / "FINAL_RESULT.md").write_text("\n".join(result_text) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
