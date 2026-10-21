from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
EVALUATION = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_evaluation_v3"
REAL_RUN = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_real_proxy_v3"
OUTPUT = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_adjudication_v1"

CURVES = EVALUATION / "candidate_b2_long_term_hma_curves_allowed.csv"
AGGREGATE = EVALUATION / "instrumental_aggregate_metrics.csv"
REGIONAL = EVALUATION / "instrumental_west_central_east_metrics.csv"
EVAL_SUMMARY = EVALUATION / "summary.json"
REAL_SUMMARY = REAL_RUN / "summary.json"

BRANCHES = [
    "balanced_background",
    "candidate_proxy6",
    "candidate_proxy9",
    "candidate_joint",
    "original_project_degC",
]
PERIODS = [(1500, 1849), (1500, 1879), (1850, 1920)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def smooth_segmented(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for start, end in ((1500, 1920), (1940, 2000)):
        mask = frame["year"].between(start, end)
        result.loc[mask] = frame.loc[mask, column].rolling(
            window=window, center=True, min_periods=max(3, window // 2)
        ).mean()
    return result


def slope_per_century(year: np.ndarray, values: np.ndarray) -> float:
    finite = np.isfinite(year) & np.isfinite(values)
    if finite.sum() < 3:
        return float("nan")
    return float(np.polyfit(year[finite], values[finite], 1)[0] * 100.0)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    OUTPUT.mkdir(parents=True)

    curves = pd.read_csv(CURVES)
    aggregate = pd.read_csv(AGGREGATE)
    regional = pd.read_csv(REGIONAL)
    evaluation = json.loads(EVAL_SUMMARY.read_text(encoding="utf-8"))
    real_run = json.loads(REAL_SUMMARY.read_text(encoding="utf-8"))

    locked = curves["year"].between(1921, 1939)
    if locked.any():
        raise RuntimeError("Locked years must not be present in the curve table")

    for branch in BRANCHES:
        curves[f"{branch}_smooth31"] = smooth_segmented(curves, branch, 31)
    curves["external_ref_mean_smooth31_recomputed"] = smooth_segmented(
        curves, "external_ref_mean_degC", 31
    )

    rows = []
    for start, end in PERIODS:
        period = curves[curves["year"].between(start, end)]
        reference = period["external_ref_mean_smooth31_recomputed"].to_numpy(float)
        years = period["year"].to_numpy(float)
        for branch in BRANCHES:
            values = period[f"{branch}_smooth31"].to_numpy(float)
            finite = np.isfinite(reference) & np.isfinite(values)
            error = values[finite] - reference[finite]
            rows.append(
                {
                    "period": f"{start}-{end}",
                    "branch": branch,
                    "n": int(finite.sum()),
                    "rmse_vs_external_ref_degC": float(np.sqrt(np.mean(error**2))),
                    "mean_bias_vs_external_ref_degC": float(np.mean(error)),
                    "correlation_vs_external_ref_r": float(np.corrcoef(values[finite], reference[finite])[0, 1]),
                    "trend_degC_per_century": slope_per_century(years[finite], values[finite]),
                    "reference_trend_degC_per_century": slope_per_century(years[finite], reference[finite]),
                    "mean_increment_from_balanced_background_degC": float(
                        np.nanmean(
                            period[f"{branch}_smooth31"].to_numpy(float)
                            - period["balanced_background_smooth31"].to_numpy(float)
                        )
                    ),
                }
            )
    long_term = pd.DataFrame(rows)
    long_term.to_csv(OUTPUT / "long_term_lowpass_metrics.csv", index=False)

    primary = aggregate[aggregate["observation"].eq("qcf")].set_index("branch")
    joint = primary.loc["candidate_joint"]
    background = primary.loc["balanced_background"]
    archived = primary.loc["archived_project"]
    early_joint = long_term[
        long_term["period"].eq("1500-1879")
        & long_term["branch"].eq("candidate_joint")
    ].iloc[0]
    early_background = long_term[
        long_term["period"].eq("1500-1879")
        & long_term["branch"].eq("balanced_background")
    ].iloc[0]
    early_archived = long_term[
        long_term["period"].eq("1500-1879")
        & long_term["branch"].eq("original_project_degC")
    ].iloc[0]

    instrumental_gate = bool(evaluation["instrumental_gate_pass"])
    low_frequency_retained = bool(
        early_joint["rmse_vs_external_ref_degC"]
        <= early_background["rmse_vs_external_ref_degC"]
    )
    improves_archived_low_frequency = bool(
        early_joint["rmse_vs_external_ref_degC"]
        < early_archived["rmse_vs_external_ref_degC"]
    )
    direct_station_coverage_complete = bool(
        evaluation.get("direct_station_spatial_coverage_complete", False)
    )
    promotion_supported = bool(
        instrumental_gate and low_frequency_retained and direct_station_coverage_complete
    )

    summary = {
        "experiment": "temperature_balanced_pooled_candidate_b2_stage_c_adjudication_v1",
        "status": "complete",
        "real_stage_c_output_sha256": real_run["output_sha256"],
        "frozen_instrumental_gate_pass": instrumental_gate,
        "long_frequency_correction_retained_vs_balanced_background": low_frequency_retained,
        "joint_improves_archived_1500_1879_lowpass_rmse": improves_archived_low_frequency,
        "direct_station_spatial_coverage_complete": direct_station_coverage_complete,
        "missing_direct_station_regions": evaluation.get("missing_direct_station_regions", []),
        "promotion_supported": promotion_supported,
        "decision": "reject_candidate_b2_for_production_promotion" if not promotion_supported else "candidate_b2_supported_for_research_promotion_review",
        "mechanistic_finding": "real proxy assimilation restores modern temporal covariance but reintroduces much of the long-term warm displacement removed by the balanced prior",
        "qcf_joint": {
            "annual_rmse_degC": float(joint["annual_rmse_degC"]),
            "annual_correlation_r": float(joint["annual_correlation_r"]),
            "decadal_rmse_degC": float(joint["decadal_rmse_degC"]),
            "decadal_correlation_r": float(joint["decadal_correlation_r"]),
            "trend_degC_per_century": float(joint["predicted_trend_degC_per_century"]),
            "trend_error_degC_per_century": float(joint["absolute_trend_error_degC_per_century"]),
        },
        "qcf_comparators": {
            "balanced_background_decadal_rmse_degC": float(background["decadal_rmse_degC"]),
            "archived_project_decadal_rmse_degC": float(archived["decadal_rmse_degC"]),
        },
        "lowpass_1500_1879": {
            "candidate_joint_rmse_degC": float(early_joint["rmse_vs_external_ref_degC"]),
            "balanced_background_rmse_degC": float(early_background["rmse_vs_external_ref_degC"]),
            "archived_project_rmse_degC": float(early_archived["rmse_vs_external_ref_degC"]),
            "candidate_joint_mean_increment_from_background_degC": float(early_joint["mean_increment_from_balanced_background_degC"]),
        },
        "sealed_1921_1939_values_used": False,
        "prohibited_source_accessed": False,
        "production_modified": False,
        "oggm_run": False,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    fig, axis = plt.subplots(figsize=(13.5, 6.4), constrained_layout=True)
    colors = {
        "balanced_background": "#777777",
        "candidate_proxy6": "#2878B5",
        "candidate_proxy9": "#2E8B57",
        "candidate_joint": "#C33C2E",
        "original_project_degC": "#111111",
    }
    labels = {
        "balanced_background": "Balanced background",
        "candidate_proxy6": "Proxy6 posterior",
        "candidate_proxy9": "Proxy9 posterior",
        "candidate_joint": "Candidate B2 joint",
        "original_project_degC": "Archived project",
    }
    for start, end in ((1500, 1920), (1940, 2000)):
        segment = curves[curves["year"].between(start, end)]
        for branch in BRANCHES:
            axis.plot(
                segment["year"], segment[f"{branch}_smooth31"],
                color=colors[branch], linewidth=2.3 if branch == "candidate_joint" else 1.15,
                alpha=1.0 if branch == "candidate_joint" else 0.78,
                label=labels[branch] if start == 1500 else None,
            )
        axis.plot(
            segment["year"], segment["external_ref_mean_smooth31_recomputed"],
            color="#7A4EAB", linewidth=1.7,
            label="External reconstruction mean" if start == 1500 else None,
        )
        axis.plot(
            segment["year"], segment["instrument_mean_degC_smooth11"],
            color="#E08B27", linewidth=1.8,
            label="Instrumental product mean" if start == 1500 else None,
        )
    axis.axhline(0.0, color="#666666", linewidth=0.7)
    axis.axvspan(1921, 1939, color="#EEEEEE", zorder=-1)
    axis.set(
        xlabel="Year", ylabel="Temperature anomaly (degC)",
        title="Candidate B2 Stage C: 31-year low-pass comparison",
    )
    axis.legend(ncol=3, fontsize=8, frameon=False)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    fig.savefig(OUTPUT / "candidate_b2_stage_c_lowpass_adjudication.png", dpi=220)
    plt.close(fig)

    report = f"""# Candidate B2 Stage C final adjudication

Decision: `{summary['decision']}`.

- Frozen instrumental gate: `{instrumental_gate}`. The joint decadal RMSE is {joint['decadal_rmse_degC']:.3f} degC, versus {background['decadal_rmse_degC']:.3f} for the balanced background and {archived['decadal_rmse_degC']:.3f} for the archived project.
- Modern temporal skill improves substantially: joint annual r={joint['annual_correlation_r']:.3f}, decadal r={joint['decadal_correlation_r']:.3f}, and trend={joint['predicted_trend_degC_per_century']:.3f} degC/century.
- The 1500-1879 31-year low-pass RMSE is {early_joint['rmse_vs_external_ref_degC']:.3f} degC for the joint posterior, {early_background['rmse_vs_external_ref_degC']:.3f} for the balanced background, and {early_archived['rmse_vs_external_ref_degC']:.3f} for the archived project.
- Real proxy assimilation shifts the 1500-1879 joint posterior by a mean {early_joint['mean_increment_from_balanced_background_degC']:+.3f} degC relative to the balanced background. It therefore restores modern covariance skill while undoing much of the desired low-frequency correction.
- Direct-station regional support is incomplete because Central HMA has no admitted qcf station in the 1940-2000 panel.

Candidate B2 is scientifically informative but does not satisfy the frozen promotion contract. It must not replace the production reconstruction. The result points to the proxy-observation centering/low-frequency operator connection, rather than background selection alone, as the next mechanism to test.
"""
    (OUTPUT / "FINAL_RESULT.md").write_text(report, encoding="utf-8")

    manifest = {
        "experiment": summary["experiment"],
        "inputs": {str(path): sha256(path) for path in [CURVES, AGGREGATE, REGIONAL, EVAL_SUMMARY, REAL_SUMMARY]},
        "outputs": {
            str(path): sha256(path)
            for path in sorted(OUTPUT.iterdir())
            if path.name != "source_manifest.json"
        },
    }
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
