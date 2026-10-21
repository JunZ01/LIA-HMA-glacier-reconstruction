from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import evaluate_temperature_candidate_b2_stage_c_v1 as frozen
import temperature_candidate_b2_stage_c_core_v1 as core


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
DEFAULT_INPUT = LOOP / "temp_candidate_d2_real_v1/candidate_d2_real_output_v1.npz"
DEFAULT_OUT = LOOP / "temp_candidate_d2_evaluation_v1"
REAL_SUMMARY = LOOP / "temp_candidate_d2_real_v1/summary.json"
EVALUATION_CONTRACT = LOOP / "temp_candidate_d2_resolution_prereg_v1/REAL_EVALUATION_CONTRACT.md"
B2_EARLY_METRICS = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_evaluation_v3/early_1850_1920_station_metrics.csv"
AUTHORIZATION_TEXT = "允许执行 Candidate D2 20年真实代理 EnKF"


def lowpass(values: np.ndarray, years: np.ndarray, window: int = 31) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype=float)
    for start, end in ((1500, 1920), (1940, 2000)):
        active = (years >= start) & (years <= end)
        output[active] = pd.Series(values[active]).rolling(
            window, center=True, min_periods=max(3, window // 2)
        ).mean().to_numpy()
    return output


def long_metric(
    years: np.ndarray,
    values: np.ndarray,
    reference: np.ndarray,
    start: int,
    end: int,
) -> dict[str, float]:
    selected = (years >= start) & (years <= end)
    finite = selected & np.isfinite(values) & np.isfinite(reference)
    error = values[finite] - reference[finite]
    return {
        "period": f"{start}-{end}",
        "n": int(finite.sum()),
        "rmse_degC": float(np.sqrt(np.mean(error**2))),
        "bias_degC": float(np.mean(error)),
        "correlation_r": float(np.corrcoef(values[finite], reference[finite])[0, 1]),
    }


def plot_curves(curves: pd.DataFrame, output: Path) -> None:
    years = curves["year"].to_numpy(int)
    fig, axis = plt.subplots(figsize=(13.5, 6.3), constrained_layout=True)
    series = (
        ("balanced_background", "Balanced background", "#777777", 1.4),
        ("candidate_d2", "Candidate D2 20-year", "#C33C2E", 2.4),
        ("original_project_degC", "Archived project", "#111111", 1.2),
        ("external_ref_mean_degC", "External reference mean", "#7A4EAB", 1.7),
        ("instrument_mean_degC", "Instrumental mean", "#E08B27", 1.7),
    )
    for start, end in ((1500, 1920), (1940, 2000)):
        active = (years >= start) & (years <= end)
        for column, label, color, width in series:
            window = 11 if column == "instrument_mean_degC" else 31
            smooth = lowpass(curves[column].to_numpy(float), years, window)
            axis.plot(years[active], smooth[active], color=color, linewidth=width, label=label if start == 1500 else None)
    axis.axvspan(1921, 1939, color="#EEEEEE", zorder=-1)
    axis.axhline(0.0, color="#666666", linewidth=0.7)
    axis.set(xlabel="Year", ylabel="Temperature anomaly (degC)", title="Candidate D2 20-year real-proxy evaluation")
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main(input_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    real_summary = json.loads(REAL_SUMMARY.read_text(encoding="utf-8"))
    if real_summary.get("status") != "complete":
        raise RuntimeError("real-run-not-complete")
    authorization = real_summary.get("authorization", {})
    if authorization.get("authorization_text") != AUTHORIZATION_TEXT or not authorization.get("authorized"):
        raise PermissionError("real-run-authorization-chain-invalid")
    if core.sha256(input_path) != real_summary.get("output_sha256"):
        raise RuntimeError("real-output-hash-mismatch")
    output.mkdir(parents=True)

    with np.load(input_path, allow_pickle=False) as result:
        years = np.asarray(result["years"], dtype=np.int32)
        if np.any((years >= 1921) & (years <= 1939)):
            raise RuntimeError("sealed-year-present")
        latitude = np.asarray(result["retained_state_latitude"], dtype=float)
        longitude = np.asarray(result["retained_state_longitude"], dtype=float)
        weights = np.asarray(result["hma_weights"], dtype=float)
        fields = {
            "balanced_background": frozen.own_baseline(np.asarray(result["balanced_background"], dtype=float), years),
            "candidate_d2": frozen.own_baseline(np.asarray(result["candidate_d2_temperature_anomaly"], dtype=float), years),
        }

    frozen.BRANCHES = {"balanced_background": "balanced_background", "candidate_d2": "candidate_d2"}
    panel = frozen.load_station_panel()
    panel, mapping = frozen.add_candidate_values(panel, years, latitude, longitude, fields)
    aggregate, regional, early = frozen.metric_tables(panel)
    seasonal = frozen.seasonal_diagnostics(panel)
    evidence = pd.read_csv(frozen.EVIDENCE)
    if evidence["sealed_period_values_used"].astype(bool).any():
        raise RuntimeError("evidence-sealed-flag")

    valid_weight = weights > 0
    curve_rows = {"year": years}
    for branch, field in fields.items():
        curve_rows[branch] = np.average(field[:, valid_weight], axis=1, weights=weights[valid_weight])
    context = pd.read_csv(frozen.LONG_CURVES)
    context_columns = [
        "year", "original_project_degC", "external_ref_mean_degC",
        "external_ref_min_degC", "external_ref_max_degC",
        "instrument_mean_degC", "instrument_min_degC", "instrument_max_degC",
    ]
    curves = pd.DataFrame(curve_rows).merge(context[context_columns], on="year", how="left", validate="one_to_one")

    primary = aggregate[aggregate["observation"].eq("qcf")].set_index("branch")
    candidate = primary.loc["candidate_d2"]
    background = primary.loc["balanced_background"]
    archived = primary.loc["archived_project"]
    instrumental_gates = {
        "decadal_rmse_beats_both": bool(candidate["decadal_rmse_degC"] < min(background["decadal_rmse_degC"], archived["decadal_rmse_degC"])),
        "decadal_correlation_beats_both": bool(candidate["decadal_correlation_r"] > max(background["decadal_correlation_r"], archived["decadal_correlation_r"])),
        "annual_rmse_within_5pct": bool(candidate["annual_rmse_degC"] <= 1.05 * min(background["annual_rmse_degC"], archived["annual_rmse_degC"])),
        "annual_correlation_within_0p05": bool(candidate["annual_correlation_r"] >= max(background["annual_correlation_r"], archived["annual_correlation_r"]) - 0.05),
        "trend_error_no_larger_than_archived": bool(candidate["absolute_trend_error_degC_per_century"] <= archived["absolute_trend_error_degC_per_century"]),
    }

    smooth_ref = lowpass(curves["external_ref_mean_degC"].to_numpy(float), years)
    long_rows = []
    for branch in ("balanced_background", "candidate_d2", "original_project_degC"):
        smooth_values = lowpass(curves[branch].to_numpy(float), years)
        row = long_metric(years, smooth_values, smooth_ref, 1500, 1879)
        row["branch"] = branch
        long_rows.append(row)
    long_table = pd.DataFrame(long_rows)
    long_index = long_table.set_index("branch")
    long_gate = bool(
        long_index.loc["candidate_d2", "rmse_degC"] < long_index.loc["original_project_degC", "rmse_degC"]
        and long_index.loc["candidate_d2", "rmse_degC"] <= 1.25 * long_index.loc["balanced_background", "rmse_degC"]
    )

    early_qcf = early[early["observation"].eq("qcf")].set_index("branch")
    proxy6_early = pd.read_csv(B2_EARLY_METRICS)
    proxy6_early = proxy6_early[(proxy6_early["observation"] == "qcf") & (proxy6_early["branch"] == "candidate_proxy6")].iloc[0]
    bridge_gates = {
        "early_rmse_beats_stage_c_proxy6": bool(early_qcf.loc["candidate_d2", "rmse_degC"] < proxy6_early["rmse_degC"]),
        "early_absolute_bias_beats_stage_c_proxy6": bool(abs(early_qcf.loc["candidate_d2", "bias_degC"]) < abs(proxy6_early["bias_degC"])),
    }

    regional_qcf = regional[regional["observation"].eq("qcf")]
    regional_gates = {}
    for region in ("West", "East"):
        table = regional_qcf[regional_qcf["region_group"].eq(region)].set_index("branch")
        regional_gates[f"{region.lower()}_rmse_or_correlation_preserved"] = bool(
            table.loc["candidate_d2", "rmse_degC"] < table.loc["archived_project", "rmse_degC"]
            or table.loc["candidate_d2", "correlation_r"] >= table.loc["archived_project", "correlation_r"] - 0.05
        )

    gates = {**instrumental_gates, "long_frequency_gate": long_gate, **bridge_gates, **regional_gates}
    numeric_gate_pass = all(gates.values())

    panel.to_csv(output / "station_year_candidate_d2_values.csv", index=False)
    mapping.to_csv(output / "station_mapping.csv", index=False)
    aggregate.to_csv(output / "instrumental_aggregate_metrics.csv", index=False)
    regional.to_csv(output / "instrumental_regional_metrics.csv", index=False)
    early.to_csv(output / "early_1850_1920_metrics.csv", index=False)
    seasonal.to_csv(output / "season_mapping_sensitivity.csv", index=False)
    evidence.to_csv(output / "independent_evidence_matrix_snapshot.csv", index=False)
    curves.to_csv(output / "candidate_d2_long_curves_allowed.csv", index=False)
    long_table.to_csv(output / "long_frequency_metrics.csv", index=False)
    plot_curves(curves, output / "candidate_d2_long_curves_allowed.png")

    summary = {
        "experiment": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "numeric_gate_pass": numeric_gate_pass,
        "gates": gates,
        "qcf_candidate_d2": candidate.to_dict(),
        "long_frequency_metrics": long_table.to_dict(orient="records"),
        "missing_direct_station_regions": ["Central"],
        "production_promotion_automatic": False,
        "sealed_1921_1939_values_used": False,
        "prohibited_source_accessed": False,
        "production_modified": False,
        "oggm_run": False,
        "decision_boundary": "numeric pass is necessary but not sufficient because Candidate D2 is post hoc and Central direct-station coverage is absent",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Candidate D2 real-proxy evaluation\n\n"
        f"Numeric gate pass: `{numeric_gate_pass}`.\n\n"
        + "\n".join(f"- `{name}`: `{value}`" for name, value in gates.items())
        + "\n\nThis evaluation cannot automatically modify production.\n",
        encoding="utf-8",
    )
    sources = {
        "facts": ROOT / "LIA_PROJECT_FACTS.md",
        "real_output": input_path,
        "real_summary": REAL_SUMMARY,
        "evaluation_contract": EVALUATION_CONTRACT,
        "early_stations": frozen.EARLY_STATIONS,
        "modern_stations": frozen.MODERN_STATIONS,
        "monthly_panel": frozen.MONTHLY_PANEL,
        "monthly_climatology": frozen.MONTHLY_CLIM,
        "evidence": frozen.EVIDENCE,
        "long_curves": frozen.LONG_CURVES,
        "driver": DRIVER,
    }
    manifest = {name: {"path": str(path), "sha256": core.sha256(path)} for name, path in sources.items()}
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    arguments = parser.parse_args()
    main(arguments.input, arguments.output)
