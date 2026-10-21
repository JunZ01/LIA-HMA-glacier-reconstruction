from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import evaluate_temperature_candidate_b2_stage_c_v1 as frozen
import evaluate_temperature_candidate_d2_real_proxy_v1 as d2_eval
import run_temperature_candidate_b2_stage_c_real_proxy_v1 as b2_reader
import run_temperature_candidate_d2_real_proxy_v1 as d2_driver
import temperature_candidate_b2_stage_c_core_v1 as core
from preflight_temperature_candidate_c_intercept_synthetic_v1 import estimate_intercepts
from preflight_temperature_candidate_d2_resolution_v1 import make_blocks
from preflight_temperature_candidate_d_decadal_v1 import block_observations


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
STATIC = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_static_package_v2/candidate_b2_stage_c_static_input_v2.npz"
REAL = LOOP / "temp_candidate_d2_real_v1/candidate_d2_real_output_v1.npz"
OUTPUT = LOOP / "temp_candidate_d2_sign_channel_evaluation_v1"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)

    with np.load(STATIC, allow_pickle=False) as package, np.load(REAL, allow_pickle=False) as real:
        years = np.asarray(package["years"], dtype=np.int32)
        if np.any((years >= 1921) & (years <= 1939)):
            raise RuntimeError("sealed-year-present")
        background = np.asarray(package["balanced_state_forced_mean"], dtype=float)
        state_residual = np.asarray(package["pooled_state_residual_library"], dtype=float)
        operator_mean = np.asarray(package["proxy6_balanced_operator_forced_mean"], dtype=float)
        operator_residual = np.asarray(package["proxy6_pooled_operator_residual_library"], dtype=float)
        variance = np.asarray(package["proxy6_residual_variance"], dtype=float)
        b1 = np.asarray(package["proxy6_psm_b0_b1_b2"], dtype=float)[:, 1]
        latitude = np.asarray(real["retained_state_latitude"], dtype=float)
        longitude = np.asarray(real["retained_state_longitude"], dtype=float)
        weights = np.asarray(real["hma_weights"], dtype=float)
        stored_all = np.asarray(real["annual_interpolated_increment"], dtype=float)
        observations, read_audit, proxy_source = b2_reader.read_allowed_proxy_observations(6, package, years)
        corrected, _, _ = estimate_intercepts(years, observations, operator_mean, minimum_count=30)

    blocks = make_blocks(years, 20)
    block_obs, block_operator, _ = block_observations(corrected, operator_mean, blocks)
    negative_blocks = np.zeros((len(blocks), background.shape[1]), dtype=float)
    nonnegative_blocks = np.zeros_like(negative_blocks)
    for block_index, row in enumerate(block_obs):
        active = np.isfinite(row)
        gain = core.kalman_gain(state_residual, operator_residual[:, active], variance[active])
        innovation = row[active] - block_operator[block_index, active]
        active_b1 = b1[active]
        negative_blocks[block_index] = gain[:, active_b1 < 0.0] @ innovation[active_b1 < 0.0]
        nonnegative_blocks[block_index] = gain[:, active_b1 >= 0.0] @ innovation[active_b1 >= 0.0]

    negative_annual = d2_driver.interpolate_field(years, blocks, negative_blocks)
    nonnegative_annual = d2_driver.interpolate_field(years, blocks, nonnegative_blocks)
    closure = float(np.max(np.abs(negative_annual + nonnegative_annual - stored_all)))
    fields = {
        "balanced_background": frozen.own_baseline(background, years),
        "d2_all": frozen.own_baseline(background + stored_all, years),
        "d2_b1_negative": frozen.own_baseline(background + negative_annual, years),
        "d2_b1_nonnegative": frozen.own_baseline(background + nonnegative_annual, years),
    }
    frozen.BRANCHES = {name: name for name in fields}
    panel = frozen.load_station_panel()
    panel, mapping = frozen.add_candidate_values(panel, years, latitude, longitude, fields)
    aggregate, regional, early = frozen.metric_tables(panel)

    valid_weight = weights > 0
    curves = pd.DataFrame({
        "year": years,
        **{
            name: np.average(field[:, valid_weight], axis=1, weights=weights[valid_weight])
            for name, field in fields.items()
        },
    })
    context = pd.read_csv(frozen.LONG_CURVES)[[
        "year", "original_project_degC", "external_ref_mean_degC", "instrument_mean_degC"
    ]]
    curves = curves.merge(context, on="year", how="left", validate="one_to_one")
    smooth_ref = d2_eval.lowpass(curves["external_ref_mean_degC"].to_numpy(float), years)
    long_rows = []
    for branch in [*fields, "original_project_degC"]:
        row = d2_eval.long_metric(
            years,
            d2_eval.lowpass(curves[branch].to_numpy(float), years),
            smooth_ref,
            1500,
            1879,
        )
        row["branch"] = branch
        long_rows.append(row)
    long_table = pd.DataFrame(long_rows)

    curve_styles = (
        ("balanced_background", "Balanced background", "#777777", 1.5),
        ("d2_all", "D2 all proxy6", "#C33C2E", 2.2),
        ("d2_b1_negative", "D2 b1 < 0 channel", "#2C7A5A", 1.6),
        ("d2_b1_nonnegative", "D2 b1 >= 0 channel", "#2878B5", 2.0),
        ("original_project_degC", "Archived project", "#111111", 1.2),
        ("external_ref_mean_degC", "External reference mean", "#7A4EAB", 1.5),
        ("instrument_mean_degC", "Instrumental mean", "#E08B27", 1.5),
    )
    fig, axis = plt.subplots(figsize=(13.5, 6.4), constrained_layout=True)
    for start, end in ((1500, 1920), (1940, 2000)):
        active = (years >= start) & (years <= end)
        for column, label, color, width in curve_styles:
            window = 11 if column == "instrument_mean_degC" else 31
            smooth = d2_eval.lowpass(curves[column].to_numpy(float), years, window)
            axis.plot(years[active], smooth[active], color=color, linewidth=width, label=label if start == 1500 else None)
    axis.axvspan(1921, 1939, color="#EEEEEE", zorder=-1)
    axis.axhline(0.0, color="#666666", linewidth=0.7)
    axis.set(xlabel="Year", ylabel="Temperature anomaly (degC)", title="Candidate D2 frozen-gain sign-channel evaluation")
    axis.legend(frameon=False, ncol=3, fontsize=8)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    fig.savefig(OUTPUT / "candidate_d2_sign_channel_curves_allowed.png", dpi=220)
    plt.close(fig)

    panel.to_csv(OUTPUT / "station_year_sign_channel_values.csv", index=False)
    mapping.to_csv(OUTPUT / "station_mapping.csv", index=False)
    aggregate.to_csv(OUTPUT / "instrumental_aggregate_metrics.csv", index=False)
    regional.to_csv(OUTPUT / "instrumental_regional_metrics.csv", index=False)
    early.to_csv(OUTPUT / "early_1850_1920_metrics.csv", index=False)
    long_table.to_csv(OUTPUT / "long_frequency_metrics.csv", index=False)
    curves.to_csv(OUTPUT / "sign_channel_long_curves_allowed.csv", index=False)
    np.savez_compressed(
        OUTPUT / "sign_channel_fields_v1.npz",
        years=years,
        negative_b1_annual_increment=negative_annual.astype(np.float32),
        nonnegative_b1_annual_increment=nonnegative_annual.astype(np.float32),
    )

    qcf = aggregate[aggregate["observation"].eq("qcf")].set_index("branch")
    early_qcf = early[early["observation"].eq("qcf")].set_index("branch")
    modern_mean = (years >= 1961) & (years <= 1990)
    early_period = (years >= 1500) & (years <= 1849)
    hma_negative = negative_annual[:, valid_weight] @ (weights[valid_weight] / weights[valid_weight].sum())
    hma_nonnegative = nonnegative_annual[:, valid_weight] @ (weights[valid_weight] / weights[valid_weight].sum())
    summary = {
        "experiment": OUTPUT.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "analysis_role": "post-hoc exact frozen-gain channel decomposition; not a new EnKF",
        "proxy6_b1_negative_count": int((b1 < 0.0).sum()),
        "proxy6_b1_nonnegative_count": int((b1 >= 0.0).sum()),
        "annual_increment_channel_closure_max_degC": closure,
        "modern_minus_early_hma_increment_contrast_degC": {
            "b1_negative": float(hma_negative[modern_mean].mean() - hma_negative[early_period].mean()),
            "b1_nonnegative": float(hma_nonnegative[modern_mean].mean() - hma_nonnegative[early_period].mean()),
        },
        "qcf_1940_2000": qcf.reset_index().to_dict(orient="records"),
        "qcf_1850_1920": early_qcf.reset_index().to_dict(orient="records"),
        "long_frequency_1500_1879": long_table.to_dict(orient="records"),
        "proxy_read_audit": read_audit,
        "sealed_1921_1939_values_used": False,
        "prohibited_source_accessed": False,
        "production_modified": False,
        "oggm_run": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Candidate D2 frozen-gain sign-channel evaluation\n\n"
        f"Exact annual closure: `{closure:.3e} degC`.\n\n"
        "This is a post-hoc decomposition of the authorized D2 gain, not a new EnKF or a production candidate.\n",
        encoding="utf-8",
    )
    sources = {"static": STATIC, "real_d2": REAL, "driver": DRIVER, "proxy6_archive": proxy_source}
    manifest = {name: {"path": str(path), "sha256": core.sha256(path)} for name, path in sources.items()}
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
