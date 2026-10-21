"""Build a read-only publication synthesis for the accepted LIA curves.

The script joins already formal evidence only.  It neither launches OGGM nor
changes the accepted registry, controller state, curve-shape screen, or any
source result.  Its mass-balance series are volume-change-equivalent balances,
not directly saved climatic surface mass balance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_lia_191_retained_32_mean_geometry_mass_balance_v1 as geometry_base


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
OUT = BASE / "lia_191_curve_publication_synthesis_v1"

REGISTRY = BASE / "lia_191_accepted_58_registry_v1/accepted_58_curve_metrics.csv"
CURVES = BASE / "lia_191_accepted_58_curves_visualization_v1/accepted_58_annual_length_curves.csv"
SHAPES = BASE / "lia_191_accepted_58_curve_shape_screen_v1/accepted_58_curve_shape_metrics.csv"
EVIDENCE_INDEX = BASE / "lia_191_accepted55_evidence_index_v1/accepted55_evidence_index.csv"
REMAINING = BASE / "lia_191_remaining133_mechanism_exhaustion_after_system3910_v1/remaining133_mechanism_inventory.csv"
REMAINING_SUMMARY = BASE / "lia_191_remaining133_mechanism_exhaustion_after_system3910_v1/summary.json"
STAGE_A = BASE / "lia_191_discovery16_stage_a_terminal_adjudication_v1/summary.json"
STAGE_B = BASE / "lia_191_discovery15_stage_b_terminal_adjudication_v1/summary.json"
PARITY = BASE / "lia_191_system2005_1d_2d_climate_area_budget_parity_v1"
TWO_D_TERMINAL = BASE / "lia_191_system2005_system7_79_terminal_adjudication_v1/summary.json"
CURRENT_STATE = ROOT / "LIA_CURRENT_STATE.json"
EXECUTION_CONTRACT = ROOT / "LIA_EXECUTION_CONTRACT.md"
FACTS = ROOT / "LIA_PROJECT_FACTS.md"

ICE_DENSITY_KG_M3 = 900.0
YEARS = np.arange(1500, 2001, dtype=int)
SHAPE_ORDER = ["retained32", "excluded26"]
SHAPE_LABELS = {
    "retained32": "Retained non-near-monotonic (n=32)",
    "excluded26": "Temporarily excluded near-monotonic (n=26)",
    "all58": "All formally accepted (n=58)",
}
COLORS = {"retained32": "#287271", "excluded26": "#d77a61", "all58": "#3f5f8f"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def scalar(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def first_curve_sources() -> pd.DataFrame:
    curves = pd.read_csv(CURVES)
    curves["system_id"] = curves["system_id"].astype(int)
    return curves.sort_values(["system_id", "year"]).groupby("system_id", as_index=False).first()


def resolve_parameter_record(curve: pd.Series) -> dict[str, Any]:
    system_id = int(curve["system_id"])
    route = str(curve["route"])
    curve_source = ROOT / str(curve["source"])
    if route == "frozen-legacy":
        parameter_source = curve_source.parent / "summary.json"
        overlay_source: Path | None = None
        config = load_json(parameter_source)
        parameters = config["parameters"]
        policy = parameters.get("temperature_offset_policy") or {}
        equilibrium_delta = scalar(
            parameters.get("spinup_delta_t_c", policy.get("spinup_delta_t_c", parameters.get("delta_t_c")))
        )
        forward_endpoint = scalar(
            parameters.get("forward_delta_t_c", policy.get("forward_delta_t_c", parameters.get("delta_t_c")))
        )
        stage2 = parameters
    else:
        overlay_source = curve_source.parent / "config_snapshot.json"
        overlay = load_json(overlay_source)
        parameter_source = overlay_source
        config = overlay
        stage2 = config.get("stage2")
        if stage2 is None:
            baseline_root = config.get("baseline_root")
            if not baseline_root:
                raise RuntimeError(f"system {system_id}: accepted config has no Stage-2 or baseline root")
            parameter_source = ROOT / str(baseline_root) / "config_snapshot.json"
            config = load_json(parameter_source)
            stage2 = config.get("stage2")
        if stage2 is None:
            raise RuntimeError(f"system {system_id}: baseline config has no Stage-2 parameters")
        equilibrium_delta = scalar(stage2.get("delta_t_c"))
        forward_endpoint = scalar(config.get("forward_delta_t_end_c"))

    record = {
        "system_id": system_id,
        "rgi_id": curve["rgi_id"],
        "route": route,
        "parameter_source": relative(parameter_source),
        "curve_source": relative(curve_source),
        "melt_f": scalar(stage2.get("melt_f")),
        "prcp_fac": scalar(stage2.get("prcp_fac")),
        "glen_a_pa_3_s_1": scalar(stage2.get("glen_a", stage2.get("glen_a_pa_3_s_1"))),
        "fs": scalar(stage2.get("fs", 0.0)),
        "temp_melt_c": scalar(stage2.get("temp_melt_c")),
        "base_temp_bias_c": scalar(stage2.get("base_temp_bias_c")),
        "equilibrium_delta_t_c": equilibrium_delta,
        "forward_delta_t_end_c": forward_endpoint,
        "late_delta_t_2000_c": None,
        "ramp_shape_exponent": scalar(config.get("ramp_shape_exponent")),
        "early_pulse_amplitude_c": scalar((config.get("early_warm_pulse") or {}).get("amplitude_c")),
        "minimum_timestep_seconds": scalar(config.get("minimum_timestep_seconds")),
    }
    if overlay_source is not None:
        overlay = load_json(overlay_source)
        record["late_delta_t_2000_c"] = scalar(overlay.get("late_delta_t_2000_c"))
        if record["forward_delta_t_end_c"] is None:
            record["forward_delta_t_end_c"] = scalar(overlay.get("forward_delta_t_end_c"))
        if record["ramp_shape_exponent"] is None:
            record["ramp_shape_exponent"] = scalar(overlay.get("ramp_shape_exponent"))
        pulse = overlay.get("early_warm_pulse") or {}
        if record["early_pulse_amplitude_c"] is None:
            record["early_pulse_amplitude_c"] = scalar(pulse.get("amplitude_c"))
        if record["minimum_timestep_seconds"] is None:
            record["minimum_timestep_seconds"] = scalar(overlay.get("minimum_timestep_seconds"))
    record["log10_glen_a"] = float(np.log10(record["glen_a_pa_3_s_1"]))
    return record


def build_accepted_geometry(registry: pd.DataFrame, shapes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    evidence = {
        int(row["system_id"]): row
        for row in pd.read_csv(EVIDENCE_INDEX, dtype=str).to_dict(orient="records")
    }
    shape_map = shapes.set_index("system_id")["shape_group"].to_dict()
    records: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for metric in registry.astype(object).where(pd.notna(registry), "").to_dict(orient="records"):
        system_id = int(metric["system_id"])
        source, length_key = geometry_base.resolve_source(metric, evidence)
        arrays = geometry_base.interpolate_source(source, length_key)
        area_m2 = arrays["area_km2"] * 1.0e6
        volume_m3 = arrays["volume_km3"] * 1.0e9
        reference_area_m2 = 0.5 * (area_m2[1:] + area_m2[:-1])
        net_mb = ICE_DENSITY_KG_M3 * np.diff(volume_m3) / reference_area_m2
        manifest.append(
            {
                "system_id": system_id,
                "rgi_id": metric["rgi_id"],
                "route": metric["route"],
                "shape_group": shape_map[system_id],
                "curve_source": relative(source),
                "curve_source_sha256": sha256(source),
            }
        )
        for offset, year in enumerate(YEARS, start=1):
            records.append(
                {
                    "system_id": system_id,
                    "rgi_id": metric["rgi_id"],
                    "route": metric["route"],
                    "shape_group": shape_map[system_id],
                    "year": int(year),
                    "length_km": arrays["length_km"][offset],
                    "area_km2": arrays["area_km2"][offset],
                    "volume_km3": arrays["volume_km3"][offset],
                    "mass_balance_reference_area_km2": reference_area_m2[offset - 1] / 1.0e6,
                    "net_volume_change_equivalent_mb_kg_m2_yr": net_mb[offset - 1],
                    "curve_source": relative(source),
                }
            )
    table = pd.DataFrame(records)
    if len(table) != 58 * len(YEARS) or table["system_id"].nunique() != 58:
        raise RuntimeError("accepted geometry table does not contain exactly 58 complete curves")
    numeric = table[["length_km", "area_km2", "volume_km3", "net_volume_change_equivalent_mb_kg_m2_yr"]]
    if not np.isfinite(numeric.to_numpy()).all() or (table[["length_km", "area_km2", "volume_km3"]] <= 0).any().any():
        raise RuntimeError("accepted geometry table contains invalid values")
    starts = table[table["year"] == 1500].set_index("system_id")
    for variable in ["length_km", "area_km2", "volume_km3"]:
        table[f"{variable}_relative_1500"] = table[variable] / table["system_id"].map(starts[variable])
    return table, pd.DataFrame(manifest)


def annual_group_statistics(table: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    groups = {
        "all58": table,
        "retained32": table[table["shape_group"] == "retained32"],
        "excluded26": table[table["shape_group"] == "excluded26"],
    }
    variables = ["length_km", "area_km2", "volume_km3", "net_volume_change_equivalent_mb_kg_m2_yr"]
    for group_name, subset in groups.items():
        for year, rows in subset.groupby("year", sort=True):
            out: dict[str, Any] = {"group": group_name, "year": int(year), "system_count": rows["system_id"].nunique()}
            for variable in variables:
                values = rows[variable].to_numpy(float)
                out[f"mean_{variable}"] = float(np.mean(values))
                out[f"median_{variable}"] = float(np.median(values))
                out[f"q25_{variable}"] = float(np.quantile(values, 0.25))
                out[f"q75_{variable}"] = float(np.quantile(values, 0.75))
            for variable in ["length_km", "area_km2", "volume_km3"]:
                out[f"mean_{variable}_relative_1500"] = float(rows[f"{variable}_relative_1500"].mean())
            weights = rows["mass_balance_reference_area_km2"].to_numpy(float)
            mb = rows["net_volume_change_equivalent_mb_kg_m2_yr"].to_numpy(float)
            out["area_weighted_net_volume_change_equivalent_mb_kg_m2_yr"] = float(np.average(mb, weights=weights))
            frames.append(pd.DataFrame([out]))
    result = pd.concat(frames, ignore_index=True)
    for column in ["mean_net_volume_change_equivalent_mb_kg_m2_yr", "area_weighted_net_volume_change_equivalent_mb_kg_m2_yr"]:
        result[f"{column}_21yr"] = result.groupby("group")[column].transform(
            lambda values: values.rolling(21, center=True, min_periods=1).mean()
        )
    return result


def parameter_statistics(parameters: pd.DataFrame) -> pd.DataFrame:
    variables = [
        "melt_f", "prcp_fac", "log10_glen_a", "equilibrium_delta_t_c",
        "forward_delta_t_end_c", "base_temp_bias_c",
    ]
    rows: list[dict[str, Any]] = []
    grouping = [("shape_group", parameters["shape_group"]), ("route", parameters["route"])]
    for grouping_name, labels in grouping:
        for group, subset in parameters.groupby(labels):
            for variable in variables:
                values = pd.to_numeric(subset[variable], errors="coerce").dropna()
                rows.append(
                    {
                        "grouping": grouping_name,
                        "group": group,
                        "parameter": variable,
                        "n": int(len(values)),
                        "mean": float(values.mean()) if len(values) else np.nan,
                        "median": float(values.median()) if len(values) else np.nan,
                        "q25": float(values.quantile(0.25)) if len(values) else np.nan,
                        "q75": float(values.quantile(0.75)) if len(values) else np.nan,
                        "minimum": float(values.min()) if len(values) else np.nan,
                        "maximum": float(values.max()) if len(values) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def discovery_overlay(remaining: pd.DataFrame, stage_a: dict[str, Any], stage_b: dict[str, Any]) -> pd.DataFrame:
    output = remaining.copy()
    output["current_discovery_status"] = "not_in_discovery16"
    output["current_discovery_evidence"] = ""
    status_sets = {
        "stage_b_short_pass_steady_not_authorized": set(stage_b["stage_b_pass_systems"]),
        "stage_b_technical_failure_retry_decision_required": set(stage_b["stage_b_technical_failure_systems"]),
        "stage_a_scientific_failure": set(stage_a["stage_a_scientific_failure_systems"]),
    }
    for status, systems in status_sets.items():
        mask = output["system_id"].isin(systems)
        output.loc[mask, "current_discovery_status"] = status
        output.loc[mask, "current_discovery_evidence"] = relative(STAGE_B if status.startswith("stage_b") else STAGE_A)
    if output["system_id"].nunique() != 133:
        raise RuntimeError("remaining evidence table does not contain exactly 133 systems")
    return output


def build_model_comparison() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    parity_summary = load_json(PARITY / "summary.json")
    parity_annual = pd.read_csv(PARITY / "annual_1d_2d_budget.csv")
    terminal = load_json(TWO_D_TERMINAL)
    one_d_initial_volume = parity_summary["initial_geometry_nonparity"]["one_d_natural_year9_volume_m3"]
    one_d_initial_area = parity_summary["initial_geometry_nonparity"]["one_d_natural_year9_area_m2"]
    two_d_initial_volume = parity_summary["two_d_mass_budget"]["initial_volume_m3"]
    two_d_initial_area = parity_summary["area_support"]["two_d_initial_active_area_m2"]
    parity_annual["one_d_volume_change_percent"] = 100.0 * (parity_annual["one_d_volume_m3"] / one_d_initial_volume - 1.0)
    parity_annual["two_d_volume_change_percent"] = 100.0 * (parity_annual["two_d_volume_m3"] / two_d_initial_volume - 1.0)
    parity_annual["one_d_model_reported_area_change_percent"] = 100.0 * (
        parity_annual["one_d_area_m2"] / one_d_initial_area - 1.0
    )
    parity_annual["two_d_model_reported_area_change_percent"] = 100.0 * (
        parity_annual["two_d_area_m2"] / two_d_initial_area - 1.0
    )

    final = parity_annual.iloc[-1]
    rows = [
        {
            "model_family": "1D conservative flowline",
            "system_id": 2005,
            "stage": "50-year parent-climate comparison",
            "volume_change_percent": float(100.0 * (final["one_d_volume_m3"] / one_d_initial_volume - 1.0)),
            "active_support_area_change_percent": float(
                100.0 * parity_summary["area_support"]["one_d_actual_area_change_fraction"]
            ),
            "scientific_outcome": "comparison-only natural-state evidence",
            "formal_reason": "not a paired formal model-selection experiment",
            "source": relative(PARITY / "summary.json"),
        },
        {
            "model_family": "2D unmasked planform",
            "system_id": 2005,
            "stage": "50-year parent-climate comparison",
            "volume_change_percent": float(100.0 * (final["two_d_volume_m3"] / two_d_initial_volume - 1.0)),
            "active_support_area_change_percent": float(
                100.0 * parity_summary["area_support"]["two_d_active_area_change_fraction"]
            ),
            "scientific_outcome": "scientific failure",
            "formal_reason": "planform support expansion climate-area positive feedback",
            "source": relative(PARITY / "summary.json"),
        },
    ]
    for system_id, record in terminal["system_results"].items():
        rows.append(
            {
                "model_family": "2D masked planform",
                "system_id": int(system_id),
                "stage": record["run_role"],
                "volume_change_percent": 100.0 * float(record["volume_change_fraction"]),
                "active_support_area_change_percent": np.nan,
                "scientific_outcome": record["scientific_class"],
                "formal_reason": record["formal_reason"],
                "source": relative(TWO_D_TERMINAL),
            }
        )
    return pd.DataFrame(rows), parity_annual, parity_summary


def style(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)


def plot_ensemble(stats: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), sharex=True)
    variables = [
        ("length_km", "A. Length", "Change from 1500 (%)"),
        ("area_km2", "B. Area", "Change from 1500 (%)"),
        ("volume_km3", "C. Volume", "Change from 1500 (%)"),
    ]
    for axis, (variable, title, ylabel) in zip(axes.ravel()[:3], variables):
        for group in ["all58", "retained32", "excluded26"]:
            subset = stats[stats["group"] == group]
            axis.plot(
                subset["year"],
                100.0 * (subset[f"mean_{variable}_relative_1500"] - 1.0),
                color=COLORS[group], linewidth=2.1 if group != "all58" else 1.5,
                linestyle="-" if group != "all58" else "--", label=SHAPE_LABELS[group],
            )
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        style(axis)
    mb_axis = axes[1, 1]
    for group in ["all58", "retained32", "excluded26"]:
        subset = stats[stats["group"] == group]
        mb_axis.plot(
            subset["year"], subset["mean_net_volume_change_equivalent_mb_kg_m2_yr_21yr"],
            color=COLORS[group], linewidth=2.0 if group != "all58" else 1.4,
            linestyle="-" if group != "all58" else "--", label=SHAPE_LABELS[group],
        )
    mb_axis.axhline(0, color="#555555", linewidth=0.8)
    mb_axis.set_title("D. Net volume-change-equivalent mass balance (21-year mean)")
    mb_axis.set_ylabel("kg m$^{-2}$ yr$^{-1}$")
    style(mb_axis)
    for axis in axes[-1, :]:
        axis.set_xlabel("Year")
    for axis in axes.ravel():
        axis.set_xlim(1500, 2000)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Accepted LIA ensemble: curve-shape-stratified geometry and net balance", y=1.035, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUT / "accepted58_shape_stratified_ensemble.png"
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_parameters(parameters: pd.DataFrame) -> Path:
    variables = [
        ("melt_f", "Melt factor"), ("prcp_fac", "Precipitation factor"),
        ("log10_glen_a", "log10 Glen A"), ("equilibrium_delta_t_c", "Equilibrium $\\Delta T$ (°C)"),
        ("forward_delta_t_end_c", "Forward endpoint $\\Delta T$ (°C)"),
        ("base_temp_bias_c", "Base temperature bias (°C)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.4))
    for axis, (variable, label) in zip(axes.ravel(), variables):
        arrays = [pd.to_numeric(parameters.loc[parameters["shape_group"] == group, variable], errors="coerce").dropna() for group in SHAPE_ORDER]
        axis.boxplot(arrays, labels=["Retained 32", "Excluded 26"], showfliers=True, widths=0.55)
        axis.set_title(label)
        axis.tick_params(axis="x", rotation=12)
        style(axis)
    fig.suptitle("Accepted-58 parameter distributions (shape groups are route-confounded)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = OUT / "accepted58_parameter_distributions.png"
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_remaining(remaining: pd.DataFrame) -> Path:
    mechanism_counts = remaining["mechanism_class"].value_counts().head(14).sort_values()
    discovery_counts = remaining["current_discovery_status"].value_counts().sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.4), gridspec_kw={"width_ratios": [1.6, 1.0]})
    axes[0].barh(mechanism_counts.index, mechanism_counts.values, color="#3f5f8f")
    axes[0].set_title("A. Largest registered closure/mechanism classes")
    axes[0].set_xlabel("Remaining systems")
    axes[0].tick_params(axis="y", labelsize=8)
    style(axes[0])
    labels = [label.replace("_", "\n") for label in discovery_counts.index]
    axes[1].bar(labels, discovery_counts.values, color=["#d77a61", "#999999", "#287271", "#bc8f3f"][: len(labels)])
    axes[1].set_title("B. Current discovery16 overlay")
    axes[1].set_ylabel("Systems")
    axes[1].tick_params(axis="x", rotation=20, labelsize=8)
    style(axes[1])
    fig.suptitle("Route-stratified failed and unresolved evidence (133 remaining systems)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUT / "remaining133_route_stratified_evidence.png"
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_1d_2d(parity: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0))
    axes[0].plot(parity["year"], parity["one_d_volume_change_percent"], color="#287271", linewidth=2.1, label="1-D")
    axes[0].plot(parity["year"], parity["two_d_volume_change_percent"], color="#d77a61", linewidth=2.1, label="2-D unmasked")
    axes[0].set_title("A. Volume response")
    axes[0].set_ylabel("Change from initial (%)")
    axes[0].legend(frameon=False)
    axes[1].plot(parity["year"], parity["one_d_model_reported_area_change_percent"], color="#287271", linewidth=2.1)
    axes[1].plot(parity["year"], parity["two_d_model_reported_area_change_percent"], color="#d77a61", linewidth=2.1)
    axes[1].set_title("B. Model-reported area response")
    axes[1].set_ylabel("Change from initial (%)")
    axes[2].plot(parity["year"], parity["one_d_annual_source_m3"] / 1e6, color="#287271", linewidth=1.5, label="1-D actual")
    axes[2].plot(parity["year"], parity["two_d_annual_source_m3"] / 1e6, color="#d77a61", linewidth=1.5, label="2-D actual")
    axes[2].plot(parity["year"], parity["two_d_fixed_initial_geometry_source_m3"] / 1e6, color="#555555", linewidth=1.2, linestyle="--", label="2-D fixed geometry")
    axes[2].set_title("C. Annual climate source")
    axes[2].set_ylabel("10$^6$ m$^3$ yr$^{-1}$")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_xlabel("Model year")
        style(axis)
    fig.suptitle("System 2005: 1-D versus unmasked 2-D climate-area budget", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUT / "system2005_1d_2d_publication_comparison.png"
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"versioned output already exists: {OUT}")

    current = load_json(CURRENT_STATE)
    if current["counts"] != {"accepted": 58, "remaining": 133, "active": 0, "queued": 0}:
        raise RuntimeError(f"unexpected current counts: {current['counts']}")
    next_action = current["next_action"]
    if not str(next_action["type"]).startswith("decision-required") or next_action["launch_authorized"]:
        raise RuntimeError("analysis requires the frozen decision-required, no-launch state")

    registry = pd.read_csv(REGISTRY)
    registry["system_id"] = registry["system_id"].astype(int)
    shapes = pd.read_csv(SHAPES)
    shapes["system_id"] = shapes["system_id"].astype(int)
    shapes["shape_group"] = np.where(
        shapes["temporary_exclusion_nearly_monotonic_decline"].astype(str).str.lower() == "true",
        "excluded26", "retained32",
    )
    if registry["system_id"].nunique() != 58 or shapes["system_id"].nunique() != 58:
        raise RuntimeError("accepted registry or shape screen is not exactly 58 unique systems")
    if shapes["shape_group"].value_counts().to_dict() != {"retained32": 32, "excluded26": 26}:
        raise RuntimeError("shape screen is not exactly retained 32 / excluded 26")
    if set(registry["system_id"]) != set(shapes["system_id"]):
        raise RuntimeError("accepted registry and curve-shape populations differ")

    OUT.mkdir(parents=True)
    geometry, geometry_sources = build_accepted_geometry(registry, shapes)
    annual = annual_group_statistics(geometry)

    first_sources = first_curve_sources()
    parameters = pd.DataFrame([resolve_parameter_record(row) for _, row in first_sources.iterrows()])
    parameters = parameters.merge(shapes[["system_id", "shape_group"]], on="system_id", how="left", validate="one_to_one")
    if len(parameters) != 58 or parameters[["melt_f", "prcp_fac", "glen_a_pa_3_s_1", "equilibrium_delta_t_c"]].isna().any().any():
        raise RuntimeError("parameter extraction is incomplete for the accepted 58")
    parameter_summary = parameter_statistics(parameters)

    remaining = pd.read_csv(REMAINING)
    remaining["system_id"] = remaining["system_id"].astype(int)
    stage_a = load_json(STAGE_A)
    stage_b = load_json(STAGE_B)
    remaining_current = discovery_overlay(remaining, stage_a, stage_b)
    mechanism_counts = remaining_current.groupby(["current_class", "mechanism_class"], dropna=False).size().reset_index(name="count")
    discovery_counts = remaining_current.groupby("current_discovery_status").size().reset_index(name="count")

    model_comparison, parity_annual, parity_summary = build_model_comparison()

    geometry.to_csv(OUT / "accepted58_annual_geometry_and_net_mass_balance.csv", index=False)
    annual.to_csv(OUT / "accepted58_shape_stratified_annual_ensemble_statistics.csv", index=False)
    geometry_sources.to_csv(OUT / "accepted58_curve_source_manifest.csv", index=False)
    parameters.to_csv(OUT / "accepted58_parameter_table.csv", index=False)
    parameter_summary.to_csv(OUT / "accepted58_parameter_distribution_statistics.csv", index=False)
    remaining_current.to_csv(OUT / "remaining133_route_stratified_evidence.csv", index=False)
    mechanism_counts.to_csv(OUT / "remaining133_mechanism_class_counts.csv", index=False)
    discovery_counts.to_csv(OUT / "current_discovery16_status_counts.csv", index=False)
    model_comparison.to_csv(OUT / "one_d_vs_two_d_comparison_table.csv", index=False)
    parity_annual.to_csv(OUT / "system2005_1d_2d_annual_plot_data.csv", index=False)

    figures = {
        "ensemble": plot_ensemble(annual).name,
        "parameters": plot_parameters(parameters).name,
        "remaining": plot_remaining(remaining_current).name,
        "one_d_two_d": plot_1d_2d(parity_annual).name,
    }

    source_paths = [
        DRIVER, FACTS, EXECUTION_CONTRACT, CURRENT_STATE, REGISTRY, CURVES, SHAPES,
        EVIDENCE_INDEX, REMAINING, REMAINING_SUMMARY, STAGE_A, STAGE_B,
        PARITY / "summary.json", PARITY / "annual_1d_2d_budget.csv", TWO_D_TERMINAL,
    ]
    manifest = pd.DataFrame(
        [{"source": relative(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in source_paths]
    )
    manifest.to_csv(OUT / "analysis_source_manifest.csv", index=False)

    change_summary: dict[str, Any] = {}
    for group in ["all58", "retained32", "excluded26"]:
        subset = annual[annual["group"] == group].set_index("year")
        change_summary[group] = {
            variable: float(100.0 * (subset.loc[2000, f"mean_{variable}_relative_1500"] - 1.0))
            for variable in ["length_km", "area_km2", "volume_km3"]
        }
        change_summary[group]["mean_net_mb_kg_m2_yr"] = float(
            subset["mean_net_volume_change_equivalent_mb_kg_m2_yr"].mean()
        )

    route_shape = (
        parameters.groupby(["route", "shape_group"]).size().reset_index(name="count")
        .pivot(index="route", columns="shape_group", values="count").fillna(0).astype(int)
    )
    summary = {
        "schema": "lia-191-curve-publication-synthesis-v1",
        "status": "read-only-publication-synthesis-complete",
        "canonical_state_id": current["state_id"],
        "canonical_state_class": "decision-required",
        "validator_expected_status": "pass",
        "population_counts": {"total": 191, "accepted": 58, "retained": 32, "temporarily_excluded": 26, "remaining": 133},
        "mass_balance_definition": "rho_ice * (V_t - V_t-1) / mean(A_t, A_t-1), rho_ice=900 kg m-3",
        "mass_balance_warning": "volume-change-equivalent/geodetic balance derived from annual geometry; not directly saved climatic surface mass balance",
        "equal_glacier_mean_change_1500_2000_percent": change_summary,
        "accepted_parameter_rows": len(parameters),
        "accepted_parameter_core_complete": True,
        "route_shape_confounding_counts": route_shape.to_dict(orient="index"),
        "remaining_evidence": {
            "mechanism_closed": int((remaining_current["current_class"] == "mechanism_closed").sum()),
            "mathematically_unreachable": int((remaining_current["current_class"] == "mathematically_unreachable").sum()),
            "discovery_stage_b_short_pass_steady_not_authorized": len(stage_b["stage_b_pass_systems"]),
            "discovery_stage_b_technical_failures": len(stage_b["stage_b_technical_failure_systems"]),
            "discovery_stage_a_scientific_failures": len(stage_a["stage_a_scientific_failure_systems"]),
        },
        "one_d_two_d_findings": {
            "unmasked_2d_volume_change_percent": float(100.0 * (parity_summary["two_d_mass_budget"]["final_volume_m3"] / parity_summary["two_d_mass_budget"]["initial_volume_m3"] - 1.0)),
            "unmasked_2d_active_area_change_percent": float(100.0 * parity_summary["area_support"]["two_d_active_area_change_fraction"]),
            "unmasked_2d_feedback_mechanism": parity_summary["scientific_conclusion"],
            "current_masked_2d_mechanism_promoted": False,
            "current_masked_2d_terminal_source": relative(TWO_D_TERMINAL),
        },
        "scientific_limits": [
            "The 32/26 split is a temporary curve-shape screen and does not change formal acceptance.",
            "Shape-group parameter contrasts are route-confounded and are descriptive, not causal.",
            "Area and volume are model diagnostics; length remains the primary observational validation variable.",
            "The 1-D/2-D comparison is mechanism evidence, not a paired population-wide model-selection experiment.",
            "No currently registered remaining-system mechanism is reopened by this analysis.",
        ],
        "model_run_performed": False,
        "controller_or_registry_modified": False,
        "accepted_count_effect": 0,
        "outputs": {
            "figures": figures,
            "tables": [
                "accepted58_annual_geometry_and_net_mass_balance.csv",
                "accepted58_shape_stratified_annual_ensemble_statistics.csv",
                "accepted58_parameter_table.csv",
                "accepted58_parameter_distribution_statistics.csv",
                "remaining133_route_stratified_evidence.csv",
                "remaining133_mechanism_class_counts.csv",
                "current_discovery16_status_counts.csv",
                "one_d_vs_two_d_comparison_table.csv",
                "system2005_1d_2d_annual_plot_data.csv",
                "analysis_source_manifest.csv",
            ],
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    readme = f"""# LIA curve publication synthesis v1

This immutable, read-only package joins the 58 formally accepted curves, the temporary 32/26 curve-shape split, the 133-system route inventory, the current discovery16 overlay, and existing 1-D/2-D mechanism evidence.

Key rules:

- Formal accepted count remains 58; the 32/26 split is descriptive only.
- Length is the primary observational validation variable.
- Area, volume, and mass balance are model diagnostics.
- Mass balance is derived from annual volume change and interval-mean area; it is not direct climatic surface mass balance.
- Parameter contrasts are route-confounded and must not be interpreted causally.
- No simulation was launched and no controller, registry, or source result was modified.

Reproduce from the project root with the environment already used by the existing pandas/matplotlib analysis drivers:

`python {relative(DRIVER)}`

The script refuses to overwrite this versioned output directory and validates the 58/32/26/133 counts before writing.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
