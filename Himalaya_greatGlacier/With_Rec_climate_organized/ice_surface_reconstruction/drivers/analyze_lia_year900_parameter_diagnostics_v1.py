from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
MILESTONE_DIR = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_milestone_balance_diagnostics_v1"
TX_DIR = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_all191_oggm_flowline_1d_v10_216_recovery_transaction_v1"
OUT = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_year900_parameter_diagnostics_v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def main():
    if OUT.exists():
        raise FileExistsError(f"immutable output already exists: {OUT}")
    OUT.mkdir(parents=True)

    states = pd.read_csv(MILESTONE_DIR / "cell_milestone_states.csv")
    states = states[states.milestone_year == 900].copy()
    matrix = pd.read_csv(TX_DIR / "level1_matrix.csv")
    parameter_columns = [
        "cell_ordinal", "lia_unique_id", "rgi_id", "delta_t_eq_degC",
        "base_temp_bias_degC", "effective_temp_bias_degC", "melt_f",
        "prcp_fac", "glen_a_pa_3_s_1", "fs", "temp_melt_degC",
    ]
    joined = states[[
        "cell_ordinal", "numerical_group", "first_confirmed_boundary_year",
        "volume_ratio_to_year0", "length_ratio_to_year0", "maximum_thickness_m",
    ]].merge(matrix[parameter_columns], on="cell_ordinal", how="left", validate="one_to_one")
    if len(joined) != 1337 or joined[parameter_columns].isna().any().any():
        raise RuntimeError("year-900 parameter join is incomplete")

    joined["stable_by_900"] = joined.numerical_group.eq("confirmed-stable")
    joined["log10_glen_a"] = np.log10(joined.glen_a_pa_3_s_1)
    joined.to_csv(OUT / "year900_all_cells_with_parameters.csv", index=False)
    joined[~joined.stable_by_900].to_csv(OUT / "year900_not_stable_cells.csv", index=False)

    delta = joined.groupby("delta_t_eq_degC").stable_by_900.agg(["sum", "count"]).reset_index()
    delta["not_stable"] = delta["count"] - delta["sum"]
    delta["stable_percent"] = 100 * delta["sum"] / delta["count"]
    delta = delta.rename(columns={"sum": "stable_cells", "count": "total_cells"})
    delta.to_csv(OUT / "delta_t_eq_stability_rates.csv", index=False)

    system_rows = []
    for system_id, group in joined.groupby("lia_unique_id"):
        stable = group[group.stable_by_900].sort_values("delta_t_eq_degC")
        one = group.iloc[0]
        system_rows.append({
            "lia_unique_id": int(system_id),
            "rgi_id": one.rgi_id,
            "minimum_delta_t_eq_stable_by_900_degC": np.nan if stable.empty else float(stable.iloc[0].delta_t_eq_degC),
            "stable_delta_count": int(len(stable)),
            "base_temp_bias_degC": float(one.base_temp_bias_degC),
            "melt_f": float(one.melt_f),
            "prcp_fac": float(one.prcp_fac),
            "glen_a_pa_3_s_1": float(one.glen_a_pa_3_s_1),
            "fs": float(one.fs),
            "temp_melt_degC": float(one.temp_melt_degC),
        })
    systems = pd.DataFrame(system_rows).sort_values("lia_unique_id")
    systems.to_csv(OUT / "system_minimum_stable_temperature_offset.csv", index=False)
    no_stable = systems[systems.stable_delta_count == 0]
    no_stable.to_csv(OUT / "systems_with_no_stable_delta_by_900.csv", index=False)

    comparison_rows = []
    compare = [
        "delta_t_eq_degC", "base_temp_bias_degC", "effective_temp_bias_degC",
        "melt_f", "prcp_fac", "glen_a_pa_3_s_1", "fs", "temp_melt_degC",
    ]
    for parameter in compare:
        for stable, label in [(True, "confirmed-stable"), (False, "not-confirmed-stable")]:
            values = joined.loc[joined.stable_by_900 == stable, parameter].to_numpy(dtype=float)
            comparison_rows.append({
                "parameter": parameter,
                "group": label,
                "n": len(values),
                "minimum": float(np.min(values)),
                "p25": float(np.quantile(values, 0.25)),
                "median": float(np.median(values)),
                "p75": float(np.quantile(values, 0.75)),
                "maximum": float(np.max(values)),
            })
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUT / "parameter_group_summary.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    blue, orange = "#167D9A", "#D95F43"
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), dpi=180)

    ax = axes[0, 0]
    ax.bar(delta.delta_t_eq_degC, delta.stable_percent, width=0.36, color=blue)
    for x, pct, n in zip(delta.delta_t_eq_degC, delta.stable_percent, delta.stable_cells):
        ax.text(x, pct + 2, f"{int(n)}/191", ha="center", va="bottom", fontsize=8)
    ax.set(title="Stable by year 900 versus temperature offset", xlabel="delta_t_eq (°C)", ylabel="Stable cells (%)", ylim=(0, 108))

    groups = [joined.loc[joined.stable_by_900, "effective_temp_bias_degC"], joined.loc[~joined.stable_by_900, "effective_temp_bias_degC"]]
    ax = axes[0, 1]
    bp = ax.boxplot(groups, tick_labels=["Stable", "Not stable"], patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], [blue, orange]): patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.set(title="Effective temperature bias", ylabel="base bias + delta_t_eq (°C)")

    groups = [joined.loc[joined.stable_by_900, "base_temp_bias_degC"], joined.loc[~joined.stable_by_900, "base_temp_bias_degC"]]
    ax = axes[0, 2]
    bp = ax.boxplot(groups, tick_labels=["Stable", "Not stable"], patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], [blue, orange]): patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.set(title="Inherited base temperature bias", ylabel="Base bias (°C)")

    for ax, column, title, ylabel in [
        (axes[1, 0], "melt_f", "Inherited melt factor", "melt_f"),
        (axes[1, 1], "prcp_fac", "Inherited precipitation factor", "prcp_fac"),
        (axes[1, 2], "log10_glen_a", "Inherited ice-deformation parameter", "log10(glen_a)")
    ]:
        groups = [joined.loc[joined.stable_by_900, column], joined.loc[~joined.stable_by_900, column]]
        bp = ax.boxplot(groups, tick_labels=["Stable", "Not stable"], patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], [blue, orange]): patch.set_facecolor(color); patch.set_alpha(0.75)
        ax.set(title=title, ylabel=ylabel)

    fig.suptitle("Year-900 numerical stability and Level-1 parameters\n531 stable cells vs 806 not confirmed stable", fontsize=15, y=1.01)
    fig.text(0.5, 0.005, "Only delta_t_eq varies within each glacier's seven-cell Level-1 row. Other parameters are inherited and their overlap is descriptive, not causal.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(OUT / "year900_parameter_overview.png", bbox_inches="tight")
    plt.close(fig)

    heat = joined.pivot(index="lia_unique_id", columns="delta_t_eq_degC", values="stable_by_900").astype(int)
    order = systems.assign(sort_key=systems.minimum_delta_t_eq_stable_by_900_degC.fillna(99)).sort_values(["sort_key", "lia_unique_id"]).lia_unique_id
    heat = heat.loc[order]
    fig, ax = plt.subplots(figsize=(8.5, 10.5), dpi=180)
    im = ax.imshow(heat.to_numpy(), aspect="auto", interpolation="nearest", cmap=matplotlib.colors.ListedColormap([orange, blue]), vmin=0, vmax=1)
    ax.set_xticks(range(len(heat.columns)), [f"{x:g}" for x in heat.columns])
    ax.set(title="Year-900 stability for every glacier-system × temperature-offset cell", xlabel="delta_t_eq (°C)", ylabel="191 glacier systems (sorted by minimum stable offset)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=blue, label="Confirmed stable"), Patch(color=orange, label="Not stable")], loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "year900_system_delta_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    glen_summary = comparison[comparison.parameter == "glen_a_pa_3_s_1"].to_dict(orient="records")
    summary = {
        "schema": "lia-191-year900-parameter-diagnostics-v1",
        "status": "complete-read-only-retrospective-analysis",
        "population_systems": 191,
        "parameter_cells": 1337,
        "stable_cells_by_900": int(joined.stable_by_900.sum()),
        "not_stable_cells_by_900": int((~joined.stable_by_900).sum()),
        "systems_with_any_stable_cell_by_900": int((systems.stable_delta_count > 0).sum()),
        "systems_with_no_stable_cell_by_900": no_stable.lia_unique_id.astype(int).tolist(),
        "temperature_offset_stability": delta.to_dict(orient="records"),
        "glen_a_group_summary": glen_summary,
        "constant_parameters": {"fs": sorted(joined.fs.unique().tolist()), "temp_melt_degC": sorted(joined.temp_melt_degC.unique().tolist())},
        "classification_source": "lia_191_milestone_balance_diagnostics_v1 proposed numerical stability rule",
        "physical_plausibility_gate_applied": False,
        "simulation_launched": False,
    }
    with (OUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    hashes = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "artifact_hashes.json":
            hashes[path.name] = sha256(path)
    with (OUT / "artifact_hashes.json").open("w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)
        f.write("\n")

    print(delta.to_string(index=False))
    print(no_stable.to_string(index=False))
    print(json.dumps({"output": str(OUT), "hashes": hashes}, indent=2))


if __name__ == "__main__":
    main()
