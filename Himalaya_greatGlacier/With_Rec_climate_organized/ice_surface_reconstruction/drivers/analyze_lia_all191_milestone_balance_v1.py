from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
TX = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_all191_oggm_flowline_1d_v10_216_recovery_transaction_v1"
OUT = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_milestone_balance_diagnostics_v1"
MILESTONES = [300, 600, 900, 1500, 2100, 3000]
ALL_BOUNDARIES = list(range(300, 3001, 300))

# User-approved prospective proposal, used here only for retrospective diagnosis.
VOL_CYCLE_MAX = 0.005
VOL_90_MAX = 0.015
LENGTH_CYCLE_MAX_M = 300.0
LENGTH_90_MAX_M = 300.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def window_metrics(by_year: dict[int, dict], years: list[int]):
    if any(y not in by_year for y in years):
        return None
    s = [by_year[y] for y in years]
    volumes = np.asarray([x["volume_m3"] for x in s], dtype=float)
    lengths = np.asarray([x["length_m"] for x in s], dtype=float)
    cycle_v = np.abs(np.diff(volumes)) / np.maximum(np.abs(volumes[:-1]), 1e-30)
    drift_v = abs(volumes[-1] - volumes[0]) / max(abs(volumes[0]), 1e-30)
    cycle_l = np.abs(np.diff(lengths))
    drift_l = abs(lengths[-1] - lengths[0])
    passed = bool(
        np.all(cycle_v <= VOL_CYCLE_MAX)
        and drift_v <= VOL_90_MAX
        and np.all(cycle_l <= LENGTH_CYCLE_MAX_M)
        and drift_l <= LENGTH_90_MAX_M
    )
    return {
        "volume_cycle_max": float(np.max(cycle_v)),
        "volume_90": float(drift_v),
        "length_cycle_max_m": float(np.max(cycle_l)),
        "length_90_m": float(drift_l),
        "passed": passed,
    }


def first_confirmed_boundary(by_year: dict[int, dict]):
    for year in ALL_BOUNDARIES:
        first = window_metrics(by_year, [year - 90, year - 60, year - 30, year])
        confirm = window_metrics(by_year, [year - 60, year - 30, year, year + 30])
        if first and confirm and first["passed"] and confirm["passed"]:
            return year, first, confirm
    return None, None, None


def state_at(by_year: dict[int, dict], milestone: int, terminal_status: str):
    if milestone in by_year:
        return by_year[milestone], False
    available = sorted(by_year)
    if terminal_status == "converged" and available and max(available) < milestone:
        return by_year[max(available)], True
    return None, False


def q(series, p):
    a = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.quantile(a, p)) if len(a) else np.nan


def main():
    if OUT.exists():
        raise FileExistsError(f"immutable output already exists: {OUT}")
    OUT.mkdir(parents=True)

    full = pd.read_csv(TX / "level1_matrix.csv").set_index("cell_ordinal", drop=False)
    production = pd.read_csv(TX / "production_manifest.csv")
    reused = pd.read_csv(TX / "reused_v10_203_converged_results.csv")

    result_paths: dict[int, Path] = {}
    for row in production.itertuples(index=False):
        result_paths[int(row.cell_ordinal)] = ROOT / "w216/m" / row.member_key / "c/result.json"
    for row in reused.itertuples(index=False):
        result_paths[int(row.cell_ordinal)] = Path(row.result_path)

    if len(result_paths) != 1337 or set(result_paths) != set(range(1, 1338)):
        raise RuntimeError(f"result coverage mismatch: {len(result_paths)}")

    cell_rows = []
    status_counts: dict[str, int] = {}
    carried_count = 0
    for ordinal in range(1, 1338):
        meta = full.loc[ordinal]
        result_path = result_paths[ordinal]
        result = read_json(result_path)
        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
        diag_path = result_path.parent / "diagnostics.jsonl"
        diagnostics = read_jsonl(diag_path)
        by_year = {int(x["year"]): x for x in diagnostics}
        if 0 not in by_year:
            raise RuntimeError(f"year zero missing: {diag_path}")
        initial = by_year[0]
        first_year, first_metrics, confirmation_metrics = first_confirmed_boundary(by_year)

        for milestone in MILESTONES:
            state, carried = state_at(by_year, milestone, result["status"])
            if state is None:
                raise RuntimeError(f"milestone {milestone} unavailable: {diag_path}")
            carried_count += int(carried)
            stable = first_year is not None and first_year <= milestone
            direct = window_metrics(by_year, [milestone - 90, milestone - 60, milestone - 30, milestone])
            cell_rows.append(
                {
                    "cell_ordinal": ordinal,
                    "lia_unique_id": int(meta["lia_unique_id"]),
                    "rgi_id": str(meta["rgi_id"]),
                    "delta_t_eq_degC": float(meta["delta_t_eq_degC"]),
                    "milestone_year": milestone,
                    "numerical_group": "confirmed-stable" if stable else "not-confirmed-stable",
                    "first_confirmed_boundary_year": first_year,
                    "terminal_original_contract_status": result["status"],
                    "state_carried_forward_after_terminal_equilibrium": carried,
                    "state_year_used": int(state["year"]),
                    "volume_m3": float(state["volume_m3"]),
                    "volume_ratio_to_year0": float(state["volume_m3"] / initial["volume_m3"]),
                    "volume_change_percent_from_year0": float(100 * (state["volume_m3"] / initial["volume_m3"] - 1)),
                    "area_m2": float(state["area_m2"]),
                    "area_ratio_to_year0": float(state["area_m2"] / initial["area_m2"]),
                    "length_m": float(state["length_m"]),
                    "length_ratio_to_year0": float(state["length_m"] / initial["length_m"]),
                    "length_change_km_from_year0": float((state["length_m"] - initial["length_m"]) / 1000),
                    "maximum_thickness_m": float(state["maximum_thickness_m"]),
                    "maximum_thickness_change_m_from_year0": float(state["maximum_thickness_m"] - initial["maximum_thickness_m"]),
                    "boundary_margin_km": float(state["boundary_margin_m"] / 1000),
                    "milestone_window_available": direct is not None,
                    "milestone_volume_cycle_max": None if direct is None else direct["volume_cycle_max"],
                    "milestone_volume_90": None if direct is None else direct["volume_90"],
                    "milestone_length_cycle_max_m": None if direct is None else direct["length_cycle_max_m"],
                    "milestone_length_90_m": None if direct is None else direct["length_90_m"],
                    "first_pass_volume_90": None if first_metrics is None else first_metrics["volume_90"],
                    "confirmation_volume_90": None if confirmation_metrics is None else confirmation_metrics["volume_90"],
                }
            )

    cells = pd.DataFrame(cell_rows)
    cells.to_csv(OUT / "cell_milestone_states.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    summaries = []
    for milestone in MILESTONES:
        ym = cells[cells.milestone_year == milestone]
        stable = ym[ym.numerical_group == "confirmed-stable"]
        unstable = ym[ym.numerical_group == "not-confirmed-stable"]
        system_group = ym.groupby("lia_unique_id")["numerical_group"].apply(list)
        any_stable = int(sum("confirmed-stable" in values for values in system_group))
        all_stable = int(sum(all(v == "confirmed-stable" for v in values) for values in system_group))
        first_count = int((cells.first_confirmed_boundary_year == milestone).sum() / len(MILESTONES))
        row = {
            "milestone_year": milestone,
            "confirmed_stable_cells": len(stable),
            "newly_confirmed_at_boundary_cells": first_count,
            "not_confirmed_stable_cells": len(unstable),
            "systems_with_at_least_one_stable_cell": any_stable,
            "systems_with_all_seven_stable_cells": all_stable,
        }
        for label, group in [("stable", stable), ("unbalanced", unstable)]:
            for metric in [
                "volume_ratio_to_year0",
                "length_ratio_to_year0",
                "length_change_km_from_year0",
                "maximum_thickness_m",
                "area_ratio_to_year0",
            ]:
                row[f"{label}_{metric}_p25"] = q(group[metric], 0.25)
                row[f"{label}_{metric}_median"] = q(group[metric], 0.50)
                row[f"{label}_{metric}_p75"] = q(group[metric], 0.75)
        summaries.append(row)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT / "milestone_summary.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"confirmed-stable": "#167D9A", "not-confirmed-stable": "#D95F43"}
    years = np.asarray(MILESTONES)

    fig, ax = plt.subplots(figsize=(11.5, 6.5), dpi=180)
    stable_counts = summary_df.confirmed_stable_cells.to_numpy()
    unstable_counts = summary_df.not_confirmed_stable_cells.to_numpy()
    ax.bar(years, stable_counts, width=180, color=colors["confirmed-stable"], label="Confirmed numerically stable")
    ax.bar(years, unstable_counts, bottom=stable_counts, width=180, color=colors["not-confirmed-stable"], label="Not confirmed stable")
    for x, s, u in zip(years, stable_counts, unstable_counts):
        ax.text(x, s / 2, str(int(s)), ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        ax.text(x, s + u / 2, str(int(u)), ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.set(title="Numerical stability at registered spin-up milestones (1,337 parameter cells)", xlabel="Model year", ylabel="Parameter-cell count")
    ax.set_xticks(years)
    ax.legend(loc="upper right", frameon=True)
    ax.text(0.01, -0.16, "Proposed retrospective rule: volume ≤0.5% per 30 y and ≤1.5% per 90 y; length ≤300 m per 30 y and per 90 y; one 30-y confirmation. No physical-plausibility gate.", transform=ax.transAxes, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "milestone_numeric_stability_counts.png", bbox_inches="tight")
    plt.close(fig)

    metrics = [
        ("length_ratio_to_year0", "Length / year-0 length", False),
        ("maximum_thickness_m", "Maximum thickness (m)", False),
        ("volume_ratio_to_year0", "Volume / year-0 volume", True),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 12), dpi=180, sharex=True)
    for ax, (metric, ylabel, logy) in zip(axes, metrics):
        for group_name, label in [("confirmed-stable", "Confirmed numerically stable"), ("not-confirmed-stable", "Not confirmed stable")]:
            med, lo, hi = [], [], []
            for year in MILESTONES:
                g = cells[(cells.milestone_year == year) & (cells.numerical_group == group_name)][metric]
                med.append(q(g, 0.5)); lo.append(q(g, 0.25)); hi.append(q(g, 0.75))
            med = np.asarray(med); lo = np.asarray(lo); hi = np.asarray(hi)
            ax.plot(years, med, marker="o", linewidth=2.2, color=colors[group_name], label=label)
            ax.fill_between(years, lo, hi, color=colors[group_name], alpha=0.18)
        if logy:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.set_xticks(years)
    axes[0].set_title("Geometry and volume trajectories by milestone stability group\n(lines = median; bands = interquartile range)")
    axes[0].legend(loc="upper left", frameon=True)
    axes[-1].set_xlabel("Model year")
    axes[-1].text(0.01, -0.28, "Earlier confirmed equilibrium states are carried forward under the unchanged repeating climate. Stability here is numerical only and does not certify physically plausible geometry.", transform=axes[-1].transAxes, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "milestone_geometry_trajectories.png", bbox_inches="tight")
    plt.close(fig)

    first = (
        cells[["cell_ordinal", "lia_unique_id", "delta_t_eq_degC", "first_confirmed_boundary_year"]]
        .drop_duplicates("cell_ordinal")
        .sort_values(["lia_unique_id", "delta_t_eq_degC"])
    )
    pivot = first.pivot(index="lia_unique_id", columns="delta_t_eq_degC", values="first_confirmed_boundary_year")
    pivot.to_csv(OUT / "system_delta_first_confirmed_year.csv")

    summary = {
        "schema": "lia-191-milestone-balance-diagnostics-v1",
        "status": "complete-read-only-retrospective-analysis",
        "population_systems": 191,
        "parameter_cells": 1337,
        "milestones": MILESTONES,
        "source_result_status_counts": status_counts,
        "source_partition": {"v10_216_new_results": 1302, "v10_203_reused_converged_results": 35},
        "retrospective_proposed_numerical_rule": {
            "volume_each_30y_relative_max": VOL_CYCLE_MAX,
            "volume_90y_relative_max": VOL_90_MAX,
            "length_each_30y_max_m": LENGTH_CYCLE_MAX_M,
            "length_90y_max_m": LENGTH_90_MAX_M,
            "confirmation": "one additional 30-year same-phase rolling window",
            "current_execution_contract_modified": False,
        },
        "physical_plausibility_gate_applied": False,
        "equilibrium_label_scope": "numerical-stability-only-not-physical-acceptance",
        "early_terminal_equilibrium_states_carried_forward": True,
        "carried_cell_milestone_states": carried_count,
        "simulation_launched": False,
        "outputs": [
            "cell_milestone_states.csv",
            "milestone_summary.csv",
            "system_delta_first_confirmed_year.csv",
            "milestone_numeric_stability_counts.png",
            "milestone_geometry_trajectories.png",
        ],
    }
    with (OUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    hashes = {}
    for path in sorted(OUT.iterdir()):
        if path.name != "artifact_hashes.json" and path.is_file():
            hashes[path.name] = sha256(path)
    with (OUT / "artifact_hashes.json").open("w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)
        f.write("\n")

    print(summary_df.to_string(index=False))
    print(json.dumps({"output": str(OUT), "hashes": hashes}, indent=2))


if __name__ == "__main__":
    main()
