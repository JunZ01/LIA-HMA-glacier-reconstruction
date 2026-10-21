"""Decompose the masked 50-year source into forcing, support, and surface effects."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from netCDF4 import Dataset

from .torch_planform_2d_monthly_ti_bridge_v1 import TorchMonthlyTIBridge


HERE = Path(__file__).resolve()
ROOT = next(parent for parent in HERE.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
STATE = ROOT / "LIA_CURRENT_STATE.json"
RUN_ROOT = BASE / "lia_191_system2005_masked_steady50_checkpointed_v3"
RUN = RUN_ROOT / "summary.json"
GEOLOGY = BASE / "lia_191_igm_planform2d_system2005_static_v1/geology.nc"
CLIMATE = BASE / "lia_191_torch_planform2d_system2005_climate_bridge_export_v1/monthly_ti_package.npz"
PARITY = BASE / "lia_191_system2005_1d_2d_climate_area_budget_parity_v1/summary.json"
OUT = BASE / "lia_191_system2005_masked_internal_source_elevation_feedback_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def annual_mb_thickness(
    bridge: TorchMonthlyTIBridge, surface: torch.Tensor, year: int
) -> torch.Tensor:
    result = torch.zeros_like(surface)
    for month in range(year * 12, (year + 1) * 12):
        result += bridge.monthly_mb(surface, month) * bridge.sec_in_month
    return result


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    state_control = json.loads(STATE.read_text(encoding="utf-8"))
    if state_control["state_id"] != "2026-08-29-v10.143-system2005-masked-steady50-volume-fail":
        raise RuntimeError("unexpected-controller-state")
    run = json.loads(RUN.read_text(encoding="utf-8"))
    parity = json.loads(PARITY.read_text(encoding="utf-8"))
    if run["completed_years"] != 50.0 or run["status"] != "masked-steady50-fail-stop":
        raise RuntimeError("masked-run-not-terminal")
    if torch.__version__ != "2.11.0+cu128" or torch.version.cuda != "12.8":
        raise RuntimeError("runtime-version-mismatch")
    with Dataset(GEOLOGY) as dataset:
        bed_np = np.asarray(dataset.variables["topg"][:], dtype=np.float64)
        initial_np = np.asarray(dataset.variables["thk"][:], dtype=np.float64)
        mask_np = np.asarray(dataset.variables["lia_source_mask"][:], dtype=float) > 0.5
        dx = float(getattr(dataset, "dx_m"))
    cell_area = dx**2
    bridge = TorchMonthlyTIBridge(CLIMATE, device="cuda:0")
    if bridge.month_count != 600:
        raise RuntimeError("climate-month-count-not-600")
    bed = torch.as_tensor(bed_np, dtype=torch.float64, device="cuda:0")
    initial = torch.as_tensor(initial_np, dtype=torch.float64, device="cuda:0")
    initial_surface = bed + initial
    initial_active = initial > bridge.active_thickness_tolerance_m
    initial_volume = float(np.sum(initial_np)) * cell_area

    states = [initial_np]
    for year in range(1, 51):
        state_path = RUN_ROOT / f"checkpoints/year_{year:03d}/state.npz"
        with np.load(state_path, allow_pickle=False) as archive:
            states.append(np.asarray(archive["thickness_m"], dtype=np.float64))
    state_stack = np.stack(states)
    ever_active = np.any(state_stack > bridge.active_thickness_tolerance_m, axis=0)
    final_active = state_stack[-1] > bridge.active_thickness_tolerance_m
    cumulative_cell_full = np.zeros_like(initial_np)
    rows: list[dict] = []
    previous_actual = 0.0
    cumulative_initial = cumulative_support = cumulative_surface = cumulative_full = 0.0
    for year in range(50):
        start_np = state_stack[year]
        start = torch.as_tensor(start_np, dtype=torch.float64, device="cuda:0")
        start_surface = bed + start
        start_active = start > bridge.active_thickness_tolerance_m
        mb_initial = annual_mb_thickness(bridge, initial_surface, year)
        mb_start = annual_mb_thickness(bridge, start_surface, year)
        initial_source = float(torch.sum(mb_initial[initial_active]).item()) * cell_area
        support_source = float(torch.sum(mb_initial[start_active]).item()) * cell_area
        surface_source = float(torch.sum(mb_start[initial_active]).item()) * cell_area
        full_source = float(torch.sum(mb_start[start_active]).item()) * cell_area
        cumulative_actual = float(run["checkpoint_metrics"][str(year + 1)]["cumulative_source_m3"])
        actual_source = cumulative_actual - previous_actual
        previous_actual = cumulative_actual
        support_effect = support_source - initial_source
        surface_effect = surface_source - initial_source
        interaction = full_source - initial_source - support_effect - surface_effect
        intra_year = actual_source - full_source
        closure = actual_source - (
            initial_source + support_effect + surface_effect + interaction + intra_year
        )
        cumulative_initial += initial_source
        cumulative_support += support_effect
        cumulative_surface += surface_effect
        cumulative_full += full_source
        cell_full = mb_start.detach().cpu().numpy() * start_active.detach().cpu().numpy() * cell_area
        cumulative_cell_full += cell_full
        rows.append(
            {
                "year": year + 1,
                "actual_source_m3": actual_source,
                "initial_geometry_source_m3": initial_source,
                "support_effect_m3": support_effect,
                "surface_effect_on_initial_support_m3": surface_effect,
                "support_surface_interaction_m3": interaction,
                "intra_year_geometry_effect_m3": intra_year,
                "decomposition_closure_m3": closure,
                "start_active_area_m2": float(torch.count_nonzero(start_active).item()) * cell_area,
                "start_mean_surface_elevation_m": float(torch.mean(start_surface[start_active]).item()),
                "end_volume_change_fraction": (float(np.sum(state_stack[year + 1])) * cell_area - initial_volume) / initial_volume,
            }
        )

    support_classes = {
        "initially_active": initial_np > bridge.active_thickness_tolerance_m,
        "initially_inactive_ever_activated": (initial_np <= bridge.active_thickness_tolerance_m) & ever_active,
        "never_active": ~ever_active,
    }
    class_rows = []
    for name, selector in support_classes.items():
        class_rows.append(
            {
                "support_class": name,
                "cell_count": int(np.count_nonzero(selector)),
                "area_m2": float(np.count_nonzero(selector) * cell_area),
                "cumulative_frozen_start_source_m3": float(np.sum(cumulative_cell_full[selector])),
                "final_active_cell_count": int(np.count_nonzero(selector & final_active)),
                "mean_final_minus_initial_surface_m": float(np.mean(state_stack[-1][selector] - initial_np[selector])) if np.any(selector) else None,
            }
        )

    source_surface = bed_np + initial_np
    selected_elevation = source_surface[ever_active]
    lower = int(np.floor(np.nanmin(selected_elevation) / 250.0) * 250)
    upper = int(np.ceil(np.nanmax(selected_elevation) / 250.0) * 250)
    elevation_rows = []
    for low in range(lower, upper, 250):
        selector = ever_active & (source_surface >= low) & (source_surface < low + 250)
        if not np.any(selector):
            continue
        elevation_rows.append(
            {
                "initial_surface_elevation_low_m": low,
                "initial_surface_elevation_high_m": low + 250,
                "ever_active_area_m2": float(np.count_nonzero(selector) * cell_area),
                "cumulative_frozen_start_source_m3": float(np.sum(cumulative_cell_full[selector])),
                "mean_surface_change_m": float(np.mean(state_stack[-1][selector] - initial_np[selector])),
                "final_active_fraction": float(np.mean(final_active[selector])),
            }
        )

    actual_total = float(run["cumulative_climate_source_m3"])
    support_total = sum(row["support_effect_m3"] for row in rows)
    surface_total = sum(row["surface_effect_on_initial_support_m3"] for row in rows)
    interaction_total = sum(row["support_surface_interaction_m3"] for row in rows)
    intra_total = sum(row["intra_year_geometry_effect_m3"] for row in rows)
    decomposition_total = cumulative_initial + support_total + surface_total + interaction_total + intra_total
    prior_fixed = float(parity["two_dimensional_budget"]["fixed_initial_geometry_50y_source_m3"])
    checks = {
        "controller_state_exact": True,
        "all_51_states_loaded": len(states) == 51,
        "all_states_finite_nonnegative": bool(np.isfinite(state_stack).all() and np.min(state_stack) >= 0.0),
        "all_states_outside_domain_zero": bool(np.max(np.abs(state_stack[:, ~mask_np])) == 0.0),
        "initial_geometry_source_closes_prior_audit": abs(cumulative_initial - prior_fixed) <= max(1e-3, abs(prior_fixed) * 1e-10),
        "decomposition_closes_actual_source": abs(decomposition_total - actual_total) <= max(1e-3, abs(actual_total) * 1e-10),
        "annual_decomposition_closure": max(abs(row["decomposition_closure_m3"]) for row in rows) <= 1e-6,
        "source_hashes_unchanged": sha256(GEOLOGY) == run["checks"] and False,
    }
    # The run summary stores the immutability verdict rather than paths; verify
    # current hashes against its immutable config snapshot.
    config = json.loads((RUN_ROOT / "config_snapshot.json").read_text(encoding="utf-8"))
    checks["source_hashes_unchanged"] = (
        sha256(GEOLOGY) == config["geology_sha256"]
        and sha256(CLIMATE) == config["climate_package_sha256"]
    )
    status = "internal-source-decomposition-pass" if all(checks.values()) else "internal-source-decomposition-fail"

    OUT.mkdir(parents=False)
    write_csv(OUT / "annual_source_decomposition.csv", rows)
    write_csv(OUT / "support_class_source_decomposition.csv", class_rows)
    write_csv(OUT / "elevation_band_source_decomposition.csv", elevation_rows)
    years = np.asarray([row["year"] for row in rows])
    million = 1e6
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(years, np.asarray([row["actual_source_m3"] for row in rows]) / million, label="Actual")
    axes[0, 0].plot(years, np.asarray([row["initial_geometry_source_m3"] for row in rows]) / million, label="Initial geometry")
    axes[0, 0].plot(years, np.asarray([row["initial_geometry_source_m3"] + row["support_effect_m3"] + row["surface_effect_on_initial_support_m3"] + row["support_surface_interaction_m3"] for row in rows]) / million, label="Frozen annual-start geometry")
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 0].set_ylabel("Annual source (million m3)")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(years, 100 * np.asarray([row["end_volume_change_fraction"] for row in rows]), color="#b33b32")
    axes[0, 1].axhline(2, color="black", linestyle="--", linewidth=0.8)
    axes[0, 1].axhline(-2, color="black", linestyle="--", linewidth=0.8)
    axes[0, 1].set_ylabel("Volume change (%)")
    axes[1, 0].plot(years, np.asarray([row["start_active_area_m2"] for row in rows]) / 1e6, color="#2d6a4f")
    axes[1, 0].set_ylabel("Active area (km2)")
    labels = ["Initial climate", "Support", "Surface", "Interaction", "Within-year"]
    totals = [cumulative_initial, support_total, surface_total, interaction_total, intra_total]
    axes[1, 1].bar(labels, np.asarray(totals) / million, color=["#5b6770", "#4c78a8", "#e45756", "#72b7b2", "#f2cf5b"])
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_ylabel("50-year contribution (million m3)")
    axes[1, 1].tick_params(axis="x", rotation=25)
    for axis in axes.flat:
        axis.set_xlabel("Model year") if axis is not axes[1, 1] else None
        axis.grid(alpha=0.2)
    fig.savefig(OUT / "internal_source_elevation_feedback.png", dpi=180)
    plt.close(fig)

    summary = {
        "schema": "lia-191-system2005-masked-internal-source-elevation-feedback-v1",
        "status": status,
        "system_id": 2005,
        "actual_50y_source_m3": actual_total,
        "decomposition_m3": {
            "initial_geometry_climate": cumulative_initial,
            "active_support_effect": support_total,
            "surface_elevation_effect_on_initial_support": surface_total,
            "support_surface_interaction": interaction_total,
            "within_year_geometry_effect": intra_total,
            "sum": decomposition_total,
        },
        "fraction_of_actual_source": {
            "initial_geometry_climate": cumulative_initial / actual_total,
            "active_support_effect": support_total / actual_total,
            "surface_elevation_effect_on_initial_support": surface_total / actual_total,
            "support_surface_interaction": interaction_total / actual_total,
            "within_year_geometry_effect": intra_total / actual_total,
        },
        "support_classes": class_rows,
        "initial_active_area_m2": float(np.count_nonzero(initial_np > bridge.active_thickness_tolerance_m) * cell_area),
        "final_active_area_m2": float(np.count_nonzero(final_active) * cell_area),
        "checks": checks,
        "source_hashes": {"geology": sha256(GEOLOGY), "climate_package": sha256(CLIMATE), "run_summary": sha256(RUN)},
        "historical_run": False,
        "production_temperature_modified": False,
        "initial_thickness_modified": False,
        "next_action": "adjudicate which internal feedback term controls the residual before proposing a new mechanism",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
