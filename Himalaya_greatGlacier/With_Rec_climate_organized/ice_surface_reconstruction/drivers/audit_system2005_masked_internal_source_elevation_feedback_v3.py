"""Numerical-only internal source decomposition; plotting is a separate process."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

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
OUT = BASE / "lia_191_system2005_masked_internal_source_elevation_feedback_v3"


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


def annual_mb(bridge: TorchMonthlyTIBridge, surface: torch.Tensor, year: int) -> torch.Tensor:
    result = torch.zeros_like(surface)
    for month in range(12 * year, 12 * (year + 1)):
        result += bridge.monthly_mb(surface, month) * bridge.sec_in_month
    return result


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    control = json.loads(STATE.read_text(encoding="utf-8"))
    if control["state_id"] != "2026-08-29-v10.143-system2005-masked-steady50-volume-fail":
        raise RuntimeError("unexpected-controller-state")
    run = json.loads(RUN.read_text(encoding="utf-8"))
    parity = json.loads(PARITY.read_text(encoding="utf-8"))
    if run["status"] != "masked-steady50-fail-stop" or run["completed_years"] != 50.0:
        raise RuntimeError("run-not-terminal")
    if torch.__version__ != "2.11.0+cu128" or torch.version.cuda != "12.8":
        raise RuntimeError("runtime-version-mismatch")
    with Dataset(GEOLOGY) as dataset:
        bed_np = np.asarray(dataset.variables["topg"][:], dtype=np.float64)
        initial_np = np.asarray(dataset.variables["thk"][:], dtype=np.float64)
        mask_np = np.asarray(dataset.variables["lia_source_mask"][:], dtype=float) > 0.5
        dx = float(getattr(dataset, "dx_m"))
    states = [initial_np]
    for year in range(1, 51):
        with np.load(RUN_ROOT / f"checkpoints/year_{year:03d}/state.npz", allow_pickle=False) as archive:
            states.append(np.asarray(archive["thickness_m"], dtype=np.float64))
    stack = np.stack(states)
    bridge = TorchMonthlyTIBridge(CLIMATE, device="cuda:0")
    if bridge.month_count != 600:
        raise RuntimeError("climate-month-count-not-600")
    cell_area = dx**2
    bed = torch.as_tensor(bed_np, dtype=torch.float64, device="cuda:0")
    initial = torch.as_tensor(initial_np, dtype=torch.float64, device="cuda:0")
    initial_surface = bed + initial
    initial_active = initial > bridge.active_thickness_tolerance_m
    initial_volume = float(np.sum(initial_np)) * cell_area
    ever_active = np.any(stack > bridge.active_thickness_tolerance_m, axis=0)
    final_active = stack[-1] > bridge.active_thickness_tolerance_m
    cumulative_cell_full = np.zeros_like(initial_np)
    rows = []
    prior_actual = 0.0
    for year in range(50):
        start = torch.as_tensor(stack[year], dtype=torch.float64, device="cuda:0")
        start_active = start > bridge.active_thickness_tolerance_m
        initial_mb = annual_mb(bridge, initial_surface, year)
        start_mb = annual_mb(bridge, bed + start, year)
        initial_source = float(torch.sum(initial_mb[initial_active]).item()) * cell_area
        support_source = float(torch.sum(initial_mb[start_active]).item()) * cell_area
        surface_source = float(torch.sum(start_mb[initial_active]).item()) * cell_area
        frozen_start_source = float(torch.sum(start_mb[start_active]).item()) * cell_area
        cumulative_actual = float(run["checkpoint_metrics"][str(year + 1)]["cumulative_source_m3"])
        actual_source = cumulative_actual - prior_actual
        prior_actual = cumulative_actual
        support_effect = support_source - initial_source
        surface_effect = surface_source - initial_source
        interaction = frozen_start_source - initial_source - support_effect - surface_effect
        within_year = actual_source - frozen_start_source
        closure = actual_source - (initial_source + support_effect + surface_effect + interaction + within_year)
        active_np = start_active.detach().cpu().numpy()
        cumulative_cell_full += start_mb.detach().cpu().numpy() * active_np * cell_area
        rows.append(
            {
                "year": year + 1,
                "actual_source_m3": actual_source,
                "initial_geometry_source_m3": initial_source,
                "active_support_effect_m3": support_effect,
                "surface_elevation_effect_m3": surface_effect,
                "support_surface_interaction_m3": interaction,
                "within_year_geometry_effect_m3": within_year,
                "closure_m3": closure,
                "start_active_area_m2": float(np.count_nonzero(active_np) * cell_area),
                "start_mean_surface_elevation_m": float(torch.mean((bed + start)[start_active]).item()),
                "end_volume_change_fraction": (float(np.sum(stack[year + 1])) * cell_area - initial_volume) / initial_volume,
            }
        )

    classes = {
        "initially_active": initial_np > bridge.active_thickness_tolerance_m,
        "initially_inactive_ever_activated": (initial_np <= bridge.active_thickness_tolerance_m) & ever_active,
        "never_active": ~ever_active,
    }
    class_rows = []
    for name, selector in classes.items():
        class_rows.append(
            {
                "support_class": name,
                "cell_count": int(np.count_nonzero(selector)),
                "area_m2": float(np.count_nonzero(selector) * cell_area),
                "cumulative_frozen_start_source_m3": float(np.sum(cumulative_cell_full[selector])),
                "final_active_cell_count": int(np.count_nonzero(selector & final_active)),
                "mean_final_minus_initial_surface_m": float(np.mean(stack[-1][selector] - initial_np[selector])) if np.any(selector) else None,
            }
        )
    initial_surface_np = bed_np + initial_np
    active_elevations = initial_surface_np[ever_active]
    low_edge = int(np.floor(np.min(active_elevations) / 250.0) * 250)
    high_edge = int(np.ceil(np.max(active_elevations) / 250.0) * 250)
    elevation_rows = []
    for low in range(low_edge, high_edge, 250):
        selector = ever_active & (initial_surface_np >= low) & (initial_surface_np < low + 250)
        if np.any(selector):
            elevation_rows.append(
                {
                    "initial_surface_elevation_low_m": low,
                    "initial_surface_elevation_high_m": low + 250,
                    "ever_active_area_m2": float(np.count_nonzero(selector) * cell_area),
                    "cumulative_frozen_start_source_m3": float(np.sum(cumulative_cell_full[selector])),
                    "mean_surface_change_m": float(np.mean(stack[-1][selector] - initial_np[selector])),
                    "final_active_fraction": float(np.mean(final_active[selector])),
                }
            )

    totals = {
        "initial_geometry_climate": sum(row["initial_geometry_source_m3"] for row in rows),
        "active_support_effect": sum(row["active_support_effect_m3"] for row in rows),
        "surface_elevation_effect_on_initial_support": sum(row["surface_elevation_effect_m3"] for row in rows),
        "support_surface_interaction": sum(row["support_surface_interaction_m3"] for row in rows),
        "within_year_geometry_effect": sum(row["within_year_geometry_effect_m3"] for row in rows),
    }
    actual_total = float(run["cumulative_climate_source_m3"])
    decomposed = sum(totals.values())
    prior_fixed = float(parity["two_d_mass_budget"]["fixed_initial_geometry_cumulative_source_m3"])
    config = json.loads((RUN_ROOT / "config_snapshot.json").read_text(encoding="utf-8"))
    checks = {
        "controller_state_exact": True,
        "all_51_states_loaded": len(states) == 51,
        "states_finite_nonnegative": bool(np.isfinite(stack).all() and np.min(stack) >= 0.0),
        "outside_domain_zero_all_states": bool(np.max(np.abs(stack[:, ~mask_np])) == 0.0),
        "initial_geometry_source_closes_prior_audit": abs(totals["initial_geometry_climate"] - prior_fixed) <= max(1e-3, abs(prior_fixed) * 1e-10),
        "decomposition_closes_actual_source": abs(decomposed - actual_total) <= max(1e-3, abs(actual_total) * 1e-10),
        "annual_closure": max(abs(row["closure_m3"]) for row in rows) <= 1e-6,
        "source_hashes_unchanged": sha256(GEOLOGY) == config["geology_sha256"] and sha256(CLIMATE) == config["climate_package_sha256"],
        "production_files_unchanged": True,
        "dynamics_not_run": True,
    }
    OUT.mkdir(parents=False)
    write_csv(OUT / "annual_source_decomposition.csv", rows)
    write_csv(OUT / "support_class_source_decomposition.csv", class_rows)
    write_csv(OUT / "elevation_band_source_decomposition.csv", elevation_rows)
    summary = {
        "schema": "lia-191-system2005-masked-internal-source-elevation-feedback-v3",
        "status": "internal-source-decomposition-pass" if all(checks.values()) else "internal-source-decomposition-fail",
        "system_id": 2005,
        "supersedes": "v2 OpenMP import interruption before analysis",
        "actual_50y_source_m3": actual_total,
        "decomposition_m3": {**totals, "sum": decomposed},
        "fraction_of_actual_source": {name: value / actual_total for name, value in totals.items()},
        "support_classes": class_rows,
        "initial_active_area_m2": float(np.count_nonzero(initial_np > bridge.active_thickness_tolerance_m) * cell_area),
        "final_active_area_m2": float(np.count_nonzero(final_active) * cell_area),
        "checks": checks,
        "source_hashes": {"geology": sha256(GEOLOGY), "climate_package": sha256(CLIMATE), "run_summary": sha256(RUN)},
        "historical_run": False,
        "production_temperature_modified": False,
        "initial_thickness_modified": False,
        "next_action": "plot and adjudicate the dominant internal feedback term",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
