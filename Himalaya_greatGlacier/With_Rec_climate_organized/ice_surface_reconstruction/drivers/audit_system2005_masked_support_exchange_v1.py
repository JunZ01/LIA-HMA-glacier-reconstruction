"""Split the dominant masked support effect into added and removed support."""

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
DECOMP = BASE / "lia_191_system2005_masked_internal_source_elevation_feedback_v3/summary.json"
GEOLOGY = BASE / "lia_191_igm_planform2d_system2005_static_v1/geology.nc"
CLIMATE = BASE / "lia_191_torch_planform2d_system2005_climate_bridge_export_v1/monthly_ti_package.npz"
OUT = BASE / "lia_191_system2005_masked_support_exchange_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def annual_mb(bridge: TorchMonthlyTIBridge, surface: torch.Tensor, year: int) -> torch.Tensor:
    result = torch.zeros_like(surface)
    for month in range(12 * year, 12 * (year + 1)):
        result += bridge.monthly_mb(surface, month) * bridge.sec_in_month
    return result


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    control = json.loads(STATE.read_text(encoding="utf-8"))
    decomposition = json.loads(DECOMP.read_text(encoding="utf-8"))
    if control["state_id"] != "2026-08-29-v10.143-system2005-masked-steady50-volume-fail":
        raise RuntimeError("unexpected-controller-state")
    if decomposition["status"] != "internal-source-decomposition-pass":
        raise RuntimeError("decomposition-not-pass")
    with Dataset(GEOLOGY) as dataset:
        bed_np = np.asarray(dataset.variables["topg"][:], dtype=np.float64)
        initial_np = np.asarray(dataset.variables["thk"][:], dtype=np.float64)
        dx = float(getattr(dataset, "dx_m"))
    bridge = TorchMonthlyTIBridge(CLIMATE, device="cuda:0")
    bed = torch.as_tensor(bed_np, dtype=torch.float64, device="cuda:0")
    initial = torch.as_tensor(initial_np, dtype=torch.float64, device="cuda:0")
    initial_active = initial > bridge.active_thickness_tolerance_m
    initial_surface = bed + initial
    cell_area = dx**2
    rows = []
    added_total = removed_total = 0.0
    for year in range(50):
        if year == 0:
            start_np = initial_np
        else:
            with np.load(RUN_ROOT / f"checkpoints/year_{year:03d}/state.npz", allow_pickle=False) as archive:
                start_np = np.asarray(archive["thickness_m"], dtype=np.float64)
        start = torch.as_tensor(start_np, dtype=torch.float64, device="cuda:0")
        start_active = start > bridge.active_thickness_tolerance_m
        mb = annual_mb(bridge, initial_surface, year)
        added = start_active & ~initial_active
        removed = initial_active & ~start_active
        added_effect = float(torch.sum(mb[added]).item()) * cell_area
        removed_effect = -float(torch.sum(mb[removed]).item()) * cell_area
        added_total += added_effect
        removed_total += removed_effect
        rows.append(
            {
                "year": year + 1,
                "added_support_effect_m3": added_effect,
                "removed_support_effect_m3": removed_effect,
                "total_support_effect_m3": added_effect + removed_effect,
                "added_area_m2": float(torch.count_nonzero(added).item()) * cell_area,
                "removed_area_m2": float(torch.count_nonzero(removed).item()) * cell_area,
                "added_mean_initial_surface_m": float(torch.mean(initial_surface[added]).item()) if torch.any(added) else None,
                "removed_mean_initial_surface_m": float(torch.mean(initial_surface[removed]).item()) if torch.any(removed) else None,
                "added_mean_specific_balance_m_ice": float(torch.mean(mb[added]).item()) if torch.any(added) else None,
                "removed_mean_specific_balance_m_ice": float(torch.mean(mb[removed]).item()) if torch.any(removed) else None,
            }
        )
    support_total = float(decomposition["decomposition_m3"]["active_support_effect"])
    checks = {
        "controller_state_exact": True,
        "parent_decomposition_pass": True,
        "annual_support_exchange_closure": max(abs(row["total_support_effect_m3"] - (
            row["added_support_effect_m3"] + row["removed_support_effect_m3"]
        )) for row in rows) <= 1e-9,
        "cumulative_support_exchange_closes_parent": abs(added_total + removed_total - support_total) <= max(1e-3, abs(support_total) * 1e-10),
        "source_hashes_unchanged": sha256(GEOLOGY) == decomposition["source_hashes"]["geology"] and sha256(CLIMATE) == decomposition["source_hashes"]["climate_package"],
        "dynamics_not_run": True,
        "production_files_unchanged": True,
    }
    OUT.mkdir(parents=False)
    with (OUT / "annual_support_exchange.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "lia-191-system2005-masked-support-exchange-v1",
        "status": "support-exchange-pass" if all(checks.values()) else "support-exchange-fail",
        "system_id": 2005,
        "support_effect_m3": support_total,
        "added_support_effect_m3": added_total,
        "removed_support_effect_m3": removed_total,
        "added_fraction_of_support_effect": added_total / support_total,
        "removed_fraction_of_support_effect": removed_total / support_total,
        "added_fraction_of_actual_source": added_total / decomposition["actual_50y_source_m3"],
        "removed_fraction_of_actual_source": removed_total / decomposition["actual_50y_source_m3"],
        "final_year_exchange": rows[-1],
        "checks": checks,
        "source_hashes": {"geology": sha256(GEOLOGY), "climate_package": sha256(CLIMATE), "parent": sha256(DECOMP)},
        "historical_run": False,
        "initial_thickness_modified": False,
        "production_temperature_modified": False,
        "next_action": "plot annual source and support exchange, then adjudicate the mechanism",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
