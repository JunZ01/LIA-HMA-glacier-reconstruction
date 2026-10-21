"""One-year source-mask MonthlyTI climate preflight for system 2005."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from netCDF4 import Dataset

from .torch_planform_2d_masked_frozen_diffusivity_core_v1 import masked_frozen_diffusivity_step
from .torch_planform_2d_monthly_ti_bridge_v1 import TorchMonthlyTIBridge
from .torch_planform_2d_system2005_adaptive_relaxation_60s_v1 import (
    ERROR_GATE,
    GLEN_N,
    GRAVITY_M_S2,
    ICE_DENSITY_KG_M3,
)
from .torch_planform_2d_system2005_steady50_monthly_ti_formal_v2 import (
    _conservative_thin_cell_projection,
)


TARGET_SECONDS = 31_536_000.0
INITIAL_DT = 0.03
MINIMUM_DT = 0.001
MAXIMUM_DT = 86_400.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def masked_safe_diffusion(state, bed, mask, *, dx, gamma, dt):
    candidate, diagnostic = masked_frozen_diffusivity_step(
        state, bed, mask, dx=dx, dy=dx, glen_n=GLEN_N, gamma=gamma, dt=dt
    )
    projection = None
    if bool(diagnostic["solver_converged"]) and int(diagnostic["negative_cell_count"]) > 0:
        result = _conservative_thin_cell_projection(state, candidate)
        if result is not None:
            candidate, projection = result
            diagnostic = dict(diagnostic)
            diagnostic["negative_cell_count"] = int(torch.count_nonzero(candidate < 0).item())
            diagnostic["minimum_thickness_m"] = float(torch.min(candidate).item())
            diagnostic["state_finite"] = bool(torch.isfinite(candidate).all())
            diagnostic["cross_domain_leakage_zero"] = float(torch.max(torch.abs(candidate[~mask])).item()) == 0.0
    return candidate, diagnostic, projection


def split_step(state, bed, mask, bridge, *, month, dx, gamma, dt, cell_area):
    source1, s1 = bridge.apply_connected_ice_source(state, bed, month_index=month, dt_seconds=0.5 * dt)
    diffused, dd, projection = masked_safe_diffusion(source1, bed, mask, dx=dx, gamma=gamma, dt=dt)
    if bool(dd["solver_converged"]) and bool(dd["state_finite"]) and int(dd["negative_cell_count"]) == 0:
        result, s2 = bridge.apply_connected_ice_source(diffused, bed, month_index=month, dt_seconds=0.5 * dt)
    else:
        result = diffused
        s2 = {"accepted_thickness_change_sum_m": 0.0, "prevented_overmelt_sum_m": 0.0, "state_finite": False, "negative_cell_count": 1}
    source_volume = cell_area * (
        float(s1["accepted_thickness_change_sum_m"]) + float(s2["accepted_thickness_change_sum_m"])
    )
    diffusion_closure = float(torch.sum(diffused - source1).item()) * cell_area
    return result, {
        "numerical_pass": bool(dd["solver_converged"]) and bool(dd["state_finite"])
        and int(dd["negative_cell_count"]) == 0 and bool(dd["cross_domain_leakage_zero"])
        and bool(s1["state_finite"]) and bool(s2["state_finite"]),
        "diffusion_closure_m3": diffusion_closure,
        "source_volume_m3": source_volume,
        "projection_applied": projection is not None,
        "outside_maximum_m": float(torch.max(torch.abs(result[~mask])).item()),
    }


def run(config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(config["project_root"])
    geology = root / config["geology"]
    package = root / config["climate_package"]
    route = root / config["expected_route"]
    output = root / config["output_root"]
    if output.exists():
        raise FileExistsError(output)
    if sha256(route) != config["expected_route_sha256"].upper():
        raise RuntimeError("route-hash-mismatch")
    if sha256(geology) != config["geology_sha256"].upper() or sha256(package) != config["climate_package_sha256"].upper():
        raise RuntimeError("scientific-input-hash-mismatch")
    if torch.__version__ != "2.11.0+cu128" or torch.version.cuda != "12.8":
        raise RuntimeError("runtime-version-mismatch")
    if tuple(torch.cuda.get_device_capability(0)) != (12, 0):
        raise RuntimeError("gpu-capability-mismatch")
    with Dataset(geology) as dataset:
        bed_np = np.asarray(dataset.variables["topg"][:], dtype=float)
        thick_np = np.asarray(dataset.variables["thk"][:], dtype=float)
        mask_np = np.asarray(dataset.variables["lia_source_mask"][:], dtype=float) > 0.5
        dx = float(getattr(dataset, "dx_m"))
    output.mkdir(parents=False)
    bed = torch.as_tensor(bed_np, dtype=torch.float64, device="cuda:0")
    initial = torch.as_tensor(thick_np, dtype=torch.float64, device="cuda:0")
    state = initial.clone()
    mask = torch.as_tensor(mask_np, dtype=torch.bool, device="cuda:0")
    bridge = TorchMonthlyTIBridge(package, device="cuda:0")
    cell_area = dx**2
    initial_volume = float(torch.sum(initial).item()) * cell_area
    tolerance = max(1e-6, initial_volume * 1e-10)
    gamma = 2.0 * float(config["glen_a"]) * (ICE_DENSITY_KG_M3 * GRAVITY_M_S2) ** GLEN_N / (GLEN_N + 2.0)

    fixed_source = 0.0
    initial_surface = bed + initial
    active = initial > bridge.active_thickness_tolerance_m
    for month in range(12):
        fixed_source += float(torch.sum(bridge.monthly_mb(initial_surface, month)[active]).item()) * cell_area * bridge.sec_in_month

    elapsed = 0.0
    proposed = INITIAL_DT
    accepted = rejected = projections = 0
    cumulative_source = 0.0
    maximum_closure = maximum_error = outside_max = 0.0
    history = []
    started = time.perf_counter()
    while elapsed < TARGET_SECONDS:
        month = min(int(math.floor(elapsed / bridge.sec_in_month + 1e-12)), 11)
        month_remaining = (month + 1) * bridge.sec_in_month - elapsed
        dt = min(proposed, month_remaining, TARGET_SECONDS - elapsed)
        full, fd = split_step(state, bed, mask, bridge, month=month, dx=dx, gamma=gamma, dt=dt, cell_area=cell_area)
        half, hd = split_step(state, bed, mask, bridge, month=month, dx=dx, gamma=gamma, dt=0.5 * dt, cell_area=cell_area)
        half2, h2d = split_step(half, bed, mask, bridge, month=month, dx=dx, gamma=gamma, dt=0.5 * dt, cell_area=cell_area)
        torch.cuda.synchronize()
        delta = half2 - state
        denominator = max(float(torch.linalg.vector_norm(delta).item()), 1e-30)
        error = float(torch.linalg.vector_norm(half2 - full).item()) / denominator
        closure = max(abs(float(hd["diffusion_closure_m3"])), abs(float(h2d["diffusion_closure_m3"])))
        source_volume = float(hd["source_volume_m3"]) + float(h2d["source_volume_m3"])
        candidate_delta = float(torch.sum(delta).item()) * cell_area
        budget_residual = candidate_delta - source_volume
        numerical = bool(fd["numerical_pass"]) and bool(hd["numerical_pass"]) and bool(h2d["numerical_pass"])
        numerical &= closure <= tolerance and abs(budget_residual) <= tolerance
        accept = numerical and error <= ERROR_GATE
        history.append({
            "trial": len(history) + 1, "accepted": accept, "elapsed_before_years": elapsed / bridge.sec_in_year,
            "month": month + 1, "dt_seconds": dt, "error_relative_l2": error,
            "diffusion_closure_m3": closure, "budget_residual_m3": budget_residual,
            "source_volume_m3": source_volume,
        })
        if accept:
            state = half2
            elapsed += dt
            accepted += 1
            cumulative_source += source_volume
            projections += int(hd["projection_applied"]) + int(h2d["projection_applied"])
            maximum_closure = max(maximum_closure, closure, abs(budget_residual))
            maximum_error = max(maximum_error, error)
            outside_max = max(outside_max, float(h2d["outside_maximum_m"]))
            proposed = min(MAXIMUM_DT, 2.0 * dt) if error <= ERROR_GATE / 4.0 else dt
        else:
            rejected += 1
            proposed = 0.5 * dt
            if proposed < MINIMUM_DT:
                break
    final_volume = float(torch.sum(state).item()) * cell_area
    cumulative_budget = final_volume - initial_volume - cumulative_source
    final_active = state > bridge.active_thickness_tolerance_m
    checks = {
        "target_1y_completed": math.isclose(elapsed, TARGET_SECONDS, rel_tol=0.0, abs_tol=1e-7),
        "mass_budget_closure": abs(cumulative_budget) <= tolerance,
        "maximum_step_closure": maximum_closure <= tolerance,
        "finite_nonnegative": bool(torch.isfinite(state).all()) and int(torch.count_nonzero(state < 0).item()) == 0,
        "outside_domain_zero": outside_max == 0.0 and int(torch.count_nonzero(final_active & ~mask).item()) == 0,
        "fixed_initial_source_negative": fixed_source < 0.0,
        "actual_source_negative": cumulative_source < 0.0,
        "source_direction_matches": (fixed_source < 0.0) == (cumulative_source < 0.0),
        "source_hashes_unchanged": sha256(geology) == config["geology_sha256"].upper() and sha256(package) == config["climate_package_sha256"].upper(),
        "runtime_exact": torch.__version__ == "2.11.0+cu128" and torch.version.cuda == "12.8",
    }
    status = "masked-climate-1y-preflight-pass" if all(checks.values()) else "masked-climate-1y-preflight-fail-stop"
    state_np = state.detach().cpu().numpy()
    np.savez_compressed(output / "final_state_1y.npz", thickness_m=state_np, domain_mask=mask_np)
    with (output / "step_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    summary = {
        "schema": "lia-191-system2005-masked-climate-1y-preflight-v1",
        "status": status, "system_id": 2005, "completed_years": elapsed / bridge.sec_in_year,
        "accepted_steps": accepted, "rejected_steps": rejected, "projection_calls_accepted_path": projections,
        "wall_seconds": time.perf_counter() - started,
        "initial_volume_m3": initial_volume, "final_volume_m3": final_volume,
        "cumulative_climate_source_m3": cumulative_source, "fixed_initial_geometry_source_m3": fixed_source,
        "source_ratio_actual_to_fixed": cumulative_source / fixed_source,
        "cumulative_mass_budget_residual_m3": cumulative_budget,
        "maximum_step_or_budget_closure_m3": maximum_closure,
        "maximum_accepted_error_relative_l2": maximum_error,
        "outside_domain_maximum_thickness_m": outside_max,
        "initial_active_area_m2": float(torch.count_nonzero(initial > bridge.active_thickness_tolerance_m).item()) * cell_area,
        "final_active_area_m2": float(torch.count_nonzero(final_active).item()) * cell_area,
        "checks": checks,
        "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
        "climate_run": True, "historical_run": False, "initial_thickness_modified": False,
        "production_temperature_modified": False,
        "next_action": "formal adjudication before any longer masked steady run",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
