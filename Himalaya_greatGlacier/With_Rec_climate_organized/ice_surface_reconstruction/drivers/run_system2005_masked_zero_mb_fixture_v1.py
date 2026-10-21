"""Run the authorized source-mask zero-MB single-step and 0.1-year fixture."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from netCDF4 import Dataset

from .torch_planform_2d_masked_frozen_diffusivity_core_v1 import masked_frozen_diffusivity_step
from .torch_planform_2d_system2005_adaptive_relaxation_60s_v1 import (
    ERROR_GATE,
    GLEN_N,
    GRAVITY_M_S2,
    ICE_DENSITY_KG_M3,
)


TARGET_SECONDS = 0.1 * 31_536_000.0
MINIMUM_DT = 0.001
MAXIMUM_DT = 86_400.0
INITIAL_DT = 0.03


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def valid(diag: dict[str, object], closure_m3: float, tolerance: float) -> bool:
    return (
        bool(diag["solver_converged"])
        and bool(diag["state_finite"])
        and int(diag["negative_cell_count"]) == 0
        and bool(diag["cross_domain_leakage_zero"])
        and abs(closure_m3) <= tolerance
    )


def run(config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(config["project_root"])
    geology = root / config["geology"]
    output = root / config["output_root"]
    route = root / config["expected_route"]
    if output.exists():
        raise FileExistsError(output)
    if sha256(route) != config["expected_route_sha256"].upper():
        raise RuntimeError("route-hash-mismatch")
    source_hash_before = sha256(geology)
    if source_hash_before != config["geology_sha256"].upper():
        raise RuntimeError("geology-hash-mismatch")
    with Dataset(geology) as dataset:
        bed_np = np.asarray(dataset.variables["topg"][:], dtype=float)
        thick_np = np.asarray(dataset.variables["thk"][:], dtype=float)
        mask_np = np.asarray(dataset.variables["lia_source_mask"][:], dtype=float) > 0.5
        dx = float(getattr(dataset, "dx_m"))
    if np.any(thick_np[~mask_np] != 0.0):
        raise RuntimeError("initial-ice-outside-domain")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("registered-sm120-gpu-unavailable")

    output.mkdir(parents=False)
    bed = torch.as_tensor(bed_np, dtype=torch.float64, device="cuda:0")
    initial = torch.as_tensor(thick_np, dtype=torch.float64, device="cuda:0")
    mask = torch.as_tensor(mask_np, dtype=torch.bool, device="cuda:0")
    state = initial.clone()
    cell_area = dx**2
    initial_volume = float(torch.sum(initial).item()) * cell_area
    tolerance = max(1e-6, initial_volume * 1e-12)
    glen_a = float(config["glen_a"])
    gamma = 2.0 * glen_a * (ICE_DENSITY_KG_M3 * GRAVITY_M_S2) ** GLEN_N / (GLEN_N + 2.0)
    elapsed = 0.0
    proposed = INITIAL_DT
    accepted = 0
    rejected = 0
    maximum_closure = 0.0
    maximum_error = 0.0
    history = []
    started = time.perf_counter()
    while elapsed < TARGET_SECONDS:
        dt = min(proposed, TARGET_SECONDS - elapsed)
        full, fd = masked_frozen_diffusivity_step(state, bed, mask, dx=dx, dy=dx, glen_n=GLEN_N, gamma=gamma, dt=dt)
        half, hd = masked_frozen_diffusivity_step(state, bed, mask, dx=dx, dy=dx, glen_n=GLEN_N, gamma=gamma, dt=0.5 * dt)
        half2, h2d = masked_frozen_diffusivity_step(half, bed, mask, dx=dx, dy=dx, glen_n=GLEN_N, gamma=gamma, dt=0.5 * dt)
        torch.cuda.synchronize()
        closure = max(
            abs(float(torch.sum(full - state).item()) * cell_area),
            abs(float(torch.sum(half - state).item()) * cell_area),
            abs(float(torch.sum(half2 - half).item()) * cell_area),
        )
        delta = half2 - state
        denominator = max(float(torch.linalg.vector_norm(delta).item()), 1e-30)
        error = float(torch.linalg.vector_norm(half2 - full).item()) / denominator
        numerical = valid(fd, float(torch.sum(full - state).item()) * cell_area, tolerance)
        numerical &= valid(hd, float(torch.sum(half - state).item()) * cell_area, tolerance)
        numerical &= valid(h2d, float(torch.sum(half2 - half).item()) * cell_area, tolerance)
        accept = numerical and error <= ERROR_GATE
        history.append({
            "trial": len(history) + 1, "accepted": accept, "elapsed_before_s": elapsed,
            "dt_s": dt, "error_relative_l2": error, "maximum_substep_closure_m3": closure,
            "maximum_change_m": float(torch.max(torch.abs(delta)).item()),
        })
        if accept:
            state = half2
            elapsed += dt
            accepted += 1
            maximum_closure = max(maximum_closure, closure)
            maximum_error = max(maximum_error, error)
            proposed = min(MAXIMUM_DT, 2.0 * dt) if error <= ERROR_GATE / 4.0 else dt
        else:
            rejected += 1
            proposed = 0.5 * dt
            if proposed < MINIMUM_DT:
                break
    torch.cuda.synchronize()
    final_volume = float(torch.sum(state).item()) * cell_area
    source_hash_after = sha256(geology)
    outside_max = float(torch.max(torch.abs(state[~mask])).item())
    checks = {
        "target_0p1y_completed": math.isclose(elapsed, TARGET_SECONDS, rel_tol=0.0, abs_tol=1e-7),
        "single_step_completed": accepted >= 1,
        "cumulative_mass_conservation": abs(final_volume - initial_volume) <= tolerance,
        "maximum_substep_mass_conservation": maximum_closure <= tolerance,
        "finite_state": bool(torch.isfinite(state).all()),
        "nonnegative_state": int(torch.count_nonzero(state < 0).item()) == 0,
        "cross_domain_leakage_zero": outside_max == 0.0,
        "initial_thickness_unchanged_before_run": True,
        "source_immutable": source_hash_before == source_hash_after,
        "adaptive_error_gate": maximum_error <= ERROR_GATE,
    }
    status = "masked-zero-mb-fixture-pass" if all(checks.values()) else "masked-zero-mb-fixture-fail-stop"
    state_np = state.detach().cpu().numpy()
    np.savez_compressed(output / "final_state_0p1y.npz", thickness_m=state_np, domain_mask=mask_np)
    with (output / "step_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "schema": "lia-191-system2005-masked-zero-mb-fixture-v1",
        "status": status,
        "system_id": 2005,
        "target_seconds": TARGET_SECONDS,
        "completed_seconds": elapsed,
        "accepted_steps": accepted,
        "rejected_steps": rejected,
        "wall_seconds": time.perf_counter() - started,
        "initial_volume_m3": initial_volume,
        "final_volume_m3": final_volume,
        "cumulative_volume_change_m3": final_volume - initial_volume,
        "mass_conservation_tolerance_m3": tolerance,
        "maximum_substep_closure_m3": maximum_closure,
        "maximum_accepted_error_relative_l2": maximum_error,
        "outside_domain_maximum_thickness_m": outside_max,
        "domain_cell_count": int(np.count_nonzero(mask_np)),
        "checks": checks,
        "source_geology_sha256_before": source_hash_before,
        "source_geology_sha256_after": source_hash_after,
        "mass_balance": "zero",
        "climate_run": False,
        "historical_run": False,
        "initial_thickness_modified": False,
        "production_temperature_modified": False,
        "next_action": "formal route adjudication before any masked climate preflight",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
