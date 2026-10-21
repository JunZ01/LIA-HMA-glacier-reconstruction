"""Frozen OGGM CPU oracle for lia-1d-gpu-batch-preflight-v2."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import pickle
from pathlib import Path
import sys
import tempfile
import time

import numpy as np


DRIVERS = Path(__file__).resolve().parent
sys.path.insert(0, str(DRIVERS))
from lia_1d_gpu_preflight_guard_v2 import OUTPUT_ROOT, PROJECT_ROOT, sha256, validate_launch_authorization


def write_json_atomic(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def write_npz_atomic(path: Path, **arrays) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def snapshot(model) -> dict:
    return {
        "year": float(model.yr),
        "length_m": float(model.length_m),
        "area_m2": float(model.area_m2),
        "volume_m3": float(model.volume_m3),
        "accepted_steps": int(model.accepted_steps),
        "rejected_steps": int(model.rejected_steps),
        "minimum_actual_timestep_seconds": float(model.minimum_actual_timestep_seconds),
        "cumulative_climatic_mass_balance_m3": float(model.cumulative_climatic_mass_balance_m3),
        "cumulative_mass_budget_residual_m3": float(model.cumulative_mass_budget_residual_m3),
    }


def main() -> int:
    authorization = validate_launch_authorization(output_must_be_absent=True)
    from oggm import cfg, utils
    from oggm.core.flowline import SemiImplicitModel
    from oggm.core.massbalance import MonthlyTIModel

    cfg.initialize(logging_level="CRITICAL")
    cfg.PARAMS["use_kcalving_for_run"] = False
    cfg.PARAMS["error_when_glacier_reaches_boundaries"] = True

    class TrackedSemiImplicitModel(SemiImplicitModel):
        def __init__(self, *args, accepted_steps=0, rejected_steps=0,
                     minimum_actual_timestep_seconds=float("inf"),
                     cumulative_climatic_mass_balance_m3=0.0,
                     cumulative_mass_budget_residual_m3=0.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.accepted_steps = accepted_steps
            self.rejected_steps = rejected_steps
            self.minimum_actual_timestep_seconds = minimum_actual_timestep_seconds
            self.cumulative_climatic_mass_balance_m3 = cumulative_climatic_mass_balance_m3
            self.cumulative_mass_budget_residual_m3 = cumulative_mass_budget_residual_m3

        def step(self, dt):
            fl = self.fls[0]
            before = float(self.volume_m3)
            smb = self.get_mb(fl.surface_h, self.yr, fl_id=0, fls=self.fls).copy()
            used = float(super().step(dt))
            source = float(np.sum(smb * fl.widths_m) * fl.dx_meter * used)
            residual = float(self.volume_m3) - before - source
            self.accepted_steps += 1
            self.minimum_actual_timestep_seconds = min(self.minimum_actual_timestep_seconds, used)
            self.cumulative_climatic_mass_balance_m3 += source
            self.cumulative_mass_budget_residual_m3 += residual
            return used

    fixture = authorization["fixture"]
    lanes = fixture["lanes"]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    cpu_root = OUTPUT_ROOT / "cpu"
    cpu_root.mkdir()
    run_started = time.perf_counter()
    records = []
    thickness_by_stage = {name: [] for name in (
        "after_step", "after_month", "after_year", "at_checkpoint",
        "uninterrupted_year10", "resumed_year10",
    )}
    numerical_seconds = 0.0

    for lane in lanes:
        lane_started = time.perf_counter()
        sid = int(lane["system_id"])
        point = int(lane["point"])
        cell = f"{sid}-{point:04d}"
        cell_root = cpu_root / cell
        checkpoint_root = cell_root / "checkpoints/year-0005"
        checkpoint_root.mkdir(parents=True)
        with gzip.open(lane["flowline_path"], "rb") as handle:
            source_flowline = pickle.load(handle)[0]
        climate_path = Path(lane["climate_path"])
        base_dir = climate_path.parents[3]
        gdir = utils.GlacierDirectory(lane["rgi_id"], base_dir=str(base_dir))
        suffix = climate_path.stem[len("gcm_data"):]
        params = lane["parameters"]
        cfg.PARAMS["temp_melt"] = float(params["temp_melt"])

        def new_mb():
            return MonthlyTIModel(
                gdir, filename="gcm_data", input_filesuffix=suffix,
                ys=1836, ye=2000, repeat=False,
                melt_f=float(params["melt_f"]), prcp_fac=float(params["prcp_fac"]),
                temp_bias=float(params["delta_t"]), check_calib_params=False,
            )

        model = TrackedSemiImplicitModel(
            [copy.deepcopy(source_flowline)], mb_model=new_mb(), y0=1836.0,
            glen_a=float(params["glen_a"]), fs=0.0, inplace=False,
            check_for_boundaries=True, fixed_dt=1.25 * cfg.SEC_IN_DAY,
            required_model_steps="monthly",
        )
        initial_volume = float(model.volume_m3)
        started = time.perf_counter()
        model.step(1.25 * cfg.SEC_IN_DAY)
        after_step = snapshot(model)
        thickness_by_stage["after_step"].append(np.asarray(model.fls[0].thick, dtype=np.float64).copy())
        model.run_until(1836.0 + 1.0 / 12.0)
        after_month = snapshot(model)
        thickness_by_stage["after_month"].append(np.asarray(model.fls[0].thick, dtype=np.float64).copy())
        model.run_until(1837.0)
        after_year = snapshot(model)
        thickness_by_stage["after_year"].append(np.asarray(model.fls[0].thick, dtype=np.float64).copy())
        model.run_until(1841.0)
        at_checkpoint = snapshot(model)
        thickness_by_stage["at_checkpoint"].append(np.asarray(model.fls[0].thick, dtype=np.float64).copy())
        checkpoint_path = checkpoint_root / "state.npz"
        checkpoint_arrays = {
            "thick": np.asarray(model.fls[0].thick, dtype=np.float64),
            "model_year": np.array([model.yr], dtype=np.float64),
            "accepted_steps": np.array([model.accepted_steps], dtype=np.int64),
            "rejected_steps": np.array([model.rejected_steps], dtype=np.int64),
            "minimum_actual_timestep_seconds": np.array([model.minimum_actual_timestep_seconds], dtype=np.float64),
            "cumulative_climatic_mass_balance_m3": np.array([model.cumulative_climatic_mass_balance_m3], dtype=np.float64),
            "cumulative_mass_budget_residual_m3": np.array([model.cumulative_mass_budget_residual_m3], dtype=np.float64),
        }
        write_npz_atomic(checkpoint_path, **checkpoint_arrays)
        checkpoint_manifest = {
            "schema": "lia-1d-gpu-preflight-cpu-checkpoint-v1",
            "transaction_fingerprint_sha256": authorization["transaction"]["transaction_fingerprint_sha256"],
            "cell": lane["cell"],
            "flowline_sha256": lane["flowline_sha256"],
            "climate_sha256": lane["climate_sha256"],
            "state_sha256": sha256(checkpoint_path),
            "model_year": 1841.0,
        }
        write_json_atomic(checkpoint_root / "manifest.json", checkpoint_manifest)
        model.run_until(1846.0)
        uninterrupted = snapshot(model)
        numerical_seconds += time.perf_counter() - started
        thickness_by_stage["uninterrupted_year10"].append(np.asarray(model.fls[0].thick, dtype=np.float64).copy())

        with np.load(checkpoint_path) as checkpoint:
            restored_source = copy.deepcopy(source_flowline)
            restored_source.thick = checkpoint["thick"].copy()
            restore_bitwise = np.array_equal(restored_source.thick, checkpoint["thick"])
            restored = TrackedSemiImplicitModel(
                [restored_source], mb_model=new_mb(), y0=float(checkpoint["model_year"][0]),
                glen_a=float(params["glen_a"]), fs=0.0, inplace=False,
                check_for_boundaries=True, fixed_dt=1.25 * cfg.SEC_IN_DAY,
                required_model_steps="monthly",
                accepted_steps=int(checkpoint["accepted_steps"][0]),
                rejected_steps=int(checkpoint["rejected_steps"][0]),
                minimum_actual_timestep_seconds=float(checkpoint["minimum_actual_timestep_seconds"][0]),
                cumulative_climatic_mass_balance_m3=float(checkpoint["cumulative_climatic_mass_balance_m3"][0]),
                cumulative_mass_budget_residual_m3=float(checkpoint["cumulative_mass_budget_residual_m3"][0]),
            )
        restored.run_until(1846.0)
        resumed = snapshot(restored)
        thickness_by_stage["resumed_year10"].append(np.asarray(restored.fls[0].thick, dtype=np.float64).copy())
        record = {
            "lane": int(lane["lane"]), "cell": lane["cell"],
            "initial_volume_m3": initial_volume,
            "after_step": after_step, "after_month": after_month,
            "after_year": after_year, "at_checkpoint": at_checkpoint,
            "uninterrupted_year10": uninterrupted, "resumed_year10": resumed,
            "checkpoint_restore_bitwise": bool(restore_bitwise),
            "checkpoint_state_sha256": checkpoint_manifest["state_sha256"],
            "wall_seconds": time.perf_counter() - lane_started,
        }
        records.append(record)
        write_json_atomic(cell_root / "summary.json", record)

    max_nx = int(fixture["max_nx"])
    padded_stages = {}
    for stage, lane_values in thickness_by_stage.items():
        padded = np.zeros((24, max_nx), dtype=np.float64)
        for lane_index, values in enumerate(lane_values):
            padded[lane_index, : len(values)] = values
        padded_stages[f"{stage}_thick"] = padded
    oracle_npz = cpu_root / "oracle_states.npz"
    write_npz_atomic(oracle_npz, **padded_stages)
    wall_seconds = time.perf_counter() - run_started
    summary = {
        "schema": "lia-1d-gpu-batch-preflight-cpu-oracle-v2",
        "status": "complete",
        "transaction_id": "lia-1d-gpu-batch-preflight-v2",
        "transaction_fingerprint_sha256": authorization["transaction"]["transaction_fingerprint_sha256"],
        "controller_state_sha256": authorization["state_sha256"],
        "route_sha256": authorization["route_sha256"],
        "python_executable": sys.executable,
        "process_id": os.getpid(),
        "cell_count": 24,
        "wall_seconds": wall_seconds,
        "numerical_seconds": numerical_seconds,
        "completed_model_years_per_wall_hour": 24.0 * 10.0 * 3600.0 / numerical_seconds,
        "oracle_states_path": str(oracle_npz.relative_to(PROJECT_ROOT)),
        "oracle_states_sha256": sha256(oracle_npz),
        "records": records,
    }
    write_json_atomic(cpu_root / "summary.json", summary)
    print(json.dumps({"status": "complete", "wall_seconds": wall_seconds,
                      "model_years_per_hour": summary["completed_model_years_per_wall_hour"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
