"""Rerun only the 200-year shooting phase at 50-year checkpoints."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import gzip
import hashlib
import json
import math
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import run_cycle4_state_shooting_3850_timing_v1 as core
from .lia_elevation_band_spinup import _clone_gdir, _configure_oggm, _gdir_path, _read_flowline
from .lia_first10_spinup_batch import _write_case_climate


REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE = "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.run_cycle4_last50y_shooting_diagnostic_v1"


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(core._json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_single_flowline(path: Path):
    with gzip.open(path, "rb") as stream:
        flowlines = pickle.load(stream)
    if len(flowlines) != 1:
        raise RuntimeError(f"expected-single-flowline:{len(flowlines)}")
    return flowlines[0]


def _geometry_exact(left, right) -> dict[str, Any]:
    checks = {
        name: core._arrays_equal(getattr(left, name), getattr(right, name))
        for name in ("bed_h", "widths_m", "surface_h", "thick", "section")
    }
    checks.update({"nx": left.nx == right.nx, "dx": left.dx == right.dx, "map_dx": left.map_dx == right.map_dx})
    return {"passed": all(checks.values()), "checks": checks}


def run_single(member_config_path: Path, output_dir: Path, workdir: Path, checkpoint_years: list[int], tolerance: float) -> dict[str, Any]:
    started = time.perf_counter()
    member_config = json.loads(member_config_path.read_text(encoding="utf-8"))
    source_member_dir = Path(member_config["paths"]["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refuse-nonempty-versioned-output:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    original_zero_path = source_member_dir / "zero_mb_summary.json"
    original_zero = json.loads(original_zero_path.read_text(encoding="utf-8"))
    if not bool(original_zero.get("passed")):
        raise RuntimeError("source-member-zero-mb-did-not-pass")

    baseline_path = Path(member_config["paths"]["elevation_band_root"]) / str(member_config["system_id"]) / "elevation_band_flowline_lia.pkl.gz"
    audit_path = baseline_path.parent / "elevation_band_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    baseline = _read_flowline(baseline_path)
    candidate, construction = core.build_candidate_geometry(
        baseline,
        member_config["q_v"],
        member_config["z_shape"],
        q_v_min=member_config["geometry"]["q_v_min"],
        q_v_max=member_config["geometry"]["q_v_max"],
        z_shape_min=member_config["geometry"]["z_shape_min"],
        z_shape_max=member_config["geometry"]["z_shape_max"],
        active_thickness_epsilon_m=member_config["geometry"]["active_thickness_epsilon_m"],
    )
    original_candidate_path = source_member_dir / "candidate_flowline_lia.pkl.gz"
    original_candidate = _load_single_flowline(original_candidate_path)
    reproduction = _geometry_exact(candidate, original_candidate)
    if not reproduction["passed"]:
        raise RuntimeError("reconstructed-candidate-does-not-match-original-member")

    oggm_version = _configure_oggm(workdir, output_dir)
    from oggm import cfg, workflow
    from oggm.core.flowline import SemiImplicitModel
    from oggm.core.massbalance import MonthlyTIModel

    cfg.PARAMS["temp_melt"] = float(member_config["temp_melt_c"])
    source_gdir = _gdir_path(Path(member_config["paths"]["source_gdir_root"]), member_config["rgi_id"])
    target_gdir = _gdir_path(workdir / "per_glacier", member_config["rgi_id"])
    _clone_gdir(source_gdir, target_gdir)
    gdir = workflow.init_glacier_directories([member_config["rgi_id"]], reset=False)[0]

    pixels = pd.read_csv(member_config["paths"]["climate_pixels"])
    pixel_rows = pixels[
        pixels["rgi_id"].astype(str).eq(member_config["rgi_id"])
        & pixels["climate_id"].astype(str).eq(member_config["climate_id"])
    ]
    if len(pixel_rows) != 1:
        raise RuntimeError(f"climate-pixel-row-count:{len(pixel_rows)}")
    climate_suffix, climate_audit = _write_case_climate(
        gdir,
        Path(member_config["paths"]["reconstruction_climate"]),
        pixel_rows.iloc[0],
        member_config["prcp_mode"],
        start_year=int(member_config["shooting_window_start"]),
        end_year=int(member_config["shooting_window_end"]),
    )
    physics = member_config["parent_physics"]
    mb_model = MonthlyTIModel(
        gdir,
        filename="gcm_data",
        input_filesuffix=climate_suffix,
        melt_f=float(physics["melt_f"]),
        prcp_fac=float(physics["prcp_fac"]),
        temp_bias=float(physics["temp_bias"]) + float(member_config["delta_t_c"]),
        ys=int(member_config["shooting_window_start"]),
        ye=int(member_config["shooting_window_end"]),
        repeat=True,
        check_calib_params=False,
    )
    model = SemiImplicitModel(
        [copy.deepcopy(candidate)],
        mb_model=mb_model,
        y0=0.0,
        glen_a=float(physics["glen_a"]),
        fs=float(physics["fs"]),
        inplace=False,
        check_for_boundaries=True,
        fixed_dt=float(member_config["fixed_dt_days"]) * float(cfg.SEC_IN_DAY),
    )
    initial_exact = core._model_initial_exact(model, candidate)
    if not initial_exact["passed"]:
        raise RuntimeError("shooting-initial-copy-not-exact")

    rows = []
    for year in checkpoint_years:
        if float(year) > float(model.yr):
            model.run_until(float(year))
        row = core._snapshot_row(model, "shooting_last50_diagnostic", float(year), int(audit["ice_node_count"]))
        rows.append(row)
        with gzip.open(output_dir / f"state_year_{int(year):03d}.pkl.gz", "wb") as stream:
            pickle.dump([copy.deepcopy(model.fls[-1])], stream, protocol=pickle.HIGHEST_PROTOCOL)
    pd.DataFrame(rows).to_csv(output_dir / "shooting_checkpoints.csv", index=False)

    row150 = next(row for row in rows if int(row["year"]) == 150)
    row200 = next(row for row in rows if int(row["year"]) == 200)
    signed_volume_50 = (float(row200["volume_m3"]) - float(row150["volume_m3"])) / float(row150["volume_m3"])
    signed_area_50 = (float(row200["area_m2"]) - float(row150["area_m2"])) / float(row150["area_m2"])
    censored_length_change = float(row200["observation_domain_censored_length_m"]) - float(row150["observation_domain_censored_length_m"])
    obs_length_change = None
    if math.isfinite(float(row150["observation_length_m"])) and math.isfinite(float(row200["observation_length_m"])):
        obs_length_change = float(row200["observation_length_m"]) - float(row150["observation_length_m"])
    diagnostic_pass = bool(math.isfinite(signed_volume_50) and abs(signed_volume_50) <= float(tolerance))
    summary = {
        "experiment_id": "cycle4_last50y_shooting_diagnostic_v1",
        "member_id": member_config["member_id"],
        "system_id": int(member_config["system_id"]),
        "rgi_id": member_config["rgi_id"],
        "theta": {"q_v": member_config["q_v"], "z_shape": member_config["z_shape"], "delta_t_c": member_config["delta_t_c"]},
        "original_zero_mb_verified_pass": True,
        "original_zero_mb_summary": str(original_zero_path),
        "original_zero_mb_sha256": _sha256(original_zero_path),
        "zero_mb_rerun": False,
        "forward_run": False,
        "eki_run": False,
        "candidate_reproduction": reproduction,
        "construction": construction,
        "shooting_initial_exact": initial_exact,
        "checkpoint_years": checkpoint_years,
        "signed_delta_volume_last50": signed_volume_50,
        "absolute_delta_volume_last50": abs(signed_volume_50),
        "signed_delta_area_last50": signed_area_50,
        "absolute_delta_area_last50": abs(signed_area_50),
        "observation_length_change_last50_m": obs_length_change,
        "censored_length_change_last50_m": censored_length_change,
        "overflow_at_150": bool(row150["observation_domain_overflow"]),
        "overflow_at_200": bool(row200["observation_domain_overflow"]),
        "overflow_any_checkpoint": any(bool(row["observation_domain_overflow"]) for row in rows),
        "diagnostic_pass_definition": "abs(V200-V150)/V150 <= 0.02 only",
        "diagnostic_pass": diagnostic_pass,
        "runtime_seconds": time.perf_counter() - started,
        "oggm_version": oggm_version,
        "python_executable": sys.executable,
        "source_member_config": str(member_config_path),
        "source_member_config_sha256": _sha256(member_config_path),
        "source_candidate_sha256": _sha256(original_candidate_path),
        "climate_audit": climate_audit,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def _subprocess_member(task: dict[str, Any]) -> dict[str, Any]:
    command = [sys.executable, "-m", MODULE, "--single-member-config", str(task["config"]), "--single-output-dir", str(task["output"]), "--single-workdir", str(task["workdir"]), "--checkpoint-years", "0", "50", "100", "150", "200", "--tolerance", "0.02"]
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    task["output"].mkdir(parents=True, exist_ok=True)
    (task["output"] / "stdout.log").write_text(result.stdout, encoding="utf-8")
    (task["output"] / "stderr.log").write_text(result.stderr, encoding="utf-8")
    return {"member": task["member"], "returncode": result.returncode, "output": str(task["output"])}


def run_batch(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_root = _resolve(config["source_generation_root"])
    output_root = _resolve(config["output_root"])
    work_root = _resolve(config["workdir_root"])
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refuse-nonempty-versioned-output:{output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = []
    for member in config["members"]:
        tasks.append({"member": member, "config": source_root / "member_configs" / f"{member}.json", "output": output_root / "members" / member, "workdir": work_root / member})
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, int(config["max_workers"]))) as pool:
        executions = list(pool.map(_subprocess_member, tasks))
    summaries = []
    for execution in executions:
        summary_path = Path(execution["output"]) / "summary.json"
        if summary_path.exists():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
        else:
            summaries.append({"member_id": execution["member"], "diagnostic_pass": False, "error": f"returncode:{execution['returncode']}"})
    pd.DataFrame(summaries).to_csv(output_root / "diagnostic_summary.csv", index=False)
    result = {
        "experiment_id": config["experiment_id"],
        "members_requested": len(tasks),
        "members_completed": sum("error" not in item for item in summaries),
        "diagnostic_pass_count": sum(bool(item.get("diagnostic_pass")) for item in summaries),
        "all_failed_last50_gate": not any(bool(item.get("diagnostic_pass")) for item in summaries),
        "forward_run": False,
        "zero_mb_rerun": False,
        "eki_run": False,
        "member_results": summaries,
    }
    _write_json(output_root / "diagnostic_decision.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "Himalaya_greatGlacier" / "With_Rec_climate_organized" / "ice_surface_reconstruction" / "config" / "cycle4_last50y_shooting_diagnostic_v1.json")
    parser.add_argument("--single-member-config", type=Path)
    parser.add_argument("--single-output-dir", type=Path)
    parser.add_argument("--single-workdir", type=Path)
    parser.add_argument("--checkpoint-years", type=int, nargs="+")
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args()
    if args.single_member_config:
        result = run_single(args.single_member_config.resolve(), args.single_output_dir.resolve(), args.single_workdir.resolve(), args.checkpoint_years, args.tolerance)
    else:
        result = run_batch(args.config.resolve())
    print(json.dumps(core._json_safe(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
