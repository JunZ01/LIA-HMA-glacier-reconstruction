"""One immutable OGGM flowline-1D spin-up cell for the v10.216 matrix."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import os
import pickle
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .lia_elevation_band_spinup import _clone_gdir, _configure_oggm, _gdir_path, _read_flowline
from .lia_first10_spinup_batch import _write_case_climate


BLOCK_YEARS = 300
MAX_YEARS = 3000
CYCLE_YEARS = 30
VOLUME_CYCLE_TOL = 0.003
VOLUME_90_TOL = 0.009
LENGTH_GRID_TOL = 1.0
STATE_RTOL = 1e-7
CONTINUATION_REL_TOL = 1e-3
THICKNESS_ABS_TOL_M = 1.0
BOUNDARY_MARGIN_M = 5000.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_history(path: Path, history: dict[int, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for year in sorted(history):
            handle.write(json.dumps(history[year], sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def topology_signature(fls) -> dict:
    items = []
    for fl in fls:
        bed = np.asarray(fl.bed_h, dtype=np.float64)
        widths = np.asarray(fl.widths_m, dtype=np.float64)
        payload = bed.tobytes() + widths.tobytes()
        items.append({
            "nx": int(fl.nx),
            "dx": float(fl.dx),
            "map_dx": float(fl.map_dx),
            "dx_meter": float(fl.dx_meter),
            "bed_width_sha256": hashlib.sha256(payload).hexdigest().upper(),
        })
    return {"flowline_count": len(items), "flowlines": items}


def snapshot(model, year: int) -> dict:
    fl = model.fls[-1]
    thick = np.asarray(fl.thick, dtype=np.float64)
    positive = np.flatnonzero(thick > 0)
    last = int(positive[-1]) if positive.size else -1
    free_nodes = len(thick) if last < 0 else len(thick) - 1 - last
    return {
        "year": int(year),
        "volume_m3": float(model.volume_m3),
        "area_m2": float(model.area_m2),
        "length_m": float(model.length_m),
        "dx_meter": float(fl.dx_meter),
        "ice_node_count": int(positive.size),
        "last_ice_index": last,
        "boundary_margin_m": float(free_nodes * fl.dx_meter),
        "array_end_thickness_m": float(thick[-1]),
        "minimum_thickness_m": float(np.min(thick)),
        "maximum_thickness_m": float(np.max(thick)),
    }


def relative_difference(left: float, right: float) -> float:
    scale = max(abs(float(left)), abs(float(right)), np.finfo(float).tiny)
    return abs(float(left) - float(right)) / scale


def convergence_metrics(history: dict[int, dict], year: int) -> dict:
    years = [year - 90, year - 60, year - 30, year]
    rows = [history[item] for item in years]
    volume_cycles = [relative_difference(rows[i]["volume_m3"], rows[i + 1]["volume_m3"]) for i in range(3)]
    length_grid_cycles = [
        abs(rows[i + 1]["length_m"] - rows[i]["length_m"]) / rows[-1]["dx_meter"]
        for i in range(3)
    ]
    volume_90 = relative_difference(rows[0]["volume_m3"], rows[-1]["volume_m3"])
    length_grid_90 = abs(rows[-1]["length_m"] - rows[0]["length_m"]) / rows[-1]["dx_meter"]
    passed = (
        all(value <= VOLUME_CYCLE_TOL for value in volume_cycles)
        and volume_90 <= VOLUME_90_TOL
        and all(value <= LENGTH_GRID_TOL for value in length_grid_cycles)
        and length_grid_90 <= LENGTH_GRID_TOL
    )
    return {
        "years": years,
        "volume_relative_cycle_differences": volume_cycles,
        "volume_relative_90y_drift": volume_90,
        "length_grid_cycle_differences": length_grid_cycles,
        "length_grid_90y_difference": length_grid_90,
        "passed": bool(passed),
    }


def save_checkpoint(root: Path, model, year: int, kind: str, cell: dict, diagnostics: dict) -> Path:
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    final = checkpoints / f"year-{year:04d}"
    if final.exists():
        raise FileExistsError(f"checkpoint-collision:{final}")
    temp = checkpoints / f".year-{year:04d}.tmp.{os.getpid()}"
    temp.mkdir(parents=False, exist_ok=False)
    state_path = temp / "state.pkl.gz"
    with gzip.open(state_path, "wb") as handle:
        pickle.dump(model.fls, handle, protocol=pickle.HIGHEST_PROTOCOL)
    state_hash = sha256(state_path)
    manifest = {
        "schema": "lia-flowline-atomic-checkpoint-v10.216",
        "fingerprint": cell["fingerprint"],
        "lia_unique_id": int(cell["lia_unique_id"]),
        "rgi_id": cell["rgi_id"],
        "year": int(year),
        "kind": kind,
        "state_sha256": state_hash,
        "topology": topology_signature(model.fls),
        "parameters": {key: cell[key] for key in (
            "delta_t_eq_degC", "base_temp_bias_degC", "melt_f", "prcp_fac",
            "glen_a_pa_3_s_1", "fs", "temp_melt_degC",
        )},
        "diagnostics": diagnostics,
    }
    atomic_json(temp / "checkpoint.json", manifest)
    os.replace(temp, final)
    return final


def load_checkpoint(path: Path, cell: dict):
    manifest_path = path / "checkpoint.json"
    state_path = path / "state.pkl.gz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["fingerprint"] != cell["fingerprint"]:
        raise ValueError("checkpoint-fingerprint-mismatch")
    if int(manifest["lia_unique_id"]) != int(cell["lia_unique_id"]) or manifest["rgi_id"] != cell["rgi_id"]:
        raise ValueError("checkpoint-member-mismatch")
    if sha256(state_path) != manifest["state_sha256"]:
        raise ValueError("checkpoint-state-hash-mismatch")
    expected = {key: cell[key] for key in manifest["parameters"]}
    if manifest["parameters"] != expected:
        raise ValueError("checkpoint-parameter-mismatch")
    with gzip.open(state_path, "rb") as handle:
        fls = pickle.load(handle)
    if topology_signature(fls) != manifest["topology"]:
        raise ValueError("checkpoint-topology-mismatch")
    return manifest, fls


def make_model(fls, mb_model, glen_a: float, year: int):
    from oggm import cfg
    from oggm.core.flowline import SemiImplicitModel

    return SemiImplicitModel(
        [copy.deepcopy(fl) for fl in fls], mb_model=mb_model, y0=float(year),
        glen_a=float(glen_a), fs=0.0, inplace=False, check_for_boundaries=True,
        fixed_dt=1.25 * cfg.SEC_IN_DAY,
    )


def state_restore_check(original_model, restored_model) -> dict:
    if topology_signature(original_model.fls) != topology_signature(restored_model.fls):
        raise ValueError("restore-topology-mismatch")
    maximum_relative = 0.0
    for original, restored in zip(original_model.fls, restored_model.fls):
        left = np.asarray(original.thick, dtype=np.float64)
        right = np.asarray(restored.thick, dtype=np.float64)
        if not np.allclose(left, right, rtol=STATE_RTOL, atol=0.0, equal_nan=False):
            raise ValueError("restore-state-rtol-failure")
        scale = np.maximum(np.abs(left), np.finfo(float).tiny)
        maximum_relative = max(maximum_relative, float(np.max(np.abs(left - right) / scale)))
    return {"rtol_threshold": STATE_RTOL, "maximum_relative_thickness_difference": maximum_relative, "passed": True}


def continuation_check(original_model, restored_model, target_year: int) -> tuple[dict, object]:
    original_model.run_until(float(target_year))
    restored_model.run_until(float(target_year))
    original = snapshot(original_model, target_year)
    restored = snapshot(restored_model, target_year)
    max_thickness = 0.0
    for left_fl, right_fl in zip(original_model.fls, restored_model.fls):
        max_thickness = max(max_thickness, float(np.max(np.abs(
            np.asarray(left_fl.thick, dtype=np.float64) - np.asarray(right_fl.thick, dtype=np.float64)
        ))))
    volume_rel = relative_difference(original["volume_m3"], restored["volume_m3"])
    area_rel = relative_difference(original["area_m2"], restored["area_m2"])
    length_grid = abs(original["length_m"] - restored["length_m"]) / restored["dx_meter"]
    passed = (
        volume_rel <= CONTINUATION_REL_TOL and area_rel <= CONTINUATION_REL_TOL
        and max_thickness <= THICKNESS_ABS_TOL_M and length_grid <= LENGTH_GRID_TOL
    )
    result = {
        "target_year": int(target_year), "volume_relative_difference": volume_rel,
        "area_relative_difference": area_rel, "maximum_thickness_absolute_difference_m": max_thickness,
        "length_grid_difference": length_grid, "relative_tolerance": CONTINUATION_REL_TOL,
        "thickness_absolute_tolerance_m": THICKNESS_ABS_TOL_M,
        "length_grid_tolerance": LENGTH_GRID_TOL, "passed": bool(passed),
    }
    return result, restored_model


def load_cell(transaction_dir: Path, fingerprint: str) -> tuple[dict, dict, dict]:
    manifest_path = transaction_dir / "transaction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("launch_authorized"):
        raise ValueError("transaction-not-launch-authorized")
    if sha256(Path(__file__)) != manifest["runtime"]["chain_driver_sha256"]:
        raise ValueError("chain-driver-hash-mismatch")
    with (transaction_dir / "level1_matrix.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["fingerprint"] == fingerprint]
    if len(rows) != 1:
        raise ValueError(f"fingerprint-row-count:{len(rows)}")
    row = rows[0]
    for key in (
        "lia_unique_id", "delta_t_eq_degC", "base_temp_bias_degC", "melt_f", "prcp_fac",
        "glen_a_pa_3_s_1", "fs", "temp_melt_degC",
    ):
        row[key] = int(row[key]) if key == "lia_unique_id" else float(row[key])
    if not manifest.get("production_authorized"):
        raise ValueError("v10.216-production-not-authorized")
    if fingerprint not in set(manifest["production"]["fingerprints"]):
        raise ValueError("fingerprint-not-authorized-for-v10.216-production")
    members = manifest["production"]["members"]
    expected_members = int(manifest["production"]["cell_count"])
    if len(members) != expected_members or len({item["key"] for item in members}) != expected_members:
        raise ValueError("short-member-map-cardinality-or-key-collision")
    matches = [item for item in members if item["fingerprint"] == fingerprint]
    if len(matches) != 1:
        raise ValueError(f"short-member-map-row-count:{len(matches)}")
    member = matches[0]
    key = str(member["key"])
    if len(key) != 5 or not key.startswith("c") or not key[1:].isdigit():
        raise ValueError(f"invalid-short-member-key:{key}")
    return manifest, row, member


def member_paths(manifest: dict, member: dict) -> tuple[Path, Path, Path]:
    short_root = Path(manifest["runtime"]["unified_short_root"]).resolve()
    member_root = (short_root / "m" / member["key"]).resolve()
    try:
        member_root.relative_to(short_root)
    except ValueError as exc:
        raise ValueError("member-root-escapes-unified-short-root") from exc
    return member_root, member_root / "w", member_root / "c"


def failure_output_dir(args) -> Path:
    try:
        manifest = json.loads((args.transaction_dir / "transaction_manifest.json").read_text(encoding="utf-8"))
        matches = [item for item in manifest["production"]["members"] if item["fingerprint"] == args.fingerprint]
        if len(matches) == 1:
            return member_paths(manifest, matches[0])[2]
    except Exception:
        pass
    return args.output_root / "x" / args.fingerprint[:8]


def run(args) -> dict:
    started = time.perf_counter()
    manifest, cell, member = load_cell(args.transaction_dir, args.fingerprint)
    expected_root = Path(manifest["runtime"]["unified_short_root"])
    if args.output_root.resolve() != expected_root.resolve():
        raise ValueError("unified-short-root-contract-mismatch")
    _, workdir, output_dir = member_paths(manifest, member)
    result_path = output_dir / "result.json"
    if result_path.exists():
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        if prior.get("terminal"):
            print(json.dumps({"status": "no-run-duplicate", "prior": prior}, ensure_ascii=False))
            return prior
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_oggm(workdir, output_dir)
    from oggm import cfg, workflow
    from oggm.core.massbalance import MonthlyTIModel

    cfg.PARAMS["temp_melt"] = float(cell["temp_melt_degC"])
    source_root = Path(manifest["inputs"]["source_gdir_root"])
    elevation_root = Path(manifest["inputs"]["elevation_band_root"])
    climate_path = Path(manifest["inputs"]["reconstruction_climate"])
    pixel_table = pd.read_csv(manifest["inputs"]["climate_pixels"])
    pixels = pixel_table[
        pixel_table["lia_unique_id"].astype(int).eq(int(cell["lia_unique_id"]))
        & pixel_table["rgi_id"].astype(str).eq(cell["rgi_id"])
        & pixel_table["climate_id"].astype(str).eq("ensemble_rec_ref")
    ]
    if len(pixels) != 1:
        raise ValueError(f"climate-pixel-row-count:{len(pixels)}")
    source = _gdir_path(source_root, cell["rgi_id"])
    target = _gdir_path(workdir / "per_glacier", cell["rgi_id"])
    _clone_gdir(source, target)
    gdir = workflow.init_glacier_directories([cell["rgi_id"]], reset=False)[0]
    climate_suffix, climate_audit = _write_case_climate(
        gdir, climate_path, pixels.iloc[0], "off", start_year=1441, end_year=1470,
    )
    mb_model = MonthlyTIModel(
        gdir, filename="gcm_data", input_filesuffix=climate_suffix,
        melt_f=float(cell["melt_f"]), prcp_fac=float(cell["prcp_fac"]),
        temp_bias=float(cell["base_temp_bias_degC"] + cell["delta_t_eq_degC"]),
        check_calib_params=False, ys=1441, ye=1470, repeat=True,
    )

    checkpoints = sorted((output_dir / "checkpoints").glob("year-*")) if (output_dir / "checkpoints").exists() else []
    history_path = output_dir / "diagnostics.jsonl"
    history: dict[int, dict] = {}
    if history_path.exists():
        for line in history_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                history[int(item["year"])] = item
    if checkpoints:
        checkpoint_manifest, fls = load_checkpoint(checkpoints[-1], cell)
        current_year = int(checkpoint_manifest["year"])
        model = make_model(fls, mb_model, cell["glen_a_pa_3_s_1"], current_year)
    else:
        flowline = _read_flowline(elevation_root / str(cell["lia_unique_id"]) / "elevation_band_flowline_lia.pkl.gz")
        current_year = 0
        model = make_model([flowline], mb_model, cell["glen_a_pa_3_s_1"], 0)
        history[0] = snapshot(model, 0)

    checks: list[dict] = []
    continuation_anomalies: list[dict] = []
    first_pass_year = None
    preflight_passed = False
    terminal = False
    terminal_status = None
    while current_year < MAX_YEARS:
        next_boundary = min(((current_year // BLOCK_YEARS) + 1) * BLOCK_YEARS, MAX_YEARS)
        for year in range(((current_year // CYCLE_YEARS) + 1) * CYCLE_YEARS, next_boundary + 1, CYCLE_YEARS):
            model.run_until(float(year))
            history[year] = snapshot(model, year)
        current_year = next_boundary
        current = history[current_year]
        if not all(math.isfinite(current[key]) for key in ("volume_m3", "area_m2", "length_m", "minimum_thickness_m")):
            raise ValueError("nonfinite-state")
        if current["volume_m3"] <= 0 or current["area_m2"] <= 0 or current["length_m"] <= 0 or current["minimum_thickness_m"] < 0:
            terminal, terminal_status = True, "scientific-domain-failure-nonpositive-or-negative-state"
        if current["boundary_margin_m"] < BOUNDARY_MARGIN_M:
            terminal, terminal_status = True, "scientific-domain-failure-boundary-margin"
        convergence = convergence_metrics(history, current_year)
        checkpoint = save_checkpoint(output_dir, model, current_year, "boundary", cell, {
            "state": current, "convergence": convergence,
        })
        atomic_history(history_path, history)
        checkpoint_manifest, restored_fls = load_checkpoint(checkpoint, cell)
        restored = make_model(restored_fls, mb_model, cell["glen_a_pa_3_s_1"], current_year)
        restore = state_restore_check(model, restored)
        if terminal:
            checks.append({"year": current_year, "convergence": convergence, "restore": restore})
            break
        if current_year + CYCLE_YEARS <= MAX_YEARS:
            continuation, restored = continuation_check(model, restored, current_year + CYCLE_YEARS)
            next_model = restored
            if not continuation["passed"]:
                continuation["action"] = "warning-continue-uninterrupted-model"
                continuation_anomalies.append(dict(continuation))
                next_model = model
            history[current_year + CYCLE_YEARS] = snapshot(next_model, current_year + CYCLE_YEARS)
            checks.append({"year": current_year, "convergence": convergence, "restore": restore, "continuation": continuation})
            if args.mode == "preflight":
                preflight_passed = True
            if convergence["passed"]:
                first_pass_year = current_year
                confirmation = convergence_metrics(history, current_year + CYCLE_YEARS)
                checks[-1]["rolling_30y_confirmation"] = confirmation
                if confirmation["passed"]:
                    current_year += CYCLE_YEARS
                    model = next_model
                    final_checkpoint = save_checkpoint(output_dir, model, current_year, "confirmed-convergence", cell, {
                        "state": history[current_year], "confirmation": confirmation,
                    })
                    atomic_history(history_path, history)
                    terminal, terminal_status = True, "converged"
                    break
            if args.mode == "preflight":
                terminal_status = "preflight-pass-checkpoint-300"
                break
            current_year += CYCLE_YEARS
            model = next_model
        else:
            checks.append({"year": current_year, "convergence": convergence, "restore": restore})
            terminal, terminal_status = True, "max-duration-nonconverged"
            break

    atomic_history(history_path, history)
    result = {
        "schema": "lia-all191-flowline-cell-result-v10.216",
        "status": terminal_status,
        "technical_failure": False,
        "scientific_nonconvergence": terminal_status == "max-duration-nonconverged",
        "terminal": bool(terminal),
        "preflight_passed": bool(preflight_passed),
        "fingerprint": cell["fingerprint"],
        "lia_unique_id": int(cell["lia_unique_id"]),
        "rgi_id": cell["rgi_id"],
        "mode": args.mode,
        "current_year": int(current_year),
        "first_pass_year": first_pass_year,
        "continuation_anomaly_count": len(continuation_anomalies),
        "continuation_anomalies": continuation_anomalies,
        "checks": checks,
        "climate": climate_audit,
        "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transaction-dir", required=True, type=Path)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("production",), required=True)
    args = parser.parse_args()
    try:
        run(args)
        return 0
    except Exception as exc:
        output_dir = failure_output_dir(args)
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema": "lia-all191-flowline-cell-result-v10.216",
            "status": "technical-failure", "technical_failure": True,
            "scientific_nonconvergence": False, "terminal": True,
            "preflight_passed": False, "fingerprint": args.fingerprint,
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
        }
        atomic_json(output_dir / "result.json", failure)
        print(json.dumps(failure, indent=2, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
