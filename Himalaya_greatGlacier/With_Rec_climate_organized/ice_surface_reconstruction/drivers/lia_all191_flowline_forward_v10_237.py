"""One immutable 1470-2000 OGGM flowline forward member for v10.237."""

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
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .lia_all191_flowline_spinup_v10_216 import (
    atomic_history,
    atomic_json,
    relative_difference,
    snapshot,
    topology_signature,
)
from .lia_elevation_band_spinup import _clone_gdir, _configure_oggm, _gdir_path
from .lia_first10_spinup_batch import _write_case_climate


START_YEAR = 1470
END_YEAR = 2000
PREFLIGHT_END_YEAR = 1770
BOUNDARY_MARGIN_M = 5000.0
STATE_RTOL = 1e-7
RESUME_REL_TOL = 1e-3
RESUME_THICKNESS_ABS_TOL_M = 1.0
RESUME_LENGTH_GRID_TOL = 1.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_contract(transaction_dir: Path, fingerprint: str, mode: str):
    manifest = read_json(transaction_dir / "transaction_manifest.json")
    if not manifest.get("launch_authorized"):
        raise ValueError("transaction-not-launch-authorized")
    if sha256(Path(__file__)) != manifest["runtime"]["worker_sha256"]:
        raise ValueError("worker-hash-mismatch")
    with (transaction_dir / "forward_matrix.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["forward_fingerprint"] == fingerprint]
    if len(rows) != 1:
        raise ValueError(f"forward-fingerprint-row-count:{len(rows)}")
    row = rows[0]
    selected = manifest["preflight"]["fingerprints"] if mode == "preflight" else manifest["production"]["fingerprints"]
    if fingerprint not in set(selected):
        raise ValueError(f"fingerprint-not-authorized-for-{mode}")
    for key in (
        "forward_candidate_ordinal", "cell_ordinal", "lia_unique_id",
        "original_strict_confirmation_state_year",
    ):
        row[key] = int(row[key])
    for key in (
        "delta_t_eq_degC", "base_temp_bias_degC", "effective_temp_bias_degC",
        "melt_f", "prcp_fac", "glen_a_pa_3_s_1", "fs", "temp_melt_degC",
    ):
        row[key] = float(row[key])
    return manifest, row


def member_paths(manifest: dict, row: dict, mode: str):
    root = Path(manifest["runtime"]["unified_short_root"]).resolve()
    key = row["preflight_member_key"] if mode == "preflight" else row["production_member_key"]
    group = "p" if mode == "preflight" else "m"
    member_root = (root / group / key).resolve()
    member_root.relative_to(root)
    return key, member_root, member_root / "w", member_root / "c"


def make_model(fls, mb_model, glen_a: float, fs: float, year: int):
    from oggm import cfg
    from oggm.core.flowline import SemiImplicitModel

    return SemiImplicitModel(
        [copy.deepcopy(fl) for fl in fls], mb_model=mb_model, y0=float(year),
        glen_a=float(glen_a), fs=float(fs), inplace=False,
        check_for_boundaries=True, fixed_dt=1.25 * cfg.SEC_IN_DAY,
    )


def save_forward_checkpoint(root: Path, model, year: int, row: dict) -> Path:
    checkpoint_root = root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    final = checkpoint_root / f"year-{year:04d}"
    if final.exists():
        raise FileExistsError(f"checkpoint-collision:{final}")
    temp = checkpoint_root / f".y{year}.tmp.{os.getpid()}"
    temp.mkdir(parents=False, exist_ok=False)
    state_path = temp / "state.pkl.gz"
    with gzip.open(state_path, "wb") as handle:
        pickle.dump(model.fls, handle, protocol=pickle.HIGHEST_PROTOCOL)
    state_hash = sha256(state_path)
    payload = {
        "schema": "lia-flowline-forward-atomic-checkpoint-v10.237",
        "forward_fingerprint": row["forward_fingerprint"],
        "source_checkpoint_state_sha256": row["confirmation_checkpoint_state_sha256"],
        "lia_unique_id": row["lia_unique_id"], "rgi_id": row["rgi_id"],
        "calendar_year": year, "state_sha256": state_hash,
        "topology": topology_signature(model.fls), "diagnostics": snapshot(model, year),
    }
    atomic_json(temp / "checkpoint.json", payload)
    os.replace(temp, final)
    return final


def load_forward_checkpoint(path: Path, row: dict):
    payload = read_json(path / "checkpoint.json")
    state_path = path / "state.pkl.gz"
    if payload["forward_fingerprint"] != row["forward_fingerprint"]:
        raise ValueError("forward-checkpoint-fingerprint-mismatch")
    if sha256(state_path) != payload["state_sha256"]:
        raise ValueError("forward-checkpoint-state-hash-mismatch")
    with gzip.open(state_path, "rb") as handle:
        fls = pickle.load(handle)
    if topology_signature(fls) != payload["topology"]:
        raise ValueError("forward-checkpoint-topology-mismatch")
    return int(payload["calendar_year"]), fls


def resume_comparison(original, restored, target_year: int) -> dict:
    original.run_until(float(target_year))
    restored.run_until(float(target_year))
    left = snapshot(original, target_year)
    right = snapshot(restored, target_year)
    max_thickness = max(
        float(np.max(np.abs(np.asarray(a.thick) - np.asarray(b.thick))))
        for a, b in zip(original.fls, restored.fls)
    )
    volume_rel = relative_difference(left["volume_m3"], right["volume_m3"])
    area_rel = relative_difference(left["area_m2"], right["area_m2"])
    length_grid = abs(left["length_m"] - right["length_m"]) / right["dx_meter"]
    passed = bool(
        volume_rel <= RESUME_REL_TOL and area_rel <= RESUME_REL_TOL
        and max_thickness <= RESUME_THICKNESS_ABS_TOL_M
        and length_grid <= RESUME_LENGTH_GRID_TOL
    )
    return {
        "target_year": target_year, "volume_relative_difference": volume_rel,
        "area_relative_difference": area_rel,
        "maximum_thickness_absolute_difference_m": max_thickness,
        "length_grid_difference": length_grid, "passed": passed,
    }


def validate_state(model, year: int) -> dict:
    state = snapshot(model, year)
    values = (state["volume_m3"], state["area_m2"], state["length_m"], state["minimum_thickness_m"])
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite-forward-state")
    if state["volume_m3"] <= 0 or state["area_m2"] <= 0 or state["length_m"] <= 0:
        raise ValueError("nonpositive-forward-state")
    if state["minimum_thickness_m"] < 0:
        raise ValueError("negative-forward-thickness")
    if state["boundary_margin_m"] < BOUNDARY_MARGIN_M:
        raise RuntimeError("scientific-domain-failure-boundary-margin")
    return state


def run(args) -> dict:
    started = time.perf_counter()
    manifest, row = load_contract(args.transaction_dir, args.fingerprint, args.mode)
    expected_root = Path(manifest["runtime"]["unified_short_root"]).resolve()
    if args.output_root.resolve() != expected_root:
        raise ValueError("unified-short-root-contract-mismatch")
    key, _, workdir, output_dir = member_paths(manifest, row, args.mode)
    result_path = output_dir / "result.json"
    if result_path.exists() and read_json(result_path).get("terminal"):
        return read_json(result_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = Path(row["confirmation_checkpoint_manifest_path"])
    source_state = Path(row["confirmation_checkpoint_state_path"])
    if sha256(source_manifest) != row["confirmation_checkpoint_manifest_sha256"]:
        raise ValueError("source-checkpoint-manifest-hash-mismatch")
    if sha256(source_state) != row["confirmation_checkpoint_state_sha256"]:
        raise ValueError("source-checkpoint-state-hash-mismatch")
    source_payload = read_json(source_manifest)
    if int(source_payload["lia_unique_id"]) != row["lia_unique_id"] or source_payload["rgi_id"] != row["rgi_id"]:
        raise ValueError("source-checkpoint-member-mismatch")
    with gzip.open(source_state, "rb") as handle:
        initial_fls = pickle.load(handle)
    if topology_signature(initial_fls) != source_payload["topology"]:
        raise ValueError("source-checkpoint-topology-mismatch")

    _configure_oggm(workdir, output_dir)
    from oggm import cfg, workflow
    from oggm.core.massbalance import MonthlyTIModel

    cfg.PARAMS["temp_melt"] = row["temp_melt_degC"]
    source_root = Path(manifest["inputs"]["source_gdir_root"])
    target = _gdir_path(workdir / "per_glacier", row["rgi_id"])
    _clone_gdir(_gdir_path(source_root, row["rgi_id"]), target)
    gdir = workflow.init_glacier_directories([row["rgi_id"]], reset=False)[0]
    pixels = pd.read_csv(manifest["inputs"]["climate_pixels"])
    selected = pixels[
        pixels["lia_unique_id"].astype(int).eq(row["lia_unique_id"])
        & pixels["rgi_id"].astype(str).eq(row["rgi_id"])
        & pixels["climate_id"].astype(str).eq("ensemble_rec_ref")
    ]
    if len(selected) != 1:
        raise ValueError(f"climate-pixel-row-count:{len(selected)}")
    suffix, climate_audit = _write_case_climate(
        gdir, Path(manifest["inputs"]["reconstruction_climate"]), selected.iloc[0],
        "off", start_year=START_YEAR, end_year=END_YEAR,
    )
    mb_model = MonthlyTIModel(
        gdir, filename="gcm_data", input_filesuffix=suffix,
        melt_f=row["melt_f"], prcp_fac=row["prcp_fac"],
        temp_bias=row["effective_temp_bias_degC"], check_calib_params=False,
        ys=START_YEAR, ye=END_YEAR, repeat=False,
    )

    history_path = output_dir / "diagnostics.jsonl"
    history = {}
    if history_path.exists():
        for line in history_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line); history[int(item["year"])] = item
    checkpoints = sorted((output_dir / "checkpoints").glob("year-*")) if (output_dir / "checkpoints").exists() else []
    if checkpoints:
        current_year, fls = load_forward_checkpoint(checkpoints[-1], row)
        model = make_model(fls, mb_model, row["glen_a_pa_3_s_1"], row["fs"], current_year)
    else:
        current_year = START_YEAR
        model = make_model(initial_fls, mb_model, row["glen_a_pa_3_s_1"], row["fs"], START_YEAR)
        history[START_YEAR] = validate_state(model, START_YEAR)
        save_forward_checkpoint(output_dir, model, START_YEAR, row)
        atomic_history(history_path, history)

    target_year = PREFLIGHT_END_YEAR if args.mode == "preflight" else END_YEAR
    resume_check = None
    while current_year < target_year:
        next_year = current_year + 1
        if args.mode == "preflight" and next_year == target_year:
            prior_checkpoint = output_dir / "checkpoints" / f"year-{current_year:04d}"
            _, restored_fls = load_forward_checkpoint(prior_checkpoint, row)
            restored = make_model(restored_fls, mb_model, row["glen_a_pa_3_s_1"], row["fs"], current_year)
            resume_check = resume_comparison(model, restored, next_year)
            if not resume_check["passed"]:
                raise ValueError("preflight-resume-comparison-failure")
            model = restored
        else:
            model.run_until(float(next_year))
        current_year = next_year
        history[current_year] = validate_state(model, current_year)
        save_forward_checkpoint(output_dir, model, current_year, row)
        atomic_history(history_path, history)

    result = {
        "schema": "lia-all191-flowline-forward-member-result-v10.237",
        "status": "preflight-pass" if args.mode == "preflight" else "completed",
        "terminal": True, "technical_failure": False, "scientific_failure": False,
        "mode": args.mode, "member_key": key,
        "forward_fingerprint": row["forward_fingerprint"],
        "lia_unique_id": row["lia_unique_id"], "rgi_id": row["rgi_id"],
        "delta_t_eq_degC": row["delta_t_eq_degC"],
        "start_year": START_YEAR, "end_year": current_year,
        "resume_check": resume_check, "climate": climate_audit,
        "runtime_seconds": time.perf_counter() - started,
        "annual_checkpoint_count": len(list((output_dir / "checkpoints").glob("year-*"))),
    }
    atomic_json(result_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def failure_dir(args):
    try:
        manifest, row = load_contract(args.transaction_dir, args.fingerprint, args.mode)
        return member_paths(manifest, row, args.mode)[3]
    except Exception:
        return args.output_root / "x" / args.fingerprint[:8]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transaction-dir", type=Path, required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "production"), required=True)
    args = parser.parse_args()
    try:
        run(args); return 0
    except Exception as exc:
        out = failure_dir(args); out.mkdir(parents=True, exist_ok=True)
        text = f"{type(exc).__name__}: {exc}"
        scientific = "scientific-domain-failure" in text
        failure = {
            "schema": "lia-all191-flowline-forward-member-result-v10.237",
            "status": "scientific-failure" if scientific else "technical-failure",
            "terminal": True, "technical_failure": not scientific,
            "scientific_failure": scientific, "forward_fingerprint": args.fingerprint,
            "error": text, "traceback": traceback.format_exc(),
        }
        atomic_json(out / "result.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        return 3 if scientific else 2


if __name__ == "__main__":
    raise SystemExit(main())
