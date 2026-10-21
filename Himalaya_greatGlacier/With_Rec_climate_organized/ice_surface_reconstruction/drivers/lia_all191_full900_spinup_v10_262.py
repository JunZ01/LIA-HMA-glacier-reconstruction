"""One immutable OGGM flowline-1D full-900-year spin-up cell for v10.262."""

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

from .lia_all191_adaptive_spinup_v10_260 import (
    _clone_gdir,
    _configure_oggm,
    _gdir_path,
    _read_flowline,
    _write_case_climate,
    atomic_history,
    atomic_json,
    convergence_metrics,
    make_model,
    relative_difference,
    sha256,
    snapshot,
    state_restore_check,
    topology_signature,
)


MAX_YEARS = 900
CHECKPOINT_INTERVAL_YEARS = 30
BOUNDARY_MARGIN_M = 5000.0
LIA_LENGTH_REL_TOL = 0.10
EXPECTED_CHECKPOINT_YEARS = list(range(0, MAX_YEARS + 1, CHECKPOINT_INTERVAL_YEARS))


def save_checkpoint(
    root: Path,
    model,
    year: int,
    kind: str,
    cell: dict,
    diagnostics: dict,
) -> Path:
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    final = checkpoints / f"year-{year:04d}"
    if final.exists():
        raise FileExistsError(f"checkpoint-collision:{final}")
    temp = checkpoints / f".y{year:04d}.tmp.{os.getpid()}"
    temp.mkdir(parents=False, exist_ok=False)
    state_path = temp / "state.pkl.gz"
    with gzip.open(state_path, "wb") as handle:
        pickle.dump(model.fls, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        "schema": "lia-flowline-atomic-checkpoint-v10.262",
        "fingerprint": cell["fingerprint"],
        "lia_unique_id": int(cell["lia_unique_id"]),
        "rgi_id": cell["rgi_id"],
        "year": int(year),
        "kind": kind,
        "state_sha256": sha256(state_path),
        "topology": topology_signature(model.fls),
        "parameters": {
            key: cell[key]
            for key in (
                "delta_t_eq_degC",
                "base_temp_bias_degC",
                "melt_f",
                "prcp_fac",
                "glen_a_pa_3_s_1",
                "fs",
                "temp_melt_degC",
                "target_lia_length_m",
                "search_round",
                "search_stage",
                "proposal_delta_t_continuous_degC",
                "checkpoint_interval_years",
                "maximum_years",
            )
        },
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
    if (
        int(manifest["lia_unique_id"]) != int(cell["lia_unique_id"])
        or manifest["rgi_id"] != cell["rgi_id"]
    ):
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


def load_cell(transaction_dir: Path, fingerprint: str) -> tuple[dict, dict, dict]:
    manifest = json.loads(
        (transaction_dir / "transaction_manifest.json").read_text(encoding="utf-8")
    )
    if not manifest.get("launch_authorized"):
        raise ValueError("transaction-not-launch-authorized")
    if sha256(Path(__file__)) != manifest["runtime"]["chain_driver_sha256"]:
        raise ValueError("chain-driver-hash-mismatch")
    with (transaction_dir / "production_matrix.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = [row for row in csv.DictReader(handle) if row["fingerprint"] == fingerprint]
    if len(rows) != 1:
        raise ValueError(f"fingerprint-row-count:{len(rows)}")
    row = rows[0]
    integer_fields = {
        "lia_unique_id",
        "search_round",
        "checkpoint_interval_years",
        "maximum_years",
    }
    float_fields = {
        "delta_t_eq_degC",
        "base_temp_bias_degC",
        "melt_f",
        "prcp_fac",
        "glen_a_pa_3_s_1",
        "fs",
        "temp_melt_degC",
        "target_lia_length_m",
        "proposal_delta_t_continuous_degC",
    }
    for key in integer_fields:
        row[key] = int(row[key])
    for key in float_fields:
        row[key] = float(row[key])
    if row["maximum_years"] != MAX_YEARS:
        raise ValueError("maximum-year-contract-mismatch")
    if row["checkpoint_interval_years"] != CHECKPOINT_INTERVAL_YEARS:
        raise ValueError("checkpoint-interval-contract-mismatch")
    selected = set(manifest["production"]["fingerprints"])
    if fingerprint not in selected or not manifest.get("production_authorized"):
        raise ValueError("fingerprint-not-authorized")
    matches = [
        item for item in manifest["production"]["members"] if item["fingerprint"] == fingerprint
    ]
    if len(matches) != 1:
        raise ValueError(f"short-member-map-row-count:{len(matches)}")
    member = matches[0]
    key = str(member["key"])
    if len(key) != 5 or not key.startswith("s") or not key[1:].isdigit():
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
        manifest = json.loads(
            (args.transaction_dir / "transaction_manifest.json").read_text(encoding="utf-8")
        )
        matches = [
            item
            for item in manifest["production"]["members"]
            if item["fingerprint"] == args.fingerprint
        ]
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
        gdir, climate_path, pixels.iloc[0], "off", start_year=1441, end_year=1470
    )
    mb_model = MonthlyTIModel(
        gdir,
        filename="gcm_data",
        input_filesuffix=climate_suffix,
        melt_f=float(cell["melt_f"]),
        prcp_fac=float(cell["prcp_fac"]),
        temp_bias=float(cell["base_temp_bias_degC"] + cell["delta_t_eq_degC"]),
        check_calib_params=False,
        ys=1441,
        ye=1470,
        repeat=True,
    )

    history_path = output_dir / "diagnostics.jsonl"
    history: dict[int, dict] = {}
    if history_path.exists():
        for line in history_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                history[int(item["year"])] = item
    checkpoints = (
        sorted((output_dir / "checkpoints").glob("year-*"))
        if (output_dir / "checkpoints").exists()
        else []
    )
    checks: list[dict] = []
    domain_flags: list[dict] = []
    if checkpoints:
        checkpoint_manifest, fls = load_checkpoint(checkpoints[-1], cell)
        current_year = int(checkpoint_manifest["year"])
        model = make_model(fls, mb_model, cell["glen_a_pa_3_s_1"], current_year)
        checks.append(
            {
                "year": current_year,
                "resume_entry_restore": {
                    "passed": True,
                    "state_sha256": checkpoint_manifest["state_sha256"],
                },
            }
        )
    else:
        flowline = _read_flowline(
            elevation_root / str(cell["lia_unique_id"]) / "elevation_band_flowline_lia.pkl.gz"
        )
        current_year = 0
        model = make_model([flowline], mb_model, cell["glen_a_pa_3_s_1"], 0)
        history[0] = snapshot(model, 0)
        checkpoint = save_checkpoint(
            output_dir, model, 0, "initial", cell, {"state": history[0]}
        )
        checkpoint_manifest, restored_fls = load_checkpoint(checkpoint, cell)
        restored = make_model(restored_fls, mb_model, cell["glen_a_pa_3_s_1"], 0)
        checks.append(
            {
                "year": 0,
                "restore": state_restore_check(model, restored),
                "state_sha256": checkpoint_manifest["state_sha256"],
            }
        )
        atomic_history(history_path, history)

    boundary_evaluations: list[dict] = []
    while current_year < MAX_YEARS:
        next_year = min(current_year + CHECKPOINT_INTERVAL_YEARS, MAX_YEARS)
        for year in range(current_year + 1, next_year + 1):
            model.run_until(float(year))
            history[year] = snapshot(model, year)
        current_year = next_year
        current = history[current_year]
        if not all(
            math.isfinite(current[key])
            for key in (
                "volume_m3",
                "area_m2",
                "length_m",
                "minimum_thickness_m",
                "maximum_thickness_m",
            )
        ):
            raise ValueError("nonfinite-state")
        if current["minimum_thickness_m"] < -1e-9:
            raise ValueError("negative-thickness-state")
        if current["boundary_margin_m"] < BOUNDARY_MARGIN_M:
            domain_flags.append(
                {
                    "year": current_year,
                    "flag": "boundary-margin-below-5km-continue-to-year900",
                    "boundary_margin_m": current["boundary_margin_m"],
                }
            )
        if current["volume_m3"] <= 0 or current["area_m2"] <= 0 or current["length_m"] <= 0:
            domain_flags.append(
                {
                    "year": current_year,
                    "flag": "nonpositive-glacier-state-continue-to-year900",
                    "volume_m3": current["volume_m3"],
                    "area_m2": current["area_m2"],
                    "length_m": current["length_m"],
                }
            )
        convergence = convergence_metrics(history, current_year) if current_year >= 90 else None
        checkpoint = save_checkpoint(
            output_dir,
            model,
            current_year,
            "recovery-30y",
            cell,
            {"state": current, "convergence": convergence},
        )
        checkpoint_manifest, restored_fls = load_checkpoint(checkpoint, cell)
        restored = make_model(restored_fls, mb_model, cell["glen_a_pa_3_s_1"], current_year)
        checks.append(
            {
                "year": current_year,
                "restore": state_restore_check(model, restored),
                "state_sha256": checkpoint_manifest["state_sha256"],
            }
        )
        if current_year in (300, 600, 900):
            boundary_evaluations.append(
                {"year": current_year, "convergence": convergence, "state": current}
            )
        atomic_history(history_path, history)

    actual_checkpoint_years = [
        int(path.name.split("-")[1])
        for path in sorted((output_dir / "checkpoints").glob("year-*"))
    ]
    if actual_checkpoint_years != EXPECTED_CHECKPOINT_YEARS:
        raise ValueError(
            f"checkpoint-cadence-failure:{actual_checkpoint_years}:"
            f"{EXPECTED_CHECKPOINT_YEARS}"
        )
    final_convergence = convergence_metrics(history, MAX_YEARS)
    terminal_status = (
        "converged-at-year900"
        if final_convergence["passed"]
        else "max-duration-nonconverged"
    )
    first_pass_year = next(
        (
            item["year"]
            for item in boundary_evaluations
            if item["convergence"] and item["convergence"]["passed"]
        ),
        None,
    )
    terminal_state = history[MAX_YEARS]
    target_length_m = float(cell["target_lia_length_m"])
    length_relative_error = relative_difference(
        terminal_state["length_m"], target_length_m
    )
    spinup_target_pass = bool(
        final_convergence["passed"] and length_relative_error <= LIA_LENGTH_REL_TOL
    )
    result = {
        "schema": "lia-all191-full900-spinup-cell-result-v10.262",
        "status": terminal_status,
        "technical_failure": False,
        "scientific_nonconvergence": not final_convergence["passed"],
        "terminal": True,
        "all900_completed": True,
        "fingerprint": cell["fingerprint"],
        "lia_unique_id": int(cell["lia_unique_id"]),
        "rgi_id": cell["rgi_id"],
        "mode": args.mode,
        "current_year": MAX_YEARS,
        "first_pass_boundary_year": first_pass_year,
        "search_stage": cell["search_stage"],
        "search_round": int(cell["search_round"]),
        "delta_t_eq_degC": float(cell["delta_t_eq_degC"]),
        "target_lia_length_m": target_length_m,
        "terminal_length_m": float(terminal_state["length_m"]),
        "lia_length_relative_error": length_relative_error,
        "lia_length_relative_tolerance": LIA_LENGTH_REL_TOL,
        "spinup_target_pass": spinup_target_pass,
        "final_year900_convergence": final_convergence,
        "boundary_evaluations": boundary_evaluations,
        "checkpoint_interval_years": CHECKPOINT_INTERVAL_YEARS,
        "checkpoint_years_expected": EXPECTED_CHECKPOINT_YEARS,
        "checkpoint_years_actual": actual_checkpoint_years,
        "checkpoint_contract_pass": all(
            item.get("restore", item.get("resume_entry_restore", {})).get("passed")
            is True
            for item in checks
        ),
        "domain_flags": domain_flags,
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
            "schema": "lia-all191-full900-spinup-cell-result-v10.262",
            "status": "technical-failure",
            "technical_failure": True,
            "scientific_nonconvergence": False,
            "terminal": True,
            "all900_completed": False,
            "fingerprint": args.fingerprint,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        atomic_json(output_dir / "result.json", failure)
        print(json.dumps(failure, indent=2, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
