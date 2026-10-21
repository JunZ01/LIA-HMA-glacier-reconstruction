"""Continue one validated v10.216 year-3000 state to year 6000/6030."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import traceback
from pathlib import Path

import pandas as pd

from . import lia_all191_flowline_spinup_v10_216 as base


START_YEAR = 3000
MAX_BOUNDARY_YEAR = 6000


def load_cell(transaction_dir: Path, fingerprint: str) -> tuple[dict, dict, dict]:
    manifest = json.loads((transaction_dir / "transaction_manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("launch_authorized") or not manifest.get("extension_authorized"):
        raise ValueError("extension-transaction-not-authorized")
    if manifest["runtime"]["driver_sha256"] != base.sha256(Path(__file__)):
        raise ValueError("extension-driver-hash-mismatch")
    with (transaction_dir / "extension_matrix.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["fingerprint"] == fingerprint]
    if len(rows) != 1:
        raise ValueError(f"extension-fingerprint-row-count:{len(rows)}")
    row = rows[0]
    for key in (
        "lia_unique_id", "delta_t_eq_degC", "base_temp_bias_degC", "melt_f", "prcp_fac",
        "glen_a_pa_3_s_1", "fs", "temp_melt_degC",
    ):
        row[key] = int(row[key]) if key == "lia_unique_id" else float(row[key])
    members = manifest["extension"]["members"]
    matches = [item for item in members if item["fingerprint"] == fingerprint]
    if len(matches) != 1 or len(members) != int(manifest["extension"]["cell_count"]):
        raise ValueError("extension-member-map-failure")
    return manifest, row, matches[0]


def member_paths(manifest: dict, member: dict) -> tuple[Path, Path, Path]:
    root = Path(manifest["runtime"]["output_root"]).resolve()
    member_root = (root / "m" / member["key"]).resolve()
    member_root.relative_to(root)
    return member_root, member_root / "w", member_root / "c"


def source_cell(cell: dict) -> dict:
    out = dict(cell)
    out["fingerprint"] = cell["source_fingerprint"]
    return out


def classify(convergence: dict) -> dict:
    volume_pass = (
        all(float(value) <= base.VOLUME_CYCLE_TOL for value in convergence["volume_relative_cycle_differences"])
        and float(convergence["volume_relative_90y_drift"]) <= base.VOLUME_90_TOL
    )
    length_pass = (
        all(float(value) <= base.LENGTH_GRID_TOL for value in convergence["length_grid_cycle_differences"])
        and float(convergence["length_grid_90y_difference"]) <= base.LENGTH_GRID_TOL
    )
    if volume_pass and not length_pass:
        label = "volume-stable-length-moving"
    elif length_pass and not volume_pass:
        label = "length-stable-volume-changing"
    elif not volume_pass and not length_pass:
        label = "volume-and-length-still-changing"
    else:
        label = "converged"
    return {"volume_gate_passed": volume_pass, "length_gate_passed": length_pass, "classification": label}


def run(args) -> dict:
    started = time.perf_counter()
    manifest, cell, member = load_cell(args.transaction_dir, args.fingerprint)
    expected_root = Path(manifest["runtime"]["output_root"])
    if args.output_root.resolve() != expected_root.resolve():
        raise ValueError("extension-output-root-mismatch")
    _, workdir, output_dir = member_paths(manifest, member)
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(f"extension-result-collision:{result_path}")
    output_dir.mkdir(parents=True, exist_ok=False)

    source_result = Path(cell["source_result_path"])
    source_checkpoint = Path(cell["source_checkpoint_path"])
    if base.sha256(source_result) != cell["source_result_sha256"]:
        raise ValueError("source-result-hash-mismatch")
    source_payload = json.loads(source_result.read_text(encoding="utf-8"))
    if not (
        source_payload.get("status") == "max-duration-nonconverged"
        and int(source_payload.get("current_year", -1)) == START_YEAR
        and source_payload.get("fingerprint") == cell["source_fingerprint"]
    ):
        raise ValueError("source-result-not-eligible")
    if base.sha256(source_checkpoint / "checkpoint.json") != cell["source_checkpoint_manifest_sha256"]:
        raise ValueError("source-checkpoint-manifest-hash-mismatch")

    base._configure_oggm(workdir, output_dir)
    from oggm import cfg, workflow
    from oggm.core.massbalance import MonthlyTIModel

    cfg.PARAMS["temp_melt"] = float(cell["temp_melt_degC"])
    source_root = Path(manifest["inputs"]["source_gdir_root"])
    climate_path = Path(manifest["inputs"]["reconstruction_climate"])
    pixels = pd.read_csv(manifest["inputs"]["climate_pixels"])
    pixel = pixels[
        pixels["lia_unique_id"].astype(int).eq(int(cell["lia_unique_id"]))
        & pixels["rgi_id"].astype(str).eq(cell["rgi_id"])
        & pixels["climate_id"].astype(str).eq("ensemble_rec_ref")
    ]
    if len(pixel) != 1:
        raise ValueError(f"climate-pixel-row-count:{len(pixel)}")
    source_gdir = base._gdir_path(source_root, cell["rgi_id"])
    target_gdir = base._gdir_path(workdir / "per_glacier", cell["rgi_id"])
    base._clone_gdir(source_gdir, target_gdir)
    gdir = workflow.init_glacier_directories([cell["rgi_id"]], reset=False)[0]
    suffix, climate_audit = base._write_case_climate(
        gdir, climate_path, pixel.iloc[0], "off", start_year=1441, end_year=1470,
    )
    mb_model = MonthlyTIModel(
        gdir, filename="gcm_data", input_filesuffix=suffix,
        melt_f=float(cell["melt_f"]), prcp_fac=float(cell["prcp_fac"]),
        temp_bias=float(cell["base_temp_bias_degC"] + cell["delta_t_eq_degC"]),
        check_calib_params=False, ys=1441, ye=1470, repeat=True,
    )

    checkpoint_manifest, fls = base.load_checkpoint(source_checkpoint, source_cell(cell))
    if int(checkpoint_manifest["year"]) != START_YEAR:
        raise ValueError("source-checkpoint-year-mismatch")
    model = base.make_model(fls, mb_model, cell["glen_a_pa_3_s_1"], START_YEAR)
    restored = base.make_model(fls, mb_model, cell["glen_a_pa_3_s_1"], START_YEAR)
    source_restore = base.state_restore_check(model, restored)
    history = {START_YEAR: base.snapshot(model, START_YEAR)}
    checks = []
    continuation_anomalies = []
    current_year = START_YEAR
    status = None
    first_pass_year = None
    final_classification = None

    for boundary in range(START_YEAR + base.BLOCK_YEARS, MAX_BOUNDARY_YEAR + 1, base.BLOCK_YEARS):
        for year in range(current_year + base.CYCLE_YEARS, boundary + 1, base.CYCLE_YEARS):
            model.run_until(float(year))
            history[year] = base.snapshot(model, year)
        current_year = boundary
        state = history[current_year]
        if not all(math.isfinite(state[key]) for key in ("volume_m3", "area_m2", "length_m", "minimum_thickness_m")):
            raise ValueError("nonfinite-state")
        if state["volume_m3"] <= 0 or state["area_m2"] <= 0 or state["length_m"] <= 0 or state["minimum_thickness_m"] < 0:
            status = "scientific-domain-failure-nonpositive-or-negative-state"
        if state["boundary_margin_m"] < base.BOUNDARY_MARGIN_M:
            status = "scientific-domain-failure-boundary-margin"
        convergence = base.convergence_metrics(history, current_year)
        checkpoint = base.save_checkpoint(output_dir, model, current_year, "extension-boundary", cell, {
            "state": state, "convergence": convergence,
        })
        base.atomic_history(output_dir / "diagnostics.jsonl", history)
        _, restored_fls = base.load_checkpoint(checkpoint, cell)
        restored = base.make_model(restored_fls, mb_model, cell["glen_a_pa_3_s_1"], current_year)
        restore = base.state_restore_check(model, restored)
        check = {"year": current_year, "convergence": convergence, "restore": restore}
        checks.append(check)
        if status is not None:
            break
        if convergence["passed"]:
            first_pass_year = current_year
            continuation, restored = base.continuation_check(model, restored, current_year + base.CYCLE_YEARS)
            next_model = restored
            if not continuation["passed"]:
                continuation["action"] = "warning-continue-uninterrupted-model"
                continuation_anomalies.append(dict(continuation))
                next_model = model
            current_year += base.CYCLE_YEARS
            history[current_year] = base.snapshot(next_model, current_year)
            confirmation = base.convergence_metrics(history, current_year)
            check["continuation"] = continuation
            check["rolling_30y_confirmation"] = confirmation
            if confirmation["passed"]:
                model = next_model
                base.save_checkpoint(output_dir, model, current_year, "confirmed-convergence", cell, {
                    "state": history[current_year], "confirmation": confirmation,
                })
                status = "converged"
                final_classification = classify(confirmation)
                break
            model = next_model
        if boundary == MAX_BOUNDARY_YEAR:
            status = "max-duration-nonconverged"
            final_classification = classify(checks[-1]["convergence"])
            break

    base.atomic_history(output_dir / "diagnostics.jsonl", history)
    result = {
        "schema": "lia-all191-flowline-extension-result-v10.220",
        "status": status,
        "technical_failure": False,
        "scientific_nonconvergence": status == "max-duration-nonconverged",
        "terminal": True,
        "fingerprint": cell["fingerprint"],
        "source_fingerprint": cell["source_fingerprint"],
        "lia_unique_id": int(cell["lia_unique_id"]),
        "rgi_id": cell["rgi_id"],
        "delta_t_eq_degC": float(cell["delta_t_eq_degC"]),
        "start_year": START_YEAR,
        "current_year": current_year,
        "first_pass_year": first_pass_year,
        "source_restore": source_restore,
        "continuation_anomaly_count": len(continuation_anomalies),
        "continuation_anomalies": continuation_anomalies,
        "final_classification": final_classification,
        "checks": checks,
        "climate": climate_audit,
        "runtime_seconds": time.perf_counter() - started,
    }
    base.atomic_json(result_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def failure_dir(args) -> Path:
    try:
        manifest = json.loads((args.transaction_dir / "transaction_manifest.json").read_text(encoding="utf-8"))
        member = next(item for item in manifest["extension"]["members"] if item["fingerprint"] == args.fingerprint)
        return member_paths(manifest, member)[2]
    except Exception:
        return args.output_root / "x" / args.fingerprint[:8]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transaction-dir", required=True, type=Path)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("extension",), required=True)
    args = parser.parse_args()
    try:
        result = run(args)
        return 0 if not result["technical_failure"] else 2
    except Exception as exc:
        output = failure_dir(args)
        output.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema": "lia-all191-flowline-extension-result-v10.220",
            "status": "technical-failure",
            "technical_failure": True,
            "scientific_nonconvergence": False,
            "terminal": True,
            "fingerprint": args.fingerprint,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        base.atomic_json(output / "result.json", failure)
        print(json.dumps(failure, indent=2, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
