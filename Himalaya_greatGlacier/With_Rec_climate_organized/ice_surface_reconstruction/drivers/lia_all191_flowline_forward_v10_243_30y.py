"""1470-2000 OGGM forward member with annual diagnostics and 30-year restart states."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import time
import traceback
from pathlib import Path

import pandas as pd

from . import lia_all191_flowline_forward_v10_237 as base


START_YEAR = 1470
END_YEAR = 2000
PREFLIGHT_END_YEAR = 1530
CHECKPOINT_INTERVAL_YEARS = 30


def load_contract(transaction_dir: Path, fingerprint: str, mode: str):
    manifest = base.read_json(transaction_dir / "transaction_manifest.json")
    if not manifest.get("launch_authorized"):
        raise ValueError("transaction-not-launch-authorized")
    if base.sha256(Path(__file__)) != manifest["runtime"]["worker_sha256"]:
        raise ValueError("worker-hash-mismatch")
    with (transaction_dir / "forward_matrix.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["forward_fingerprint"] == fingerprint]
    if len(rows) != 1:
        raise ValueError(f"forward-fingerprint-row-count:{len(rows)}")
    row = rows[0]
    selected = manifest["preflight"]["fingerprints"] if mode == "preflight" else manifest["production"]["fingerprints"]
    if fingerprint not in set(selected):
        raise ValueError(f"fingerprint-not-authorized-for-{mode}")
    for key in ("forward_candidate_ordinal", "cell_ordinal", "lia_unique_id", "original_strict_confirmation_state_year"):
        row[key] = int(row[key])
    for key in ("delta_t_eq_degC", "base_temp_bias_degC", "effective_temp_bias_degC",
                "melt_f", "prcp_fac", "glen_a_pa_3_s_1", "fs", "temp_melt_degC"):
        row[key] = float(row[key])
    return manifest, row


def member_paths(manifest: dict, row: dict, mode: str):
    root_key = "preflight_root" if mode == "preflight" else "unified_short_root"
    root = Path(manifest["runtime"][root_key]).resolve()
    key = row["preflight_member_key"] if mode == "preflight" else row["production_member_key"]
    group = "p" if mode == "preflight" else "m"
    member_root = (root / group / key).resolve()
    member_root.relative_to(root)
    return key, member_root, member_root / "w", member_root / "c"


def run(args) -> dict:
    started = time.perf_counter()
    manifest, row = load_contract(args.transaction_dir, args.fingerprint, args.mode)
    root_key = "preflight_root" if args.mode == "preflight" else "unified_short_root"
    expected_root = Path(manifest["runtime"][root_key]).resolve()
    if args.output_root.resolve() != expected_root:
        raise ValueError("output-root-contract-mismatch")
    key, _, workdir, output_dir = member_paths(manifest, row, args.mode)
    result_path = output_dir / "result.json"
    if result_path.exists() and base.read_json(result_path).get("terminal"):
        return base.read_json(result_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = Path(row["confirmation_checkpoint_manifest_path"])
    source_state = Path(row["confirmation_checkpoint_state_path"])
    if base.sha256(source_manifest) != row["confirmation_checkpoint_manifest_sha256"]:
        raise ValueError("source-checkpoint-manifest-hash-mismatch")
    if base.sha256(source_state) != row["confirmation_checkpoint_state_sha256"]:
        raise ValueError("source-checkpoint-state-hash-mismatch")
    source_payload = base.read_json(source_manifest)
    if int(source_payload["lia_unique_id"]) != row["lia_unique_id"] or source_payload["rgi_id"] != row["rgi_id"]:
        raise ValueError("source-checkpoint-member-mismatch")
    with gzip.open(source_state, "rb") as handle:
        initial_fls = pickle.load(handle)
    if base.topology_signature(initial_fls) != source_payload["topology"]:
        raise ValueError("source-checkpoint-topology-mismatch")

    base._configure_oggm(workdir, output_dir)
    from oggm import cfg, workflow
    from oggm.core.massbalance import MonthlyTIModel

    cfg.PARAMS["temp_melt"] = row["temp_melt_degC"]
    source_root = Path(manifest["inputs"]["source_gdir_root"])
    target = base._gdir_path(workdir / "per_glacier", row["rgi_id"])
    base._clone_gdir(base._gdir_path(source_root, row["rgi_id"]), target)
    gdir = workflow.init_glacier_directories([row["rgi_id"]], reset=False)[0]
    pixels = pd.read_csv(manifest["inputs"]["climate_pixels"])
    selected = pixels[
        pixels["lia_unique_id"].astype(int).eq(row["lia_unique_id"])
        & pixels["rgi_id"].astype(str).eq(row["rgi_id"])
        & pixels["climate_id"].astype(str).eq("ensemble_rec_ref")
    ]
    if len(selected) != 1:
        raise ValueError(f"climate-pixel-row-count:{len(selected)}")
    suffix, climate_audit = base._write_case_climate(
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
                item = json.loads(line)
                history[int(item["year"])] = item
    checkpoint_root = output_dir / "checkpoints"
    checkpoints = sorted(checkpoint_root.glob("year-*")) if checkpoint_root.exists() else []
    if checkpoints:
        current_year, fls = base.load_forward_checkpoint(checkpoints[-1], row)
        model = base.make_model(fls, mb_model, row["glen_a_pa_3_s_1"], row["fs"], current_year)
    else:
        current_year = START_YEAR
        model = base.make_model(initial_fls, mb_model, row["glen_a_pa_3_s_1"], row["fs"], START_YEAR)
        history[START_YEAR] = base.validate_state(model, START_YEAR)
        base.save_forward_checkpoint(output_dir, model, START_YEAR, row)
        base.atomic_history(history_path, history)

    target_year = PREFLIGHT_END_YEAR if args.mode == "preflight" else END_YEAR
    resume_check = None
    while current_year < target_year:
        next_year = current_year + 1
        model.run_until(float(next_year))
        current_year = next_year
        history[current_year] = base.validate_state(model, current_year)
        scheduled = current_year == target_year or (current_year - START_YEAR) % CHECKPOINT_INTERVAL_YEARS == 0
        if scheduled:
            base.save_forward_checkpoint(output_dir, model, current_year, row)
            base.atomic_history(history_path, history)
        if args.mode == "preflight" and current_year == START_YEAR + CHECKPOINT_INTERVAL_YEARS:
            prior = output_dir / "checkpoints" / f"year-{current_year:04d}"
            _, restored_fls = base.load_forward_checkpoint(prior, row)
            restored = base.make_model(restored_fls, mb_model, row["glen_a_pa_3_s_1"], row["fs"], current_year)
            resume_check = base.resume_comparison(model, restored, target_year)
            if not resume_check["passed"]:
                raise ValueError("preflight-30y-resume-comparison-failure")
            model = restored
            current_year = target_year
            history[current_year] = base.validate_state(model, current_year)
            base.save_forward_checkpoint(output_dir, model, current_year, row)
            base.atomic_history(history_path, history)

    result = {
        "schema": "lia-all191-flowline-forward-member-result-v10.243",
        "status": "preflight-pass" if args.mode == "preflight" else "completed",
        "terminal": True, "technical_failure": False, "scientific_failure": False,
        "mode": args.mode, "member_key": key, "forward_fingerprint": row["forward_fingerprint"],
        "source_forward_fingerprint": row["source_forward_fingerprint"],
        "lia_unique_id": row["lia_unique_id"], "rgi_id": row["rgi_id"],
        "delta_t_eq_degC": row["delta_t_eq_degC"], "start_year": START_YEAR,
        "end_year": current_year, "resume_check": resume_check, "climate": climate_audit,
        "runtime_seconds": time.perf_counter() - started,
        "full_state_checkpoint_interval_years": CHECKPOINT_INTERVAL_YEARS,
        "full_state_checkpoint_count": len(list(checkpoint_root.glob("year-*"))),
        "annual_diagnostics_retained": True,
    }
    base.atomic_json(result_path, result)
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
        run(args)
        return 0
    except Exception as exc:
        out = failure_dir(args)
        out.mkdir(parents=True, exist_ok=True)
        text = f"{type(exc).__name__}: {exc}"
        scientific = "scientific-domain-failure" in text
        failure = {
            "schema": "lia-all191-flowline-forward-member-result-v10.243",
            "status": "scientific-failure" if scientific else "technical-failure",
            "terminal": True, "technical_failure": not scientific,
            "scientific_failure": scientific, "forward_fingerprint": args.fingerprint,
            "error": text, "traceback": traceback.format_exc(),
        }
        base.atomic_json(out / "result.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=base.sys.stderr)
        return 3 if scientific else 2


if __name__ == "__main__":
    raise SystemExit(main())
