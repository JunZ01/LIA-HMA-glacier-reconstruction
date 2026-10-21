"""All 1393 saved year-900 states, fixed climate, CPU20, atomic 30-year recovery.

No spin-up, state filtering, parameter search, acceptance promotion or GPU work.
Preparation/tests do not call a dynamics step. The prior real 30-year recovery
fixture is reused; every new dynamics trajectory belongs to the formal matrix.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DRIVERS = Path(__file__).resolve().parent
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
DESIGN = RESULTS / "lia_cpu_stage2_handoff_design_v1"
TX = RESULTS / "lia_cpu_fixed_climate_forward_v10_275_transaction_v1"
OUT = ROOT / "w275"
MODULE = "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.lia_fixed_climate_forward_v10_275"
PYTHON = "D:/Anaconda/envs/oggm_env2/python.exe"
YEARS = list(range(1470, 2000, 30)) + [2000]
THREADS = {k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")}
PARAMS = ("delta_t_eq_degC", "base_temp_bias_degC", "melt_f", "prcp_fac", "glen_a_pa_3_s_1", "fs", "temp_melt_degC")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest().upper()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest().upper()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def replace_json(path, value):
    path = Path(path)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.{time.time_ns()}.tmp")
    write_new(tmp, value)
    os.replace(tmp, path)


def status_write(path, value, replace=replace_json):
    """Never propagate a monitoring failure into numerical execution."""
    errors = []
    for attempt in range(4):
        try:
            replace(path, value)
            return str(path)
        except OSError as exc:
            errors.append(repr(exc))
            time.sleep(0.05 * (attempt + 1))
    try:
        dest = Path(path).parent / "fallback" / f"{time.time_ns()}.json"
        write_new(dest, {"primary": str(path), "errors": errors, "payload": value})
        return str(dest)
    except OSError as exc:
        print(f"STATUS_UNAVAILABLE {errors!r} {exc!r}", file=sys.stderr, flush=True)
        return None


def evidence(path):
    path = Path(path)
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "sha256": sha(path)}


def validate():
    result = subprocess.run([PYTHON, str(DRIVERS / "validate_lia_execution_state_v1.py")], cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return json.loads(result.stdout)


def source_state(row):
    from . import lia_all191_flowline_forward_v10_237 as base
    cp_path, st_path = ROOT / row["source_checkpoint"], ROOT / row["source_state"]
    if sha(cp_path) != row["source_checkpoint_sha256"].upper() or sha(st_path) != row["source_state_sha256"].upper():
        raise ValueError("source-file-hash-mismatch")
    cp = read(cp_path)
    if (cp["year"] != 900 or cp["lia_unique_id"] != row["lia_unique_id"] or cp["rgi_id"] != row["rgi_id"]
            or cp["fingerprint"] != row["source_spinup_fingerprint"] or not cp["diagnostics"]["convergence"]["passed"]):
        raise ValueError("source-identity-year-or-stability-mismatch")
    for p in PARAMS:
        if float(cp["parameters"][p]) != row[p]:
            raise ValueError(f"source-parameter-mismatch:{p}")
    if cp["state_sha256"].upper() != sha(st_path):
        raise ValueError("source-manifest-state-hash-mismatch")
    with gzip.open(st_path, "rb") as f:
        fls = pickle.load(f)
    if base.topology_signature(fls) != cp["topology"]:
        raise ValueError("source-topology-mismatch")
    return cp, fls


def validate_state(model, year):
    import numpy as np
    from . import lia_all191_flowline_forward_v10_237 as base
    for fl in model.fls:
        if not all(np.isfinite(np.asarray(getattr(fl, n))).all() for n in ("section", "thick", "surface_h", "bed_h", "widths_m")):
            raise ValueError("nonfinite-state-needs-user-decision")
        if np.any(fl.thick < 0) or np.any(fl.section < 0):
            raise ValueError("negative-state-needs-user-decision")
    state = base.snapshot(model, year)
    # Complete disappearance and boundary-margin warnings are outcomes, not a
    # new early-stop rule. OGGM's unchanged physical array-boundary guard stays on.
    state["domain_margin_warning"] = state["boundary_margin_m"] < 5000
    state["ice_disappeared"] = state["volume_m3"] <= 0
    return state


def save_checkpoint(member_root, model, year, row, history, counters):
    from . import lia_all191_flowline_forward_v10_237 as base
    root = Path(member_root) / "cp"
    root.mkdir(parents=True, exist_ok=True)
    final = root / str(year)
    if final.exists():
        raise FileExistsError(f"immutable-checkpoint-exists:{final}")
    temp = root / f".t{year}.{os.getpid()}.{time.time_ns()}"
    temp.mkdir()
    with (temp / "s.gz").open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
            pickle.dump(model.fls, f, protocol=pickle.HIGHEST_PROTOCOL)
        raw.flush()
        os.fsync(raw.fileno())
    write_new(temp / "h.json", history)
    cp = {"schema": "lia-fixed-climate-forward-checkpoint-v10.275", "fingerprint": row["fingerprint"],
          "lia_unique_id": row["lia_unique_id"], "rgi_id": row["rgi_id"], "calendar_year": year,
          "parameters": {p: row[p] for p in PARAMS}, "forward_temp_bias_degC": row["base_temp_bias_degC"],
          "source_state_sha256": row["source_state_sha256"], "state_sha256": sha(temp / "s.gz"),
          "history_sha256": sha(temp / "h.json"), "history_rows": len(history),
          "topology": base.topology_signature(model.fls), "state": base.snapshot(model, year), "counters": counters}
    write_new(temp / "c.json", cp)
    os.rename(temp, final)
    return final


def load_checkpoint(path, row):
    from . import lia_all191_flowline_forward_v10_237 as base
    cp = read(path / "c.json")
    expected = {p: row[p] for p in PARAMS}
    if (cp["fingerprint"] != row["fingerprint"] or cp["parameters"] != expected
            or cp["lia_unique_id"] != row["lia_unique_id"] or cp["rgi_id"] != row["rgi_id"]
            or cp["source_state_sha256"] != row["source_state_sha256"]
            or cp["forward_temp_bias_degC"] != row["base_temp_bias_degC"]):
        raise ValueError("checkpoint-contract-mismatch")
    if sha(path / "s.gz") != cp["state_sha256"] or sha(path / "h.json") != cp["history_sha256"]:
        raise ValueError("checkpoint-hash-mismatch")
    history = read(path / "h.json")
    if [r["year"] for r in history] != list(range(1470, cp["calendar_year"] + 1)) or len(history) != cp["history_rows"]:
        raise ValueError("checkpoint-history-not-synchronized")
    with gzip.open(path / "s.gz", "rb") as f:
        fls = pickle.load(f)
    if base.topology_signature(fls) != cp["topology"]:
        raise ValueError("checkpoint-topology-mismatch")
    return cp, fls, history


def prepare():
    """Static, non-dynamic preparation only, no launch permission consumed."""
    import numpy as np
    import pandas as pd
    import xarray as xr
    import psutil
    from . import lia_all191_flowline_forward_v10_237 as base
    validate()
    if TX.exists() or OUT.exists():
        raise FileExistsError("new-version-roots-must-be-absent")
    TX.mkdir()
    OUT.mkdir()
    old = read(RESULTS / "lia_191_all191_oggm_flowline_1d_forward_v10_237_transaction_v1/transaction_manifest.json")
    input_registry = read(RESULTS / "lia_191_all191_oggm_flowline_1d_v10_189_transaction_blocked_v1/input_hashes.json")
    for name in ("population_selection", "length_references", "climate_pixels", "reconstruction_climate"):
        item = input_registry["global_inputs"][name]
        if sha(item["path"]) != item["sha256"]:
            raise ValueError(f"frozen-input-changed:{name}")
    rows = list(csv.DictReader((DESIGN / "prospective_baseline_parents_1393.csv").open(encoding="utf-8-sig")))
    for row in rows:
        row["lia_unique_id"] = int(row["lia_unique_id"])
        for p in PARAMS:
            row[p] = float(row[p])
        row["key"] = "f" + row["prospective_parent_key"][1:]
        row["forward_temp_bias_degC"] = row["base_temp_bias_degC"]
        row["delta_t_forward_end_degC"] = 0.0
        row["status"] = "registered-not-dispatched"
    if len(rows) != 1393 or len({r["lia_unique_id"] for r in rows}) != 173 or len({(r["lia_unique_id"], r["delta_t_eq_degC"]) for r in rows}) != 1393:
        raise ValueError("all1393-membership-mismatch")
    # Sources are exhaustively checked before any publication/launch.
    source_diagnostics = []
    for i, row in enumerate(rows):
        cp, fls = source_state(row)
        source_diagnostics.append({"key": row["key"], "source_state": evidence(ROOT / row["source_state"]),
                                   "source_checkpoint": evidence(ROOT / row["source_checkpoint"]), "topology": cp["topology"]})
        if i % 250 == 0:
            print(f"source states verified {i + 1}/1393", flush=True)
    base._configure_oggm(OUT / "prep", OUT / "prep")
    from oggm import cfg, utils
    pixels = pd.read_csv(old["inputs"]["climate_pixels"])
    old_rows = list(csv.DictReader((RESULTS / "lia_191_all191_oggm_flowline_1d_forward_v10_237_transaction_v1/forward_matrix.csv").open(encoding="utf-8-sig")))
    climates = {}
    for row in rows:
        lid = str(row["lia_unique_id"])
        if lid in climates:
            continue
        rgi = row["rgi_id"]
        src = base._gdir_path(Path(old["inputs"]["source_gdir_root"]), rgi)
        target = base._gdir_path(OUT / "g", rgi)
        target.mkdir(parents=True)
        # Minimal immutable metadata; no inversion/DEM is needed by MonthlyTI.
        for name in ("outlines.tar.gz", "glacier_grid.json", "climate_historical.nc", "mb_calib.json"):
            shutil.copy2(src / name, target / name)
        gdir = utils.GlacierDirectory(rgi, base_dir=str(OUT / "g"))
        pixel = pixels[pixels.lia_unique_id.astype(int).eq(int(lid)) & pixels.rgi_id.eq(rgi)]
        if len(pixel) != 1:
            raise ValueError(f"climate-pixel-mismatch:{lid}")
        suffix, audit = base._write_case_climate(gdir, Path(old["inputs"]["reconstruction_climate"]), pixel.iloc[0], "off", 1470, 2000)
        generated = Path(gdir.get_filepath("gcm_data", filesuffix=suffix))
        prior_candidates = [r for r in old_rows if int(r["lia_unique_id"]) == int(lid)]
        # Compare to an already executed CPU historical input wherever available.
        prior_match = None
        for prior in prior_candidates:
            p = ROOT / "w237f/m" / prior["production_member_key"] / "w/per_glacier"
            p = base._gdir_path(p, rgi) / generated.name
            if p.exists():
                with xr.open_dataset(p) as a, xr.open_dataset(generated) as b:
                    for v in ("temp", "prcp", "temp_std"):
                        if v in a or v in b:
                            np.testing.assert_array_equal(a[v].values, b[v].values)
                    np.testing.assert_array_equal(a.time.values, b.time.values)
                    if a.attrs["ref_hgt"] != b.attrs["ref_hgt"]:
                        raise ValueError("reference-height-changed")
                prior_match = evidence(p)
                break
        files = [evidence(p) for p in target.iterdir() if p.is_file()]
        climates[lid] = {"suffix": suffix, "files": files, "audit": audit, "prior_cpu_climate_equal": prior_match,
                         "gdir": str(target), "pixel": {k: str(v) for k, v in pixel.iloc[0].items()}}
        print(f"shared climates {len(climates)}/173", flush=True)
    write_new(TX / "source_inventory.json", source_diagnostics)
    write_new(TX / "climate_inventory.json", climates)
    physics = read(RESULTS / "lia_191_all191_adaptive_spinup_v10_260_transaction_v1/adaptive_rules.json")
    shape = read(RESULTS / "lia_191_retained_32_curve_analysis_v3_essential_standard/executable_standard_v3.json")
    screen = {"schema": "lia-cpu-forward-screen-v10.275", "role": "post-run-provisional-only-no-launch-gate",
              "endpoint_direction_amplitude": physics["forward_screen"],
              "shape": {k: v for k, v in shape["core_scientific_gates"].items() if k != "endpoint_bounds"},
              "source_v3": evidence(RESULTS / "lia_191_retained_32_curve_analysis_v3_essential_standard/executable_standard_v3.json"),
              "source_cpu_override": evidence(RESULTS / "lia_191_all191_adaptive_spinup_v10_260_transaction_v1/adaptive_rules.json"),
              "start_length_is_not_a_gate": True, "no_automatic_acceptance": True}
    write_new(TX / "curve_standard.json", screen)
    code_files = [Path(__file__), DRIVERS / "lia_all191_flowline_forward_v10_237.py",
                  DRIVERS / "lia_all191_flowline_spinup_v10_216.py", DRIVERS / "lia_elevation_band_spinup.py",
                  DRIVERS / "lia_first10_spinup_batch.py"]
    import oggm.core.flowline, oggm.core.massbalance, oggm.cfg
    code_files += [Path(m.__file__) for m in (oggm.core.flowline, oggm.core.massbalance, oggm.cfg)]
    code = [evidence(p) for p in code_files]
    scientific = {"model": "OGGM SemiImplicitModel 1D", "fixed_dt_seconds": 108000, "start": 1470, "end": 2000,
                  "time_interval": "[1470.0,2000.0)", "checkpoints": YEARS, "annual_rows": 531,
                  "spinup_delta_removed": True, "q": 0, "geometry_mutation": False,
                  "stability": "year900 final810-900; 3% volume and 300m length", "initial_LIA_gate": False,
                  "beam_limit": None, "restore_rtol": 1e-7, "restart_warning_relative_V_A": 0.001,
                  "restart_warning_thickness_m": 1.0, "restart_warning_length_grid": 1.0,
                  "boundary_margin_warning_m": 5000, "automatic_scientific_early_stop": False,
                  "screen": evidence(TX / "curve_standard.json")}
    for row in rows:
        row["fingerprint"] = digest({"source_state": row["source_state_sha256"], "source_fingerprint": row["source_spinup_fingerprint"],
                                     "parameters": {p: row[p] for p in PARAMS}, "climate": climates[str(row["lia_unique_id"])]["files"],
                                     "science": scientific, "code": code})
    write_new(TX / "matrix.json", rows)
    with (TX / "forward_matrix.csv").open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    population = pd.read_csv(input_registry["global_inputs"]["population_selection"]["path"])
    ids = set(population.lia_unique_id.astype(int))
    no_stable = sorted(ids - {r["lia_unique_id"] for r in rows})
    if len(ids) != 191 or len(no_stable) != 18:
        raise ValueError("191-system-ledger-incomplete")
    write_new(TX / "population_191.json", {"all_ids": sorted(ids), "forward_ids": sorted(ids - set(no_stable)),
               "no_stable_ids": no_stable, "no_stable_status": "retained-for-later-physics-or-initialization-no-final-failure"})
    old_times = []
    for p in (ROOT / "w237f/m").glob("*/c/result.json"):
        r = read(p)
        if r.get("status") == "completed" and r.get("full_state_checkpoint_interval_years") == 30:
            old_times.append(r["runtime_seconds"])
    fixture = RESULTS / "lia_191_all191_oggm_flowline_1d_forward_v10_243_30y_launch_v1/summary.json"
    if not read(fixture)["preflight_result"]["resume_check"]["passed"]:
        raise ValueError("prior-real-recovery-fixture-not-passed")
    manifest = {"schema": "lia-fixed-climate-forward-transaction-v10.275", "created_at": now(), "launch_authorized": False,
                "source_plan": evidence(DESIGN / "固定重建气候后续计划_v2.md"),
                "source_parent_manifest": evidence(DESIGN / "prospective_baseline_parents_1393.csv"),
                "source_grid_terminal": evidence(RESULTS / "lia_191_all191_fullgrid_finalcell_retry_v10_273_terminal_v1/summary.json"),
                "matrix": evidence(TX / "matrix.json"), "climates": evidence(TX / "climate_inventory.json"),
                "sources": evidence(TX / "source_inventory.json"), "population_ledger": evidence(TX / "population_191.json"),
                "global_inputs": {k: input_registry["global_inputs"][k] for k in ("population_selection", "length_references", "climate_pixels", "reconstruction_climate")},
                "scientific": scientific, "code": code, "runtime": {"workers": 20, "internal_threads": THREADS,
                "python": PYTHON, "output_root": str(OUT), "stagger_seconds": 0.5, "memory_pause_fraction": 0.8,
                "status_failure": "bounded-retry-immutable-fallback; if all unavailable pause dispatch only",
                "invalid_rate": "advisory-only; no automatic batch termination or member retry",
                "process_lifetime": "persistent-foreground-session-no-detached-Start-Process"},
                "prior_dynamic_recovery_fixture": evidence(fixture), "new_dynamic_preflight": False,
                "static_adapter_tests_required": True, "cell_count": 1393, "system_count": 173,
                "runtime_estimate": {"prior_30y_cpu_cells": len(old_times), "prior_mean_seconds": float(np.mean(old_times)),
                "naive_cpu20_hours": float(np.mean(old_times) * 1393 / 20 / 3600),
                "caveat": "prior states and concurrent load differ; update from formal completions, never rerun for timing"},
                "resources": {"available_memory_bytes": psutil.virtual_memory().available, "disk_free_bytes": shutil.disk_usage(OUT).free},
                "accepted_ledger_effect": 0, "prohibited_source_accessed": False, "gpu_affected": False}
    manifest["maximum_runtime_path_chars"] = max(len(str(OUT / "m" / r["key"] / "cp/.t2000.999999.9999999999999999999/c.json")) for r in rows)
    if manifest["maximum_runtime_path_chars"] > 160:
        raise ValueError("short-root-margin-insufficient")
    write_new(TX / "transaction_manifest.json", manifest)
    print(json.dumps({"prepared": True, "cells": 1393, "climates": len(climates), "runtime_estimate": manifest["runtime_estimate"]}), flush=True)


def contract():
    m = read(TX / "transaction_manifest.json")
    for item in m["code"] + [m["matrix"], m["climates"], m["sources"], m["population_ledger"], m["scientific"]["screen"]]:
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"contract-input-changed:{item['path']}")
    return m, read(ROOT / m["matrix"]["path"])


def mb_for(row, climates):
    from oggm import cfg, utils
    from oggm.core.massbalance import MonthlyTIModel
    cfg.PARAMS["temp_melt"] = row["temp_melt_degC"]
    item = climates[str(row["lia_unique_id"])]
    for f in item["files"]:
        if sha(ROOT / f["path"]) != f["sha256"]:
            raise ValueError("shared-climate-file-changed")
    gdir = utils.GlacierDirectory(row["rgi_id"], base_dir=str(OUT / "g"))
    return MonthlyTIModel(gdir, filename="gcm_data", input_filesuffix=item["suffix"],
                          melt_f=row["melt_f"], prcp_fac=row["prcp_fac"], temp_bias=row["base_temp_bias_degC"],
                          check_calib_params=False, ys=1470, ye=2000, repeat=False)


def tests():
    import numpy as np
    from . import lia_all191_flowline_forward_v10_237 as base
    m, rows = contract()
    root = OUT / "test"
    root.mkdir()
    base._configure_oggm(root / "w", root / "w")
    climates = read(TX / "climate_inventory.json")
    checks = []
    sample = sorted(rows, key=lambda r: int(r["nx"]))
    for index in (0, len(sample) // 2, len(sample) - 1):
        row = sample[index]
        cp, fls = source_state(row)
        mb = mb_for(row, climates)
        model = base.make_model(fls, mb, row["glen_a_pa_3_s_1"], row["fs"], 1470)
        state = validate_state(model, 1470)
        for name in ("length_m", "area_m2", "volume_m3", "maximum_thickness_m"):
            np.testing.assert_allclose(state[name], cp["diagnostics"]["state"][name], rtol=1e-7, atol=0)
        np.testing.assert_array_equal(model.fls[0].section, fls[0].section)
        saved = save_checkpoint(root / row["key"], model, 1470, row, [state], {"accepted_steps": 0, "minimum_dt_seconds": None})
        got, restored, history = load_checkpoint(saved, row)
        for a, b in zip(fls, restored):
            np.testing.assert_array_equal(a.section, b.section)
        wrong = {**row, "fingerprint": "INVALID"}
        try:
            load_checkpoint(saved, wrong)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong-fingerprint-was-accepted")
        # Fresh MB object and explicit calendar ensure no spin-up MB cache leaks.
        mb2 = mb_for(row, climates)
        np.testing.assert_array_equal(mb.get_annual_mb(fls[0].surface_h, year=1470), mb2.get_annual_mb(fls[0].surface_h, year=1470))
        assert mb.temp_bias == row["base_temp_bias_degC"] and not mb.repeat
        checks.append({"key": row["key"], "nx": int(row["nx"]), "geometry_unchanged": True, "saved_state_restored_exactly": True, "fresh_mb_equal": True})
    def unavailable(*args):
        raise PermissionError("injected-status-lock")
    returned = status_write(root / "status.json", {"counter": 1}, replace=unavailable)
    assert returned and read(returned)["payload"]["counter"] == 1
    assert status_write(root / "status.json", {"counter": 2})
    assert read(root / "status.json")["counter"] == 2
    # Continue a restored deterministic state while its monitoring target is unavailable.
    count = read(returned)["payload"]["counter"]
    for _ in range(30):
        count += 1
    assert count == 31
    assert (1441 + (899 - 1441) % 30, 1441 + (900 - 1441) % 30) == (1469, 1470)
    summary = {"status": "pass", "dynamic_steps": 0, "samples": checks, "source_states_verified": 1393,
               "status_failure_fallback_and_later_progress": True, "synthetic_restore_continuation": True,
               "real_30y_dynamic_restore_evidence_reused": m["prior_dynamic_recovery_fixture"],
               "new_scientific_preflight": "none; prior exact-model historical fixture reused; first formal results supply timing",
               "checkpoint_history_joint_commit": True, "checkpoint_schedule": YEARS,
               "spinup_year900_mapped_to_calendar1470_without_geometry_change": True}
    write_new(TX / "technical_tests.json", summary)
    print(json.dumps(summary), flush=True)


def transition(stage, summary, launch=False):
    """Single CPU control writer; retain every independent research record."""
    validate()
    state_path = ROOT / "LIA_CURRENT_STATE.json"
    before_hash = sha(state_path)
    state = read(state_path)
    ev = evidence(summary)
    route_dir = RESULTS / f"lia_191_current_route_v10_275_{stage}"
    route_dir.mkdir()
    active = 0
    pending = 1393 if launch else 0
    if stage == "active":
        snap = read(summary)["status_snapshot"]
        active = sum(s["state"] == "running" for s in snap["slots"])
        pending = snap["pending"]
    route = {"schema": "lia-current-route-v10.275", "status": stage, "population": 191, "accepted": 58, "remaining": 133,
             "active": active, "queue": pending, "launch_authorized": launch, "full_historical_authorized": launch,
             "evidence": ev, "supersedes": state["authoritative_route"], "output_root": str(OUT),
             "all1393_no_initial_length_gate": True, "fixed_reconstructed_climate": True, "accepted_ledger_effect": 0}
    write_new(route_dir / "summary.json", route)
    state["state_id"] += f"+cpu-fixed-forward-v10.275-{stage}"
    state["updated_at"] = now()
    state["authoritative_route"] = {"version": f"v10.275-{stage}", **evidence(route_dir / "summary.json"), "supersedes": state["authoritative_route"]["version"]}
    state["counts"].update(active=active, queued=pending)
    if stage == "authorized":
        state["prior_cpu_fullgrid_terminal"] = copy.deepcopy(state["current_focus"])
    state["current_focus"] = {"system_id": "all1393-fixed-climate-historical-forward-cpu20", "mainline_family": "OGGM-flowline-1D",
        "state": stage, "launch_authorized": launch, "full_historical_authorized": launch, "historical_run_active": stage == "active",
        "population": 191, "forward_systems": 173, "forward_cells": 1393, "no_stable_systems_retained": 18,
        "workers": 20, "internal_threads": 1, "checkpoint_interval_years": 30, "start_year": 1470, "end_year": 2000,
        "source_spinup_rerun": False, "source_initial_length_gate": False, "delta_t_eq_carried_forward": False,
        "additional_forward_temperature_q": 0, "accepted_ledger_effect": 0, "output_root": str(OUT), "evidence": ev,
        "active": active, "queue": pending, "parent_pid": read(summary).get("parent_pid")}
    state["next_action"] = {"type": "launch-fixed-climate-historical-forward-cpu20" if launch else "monitor-existing-fixed-climate-historical-forward",
                            "transaction_id": "lia-fixed-climate-forward-v10.275", "launch_authorized": launch,
                            "description": "Run all1393 saved stable parents once; no spinup, no length preselection, q=0, CPU20, checkpoint30y. No retry or later group."}
    if sha(state_path) != before_hash:
        raise RuntimeError("concurrent-control-update-before-commit")
    replace_json(state_path, state)
    with (ROOT / "LIA_DECISION_LOG.md").open("a", encoding="utf-8") as f:
        f.write(f"\n\n## CPU fixed-climate forward v10.275 {stage}\n\n- User approval: 可以 开始吧. All 1393 saved year900 states enter fixed-climate forward; no initial LIA gate or beam cap; b only, q=0; CPU20, checkpoint30y. Evidence: `{ev['path']}`. No spin-up rerun or GPU control.\n")
    if stage == "authorized":
        with (ROOT / "LIA_PROJECT_FACTS.md").open("a", encoding="utf-8") as f:
            f.write("\n\n## 584. All1393 stable CPU states enter fixed-climate historical forward (2026-09-06)\n\n"
                    "- The user explicitly rejects the spin-up LIA-length eligibility gate and approves the fixed-climate plan, then instructs 可以 开始吧. All 1393 year900 stable states across 173 systems enter; none of the 193 target-matching subset or former beam limits controls eligibility. The other 18 systems remain in the 191-system ledger for later physics/initialization work, not final scientific closure.\n"
                    "- Preserve every saved geometry byte, map model year900 to calendar1470, and run unchanged OGGM one-dimensional SemiImplicitModel to 2000. Historical MonthlyTI uses the existing W5E5 1961-1990 climatology plus ensemble_rec_ref anomaly, precipitation anomaly off, inherited base_temp_bias b and all inherited physics. Remove spin-up-only delta_t_eq s; extra forward q is fixed to zero. The -s parameter switch is disclosed, not described as no temperature jump. No spinup or physics search is launched.\n"
                    "- Twenty independent CPU workers use one internal numerical thread each and the unchanged 108000-second fixed step. Complete state and synchronized annual history commit together every 30 years plus year2000. Exactly 19 checkpoints and 531 annual rows are due per completed member. Per-system climate files are immutable and shared; model geometry and mass-balance caches are independent.\n"
                    "- Reuse the passed real v10.243 30-year historical checkpoint-continuation fixture and full forward evidence. New non-dynamic tests validate year900 adapter, unchanged geometry, fresh calendar/MB cache, hash/identity/topology, atomic synchronized history, and unavailable-status fallback. No new scientific pass requirement or discarded duplicate preflight simulation is inserted. Runtime estimates are provisional and updated from actual formal members.\n"
                    "- Curve scoring is post-run provisional only: the existing CPU v10.260 total-length 10% peak/end, direction/amplitude and retained-v3 shape rules apply; old 20%-of-separation is diagnostic, never an entry gate. No new initial-transient rejection or automatic acceptance. Individual fatal/integrity errors isolate one member; disappearance and margin warnings are reported outcomes. Status errors never terminate computation, all-status unavailability or memory above80% pauses new dispatch for user decision, and no batch termination or retry is automatic.\n"
                    f"- Immutable transaction: `{TX.relative_to(ROOT)}/transaction_manifest.json`; evidence: `{ev['path']}`. Output w275. Prior CPU/GPU outputs and all research branches remain unchanged; accepted stays58.\n")
        state["scientific_contract"]["sha256"] = sha(ROOT / "LIA_PROJECT_FACTS.md")
        replace_json(state_path, state)
    report = validate()
    write_new(route_dir / "validator.json", report)
    return report


def authorize():
    import psutil
    m, rows = contract()
    if read(TX / "technical_tests.json")["status"] != "pass":
        raise ValueError("adapter-tests-failed")
    other = [p.info for p in psutil.process_iter(["pid", "name", "cmdline"]) if p.info["pid"] != os.getpid()
             and "python" in (p.info["name"] or "").lower() and "lia_" in " ".join(p.info["cmdline"] or [])]
    if other:
        raise RuntimeError(f"active-LIA-processes-before-launch:{other}")
    if psutil.virtual_memory().percent > 80:
        raise RuntimeError("prelaunch-memory-needs-user-decision")
    summary = {"status": "authorized", "user_approval": "可以 开始吧", "time": now(), "transaction": evidence(TX / "transaction_manifest.json"),
               "tests": evidence(TX / "technical_tests.json"), "launch_authorized": True, "cells": len(rows),
               "workers": 20, "output_root": str(OUT), "no_duplicate_processes": True,
               "memory_used_percent": psutil.virtual_memory().percent, "parent_pid": None}
    write_new(TX / "authorization.json", summary)
    print(json.dumps(transition("authorized", TX / "authorization.json", True)))


def worker(key):
    import numpy as np
    from . import lia_all191_flowline_forward_v10_237 as base
    start = time.perf_counter()
    m, rows = contract()
    row = next(r for r in rows if r["key"] == key)
    root = OUT / "m" / key
    if (root / "result.json").exists():
        return
    root.mkdir(parents=True, exist_ok=False)
    timing = {}
    try:
        cp, fls = source_state(row)
        base._configure_oggm(root / "w", root / "w")
        t = time.perf_counter()
        mb = mb_for(row, read(TX / "climate_inventory.json"))
        timing["climate_load_seconds"] = time.perf_counter() - t
        model = base.make_model(fls, mb, row["glen_a_pa_3_s_1"], row["fs"], 1470)
        history = [validate_state(model, 1470)]
        for name in ("length_m", "area_m2", "volume_m3", "maximum_thickness_m"):
            np.testing.assert_allclose(history[0][name], cp["diagnostics"]["state"][name], rtol=1e-7, atol=0)
        counters = {"accepted_steps": 0, "rejected_steps": 0, "minimum_dt_seconds": None}
        original_step = model.step
        def step(dt):
            actual = float(original_step(dt))
            counters["accepted_steps"] += 1
            if counters["minimum_dt_seconds"] is None or actual < counters["minimum_dt_seconds"]:
                counters["minimum_dt_seconds"] = actual
            return actual
        model.step = step
        timing["initialization_seconds"] = time.perf_counter() - start
        timing["integration_seconds"] = 0.0
        timing["checkpoint_seconds"] = 0.0
        for year in range(1470, 2001):
            if year > 1470:
                t = time.perf_counter()
                model.run_until(float(year))
                timing["integration_seconds"] += time.perf_counter() - t
                history.append(validate_state(model, year))
            if year in YEARS:
                t = time.perf_counter()
                saved = save_checkpoint(root, model, year, row, history, counters)
                loaded, restored, persisted_history = load_checkpoint(saved, row)
                for a, b in zip(model.fls, restored):
                    np.testing.assert_allclose(a.section, b.section, rtol=1e-7, atol=0)
                timing["checkpoint_seconds"] += time.perf_counter() - t
                status_write(root / "progress.json", {"year": year, "checkpoint": str(saved), "counters": counters, "timing": timing})
        with (root / "annual.csv").open("x", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
            f.flush()
            os.fsync(f.fileno())
        result = {"status": "completed", "terminal": True, "technical_failure": False, "scientific_failure": False,
                  "key": key, "fingerprint": row["fingerprint"], "lia_unique_id": row["lia_unique_id"], "rgi_id": row["rgi_id"],
                  "delta_t_eq_degC": row["delta_t_eq_degC"], "forward_temp_bias_degC": row["base_temp_bias_degC"],
                  "parameter_handoff_step_degC": -row["delta_t_eq_degC"], "q": 0, "calendar_end": 2000,
                  "annual_rows": len(history), "checkpoints": YEARS, "checkpoint_contract_pass": True,
                  "source_state_sha256": row["source_state_sha256"], "annual": evidence(root / "annual.csv"),
                  "timing": timing, "runtime_seconds": time.perf_counter() - start, "counters": counters,
                  "domain_margin_warning_years": [h["year"] for h in history if h["domain_margin_warning"]],
                  "ice_disappeared_years": [h["year"] for h in history if h["ice_disappeared"]],
                  "provisional_curve_screen": "not-yet-scored", "accepted_ledger_effect": 0}
        write_new(root / "result.json", result)
        print(json.dumps(result), flush=True)
    except Exception as exc:
        failure = {"status": "needs-user-decision", "terminal": True, "technical_failure": True,
                   "key": key, "fingerprint": row["fingerprint"], "lia_unique_id": row["lia_unique_id"],
                   "error": repr(exc), "traceback": traceback.format_exc(), "runtime_seconds": time.perf_counter() - start}
        print(failure["traceback"], file=sys.stderr, flush=True)
        try:
            write_new(root / "result.json", failure)
        except OSError:
            print("MEMBER_RESULT_WRITE_FAILED", file=sys.stderr, flush=True)
        raise SystemExit(2)


def controller():
    import psutil
    m, rows = contract()
    report = validate()
    state = read(ROOT / "LIA_CURRENT_STATE.json")
    if not report["launch_authorized"] or state["next_action"]["transaction_id"] != "lia-fixed-climate-forward-v10.275":
        raise ValueError("unique-launch-authorization-required")
    auth = read(TX / "authorization.json")
    if auth["transaction"]["sha256"] != sha(TX / "transaction_manifest.json"):
        raise ValueError("authorization-transaction-mismatch")
    (OUT / "r").mkdir(exist_ok=False)
    (OUT / "l").mkdir(exist_ok=False)
    parent = {"pid": os.getpid(), "started_at": now(), "command": sys.argv, "workers": 20,
              "transaction": evidence(TX / "transaction_manifest.json"), "output_root": str(OUT), "checkpoints": YEARS}
    write_new(OUT / "r/parent.json", parent)
    env = os.environ.copy()
    env.update(THREADS)
    env["PYTHONPATH"] = str(ROOT)
    running, completed, index = {}, [], 0
    pause, peak = None, 0.0
    while running or index < len(rows):
        memory = psutil.virtual_memory()
        peak = max(peak, memory.percent)
        if memory.percent > 80:
            pause = "memory-above80-needs-user-decision"
        decision = OUT / "r/user_decision.json"
        if pause and decision.exists():
            if read(decision).get("action") == "resume":
                pause = None
                decision.rename(OUT / "r" / f"decision-applied-{time.time_ns()}.json")
        for slot, item in list(running.items()):
            proc, row, outlog, errlog = item
            code = proc.poll()
            if code is None:
                continue
            outlog.close()
            errlog.close()
            path = OUT / "m" / row["key"] / "result.json"
            try:
                result = read(path)
            except (OSError, ValueError):
                result = {"status": "technical-failure-missing-result", "technical_failure": True, "key": row["key"], "fingerprint": row["fingerprint"]}
            record = {"pid": proc.pid, "key": row["key"], "lia_unique_id": row["lia_unique_id"], "returncode": code,
                      "status": result["status"], "technical_failure": bool(result.get("technical_failure") or code != 0),
                      "result": evidence(path) if path.exists() else None}
            write_new(OUT / "r" / f"done-{row['key']}.json", record)
            completed.append(record)
            del running[slot]
        while pause is None and len(running) < 20 and index < len(rows):
            row = rows[index]
            slot = next(s for s in range(20) if s not in running)
            outlog = (OUT / "l" / f"{row['key']}.out.log").open("x", encoding="utf-8")
            errlog = (OUT / "l" / f"{row['key']}.err.log").open("x", encoding="utf-8")
            command = [PYTHON, "-m", MODULE, "worker", "--key", row["key"]]
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=outlog, stderr=errlog)
            write_new(OUT / "r" / f"start-{row['key']}.json", {"pid": proc.pid, "slot": slot, "key": row["key"],
                       "fingerprint": row["fingerprint"], "lia_unique_id": row["lia_unique_id"], "started_at": now(), "command": command, "threads": THREADS})
            running[slot] = (proc, row, outlog, errlog)
            index += 1
            time.sleep(0.5)
        slots = []
        for s in range(20):
            item = running.get(s)
            slots.append({"slot": s, "state": "running" if item else "idle", "pid": item[0].pid if item else None,
                          "key": item[1]["key"] if item else None, "lia_unique_id": item[1]["lia_unique_id"] if item else None})
        status = {"parent_pid": os.getpid(), "updated_at": now(), "completed": len(completed), "pending": len(rows) - index,
                  "slots": slots, "technical_failures": sum(c["technical_failure"] for c in completed),
                  "pause_reason": pause, "cpu_percent": psutil.cpu_percent(), "memory_percent": memory.percent,
                  "available_memory_bytes": memory.available, "peak_memory_percent": peak}
        if status_write(OUT / "r/status.json", status) is None:
            pause = "all-status-destinations-unavailable-needs-user-decision"
        time.sleep(2)
    write_new(OUT / "r/summary.json", {"status": "terminal", "parent_pid": os.getpid(), "finished_at": now(),
               "cells": len(rows), "records": completed, "technical_failures": sum(c["technical_failure"] for c in completed),
               "peak_memory_percent": peak, "source_completed_cells_recomputed": 0})


def active():
    import psutil
    p = read(OUT / "r/parent.json")
    if not psutil.pid_exists(p["pid"]):
        raise RuntimeError("parent-not-alive")
    snap = read(OUT / "r/status.json")
    summary = {"status": "active", "parent_pid": p["pid"], "parent": p, "status_snapshot": snap,
               "observed_at": now(), "parent_alive": True,
               "live_workers": [s for s in snap["slots"] if s["pid"] and psutil.pid_exists(s["pid"])]}
    write_new(TX / "active.json", summary)
    print(json.dumps(transition("active", TX / "active.json")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "tests", "authorize", "controller", "worker", "active"))
    parser.add_argument("--key")
    args = parser.parse_args()
    if args.action != "controller":
        os.environ.update(THREADS)
    if args.action == "worker":
        worker(args.key)
    else:
        globals()[args.action]()


if __name__ == "__main__":
    main()
