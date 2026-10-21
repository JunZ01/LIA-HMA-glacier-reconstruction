"""Build a prospective CPU melt-factor matrix from immutable JSON evidence only.
No OGGM imports, dynamics, model state deserialization, source writes or launch.
"""
from __future__ import annotations
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RES = ROOT / "validation_data/results/paleo_surface_validation_v0"
OUT = RES / "lia_cpu_melt_factor_design_v10_278_v1"
POL = RES / "lia_cpu_direction_and_fixed_geometry_mb_v10_277_v1"
FIELDS = ("base_temp_bias_degC", "melt_f", "prcp_fac", "glen_a_pa_3_s_1", "fs", "temp_melt_degC")
DELTAS = [i / 10 for i in range(-10, 11)]
MULTIPLIERS = [0.7, 1.3]

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest().upper()

def ref(path):
    p = Path(path)
    return {"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)}

def save(name, value):
    path = OUT / name
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return ref(path)

def distribution(values):
    vals = sorted(values)
    return {"n": len(vals), "sum": sum(vals), "mean": statistics.mean(vals),
            "median": statistics.median(vals), "min": min(vals), "max": max(vals),
            "p10": vals[int((len(vals)-1)*0.1)], "p90": vals[int((len(vals)-1)*0.9)]}

def main():
    if OUT.exists():
        raise FileExistsError("Immutable design already exists; never overwrite or rerun.")
    state = read(ROOT / "LIA_CURRENT_STATE.json")
    if state["scientific_contract"]["sha256"] != sha(ROOT / "LIA_PROJECT_FACTS.md"):
        raise ValueError("Facts/state mismatch")
    population = read(POL / "population_191_v2.json")
    scores = read(POL / "rescored_cells_1393.json")
    ids = sorted(p["lia_unique_id"] for p in population)
    passed = {p["lia_unique_id"] for p in population if p["provisional_curve_pass_v2_count"] > 0}
    no_stable = {p["lia_unique_id"] for p in population if p["stable_parent_count"] == 0}
    unresolved = set(ids) - passed
    assert len(ids) == len(set(ids)) == 191
    assert (len(passed), len(no_stable), len(unresolved)) == (9, 18, 182)
    assert sum(bool(c["provisional_curve_pass_v2"]) for c in scores) == 77

    baseline = {}
    raw_counts, excluded_counts = Counter(), Counter()
    from_scratch_runtimes = defaultdict(list)
    model_params = {}
    staging_path = RES / "lia_191_all191_fullgrid_recovery_v10_270_transaction_v1/recovery_staging_audit.json"
    resumed_keys = set(read(staging_path)["member_keys"])
    assert len(resumed_keys) == 20
    for label in ("w264", "w267", "w270", "w273"):
        for path in sorted((ROOT / label / "m").glob("*/c/result.json")):
            r = read(path)
            if not r.get("all900_completed") or r.get("technical_failure") or not r.get("terminal"):
                excluded_counts[label] += 1
                continue
            cp_path = path.parent / "checkpoints/year-0900/checkpoint.json"
            cp = read(cp_path)
            sid, delta = int(r["lia_unique_id"]), float(r["delta_t_eq_degC"])
            assert cp["year"] == 900 and cp["lia_unique_id"] == sid
            assert cp["fingerprint"] == r["fingerprint"]
            assert cp["parameters"]["delta_t_eq_degC"] == delta
            assert sid in ids and delta in DELTAS
            params = {k: float(cp["parameters"][k]) for k in FIELDS}
            if sid in model_params:
                assert params == model_params[sid], ("baseline parameter drift", sid)
            model_params[sid] = params
            key = (sid, delta)
            if key in baseline:
                raise ValueError(("duplicate valid final baseline", key))
            baseline[key] = {
                "lia_unique_id": sid, "rgi_id": r["rgi_id"], "delta_t_eq_degC": delta,
                "melt_f_multiplier": 1.0, "stable": bool(r["final_year900_convergence"]["passed"]),
                "result": ref(path), "checkpoint_parameters_source": ref(cp_path),
                "original_fingerprint": r["fingerprint"], "source_root": label,
                "existing_result_action": "reuse-read-only-never-recompute"
            }
            raw_counts[label] += 1
            if label in ("w264", "w267") or (label == "w270" and path.parents[1].name not in resumed_keys):
                t = float(r["runtime_seconds"])
                if not math.isfinite(t) or t <= 0:
                    raise ValueError(("bad runtime", path))
                from_scratch_runtimes[sid].append(t)
    assert len(baseline) == 4011, len(baseline)
    assert sum(r["stable"] for r in baseline.values()) == 1393
    assert set(model_params) == set(ids)
    for sid in ids:
        assert {d for (i,d) in baseline if i == sid} == set(DELTAS)
        assert sum(r["stable"] for (i,d),r in baseline.items() if i == sid) == next(
            p["stable_parent_count"] for p in population if p["lia_unique_id"] == sid)

    forward_times = defaultdict(list)
    forward_inventories = []
    for path in sorted((ROOT / "w275/m").glob("*/result.json")):
        r = read(path)
        assert r["status"] == "completed" and not r["technical_failure"]
        forward_times[int(r["lia_unique_id"])].append(float(r["runtime_seconds"]))
        forward_inventories.append(ref(path))
    assert sum(map(len, forward_times.values())) == 1393
    assert set(unresolved) <= set(from_scratch_runtimes)
    scratch_stats = {sid: distribution(times) for sid, times in from_scratch_runtimes.items()}
    forward_stats = {sid: distribution(times) for sid, times in forward_times.items()}
    fallback_mean = statistics.mean([t for v in forward_times.values() for t in v])
    spin_worker_seconds = sum(42 * scratch_stats[sid]["mean"] for sid in unresolved)
    forward_per_all_stable = sum(42 * forward_stats.get(sid, {"mean": fallback_mean})["mean"] for sid in unresolved)
    former_stable_unresolved = sum(r["stable"] for (sid,d),r in baseline.items() if sid in unresolved)
    stable_fraction_reference = former_stable_unresolved / (len(unresolved)*21)
    timing = {
        "method": "sum of glacier-specific existing from-scratch member runtime means times42; divide20 workers; forward scenarios are not success predictions",
        "spinup_full_duration_samples": sum(map(len,from_scratch_runtimes.values())),
        "spinup_source_roots": ["w264","w267","w270 (864 verified year-zero starts only)"],
        "recovery_runtime_excluded": {"w270": sorted(resumed_keys), "w273":"all; year420 continuation"},
        "baseline_valid_results_per_root": dict(raw_counts),
        "discarded_nonterminal_or_technical_result_files": dict(excluded_counts),
        "spinup_seconds_by_glacier": scratch_stats,
        "forward_seconds_by_glacier": forward_stats,
        "no_stable_forward_timing_fallback": {"systems": sorted(no_stable), "method":"global1393-member mean, no glacier-specific forward measurement", "seconds":fallback_mean},
        "ideal_cpu20_spinup_hours": spin_worker_seconds/20/3600,
        "ideal_cpu20_forward_all_stable_hours": forward_per_all_stable/20/3600,
        "old_unresolved_stable_fraction": stable_fraction_reference,
        "old_unresolved_stable_cells": former_stable_unresolved,
        "scenario_total_hours": {
            str(fraction): (spin_worker_seconds+fraction*forward_per_all_stable)/20/3600
            for fraction in (0, stable_fraction_reference, 1)
        },
        "planning_overhead_multiplier_range": [1.0,1.5],
        "overhead_range_role": "explicit planning assumption, not measured confidence interval; not an upper bound on new physics cost",
        "limitations": [
            "Changed melt factor can change extent, stiffness and runtime; past means are not guaranteed.",
            "Existing jobs ran under their recorded concurrent workloads; no exclusive-machine benchmark.",
            "New stable count is unknown. Full7644 forward is a count ceiling, not a promised runtime ceiling.",
            "No dynamics benchmark was run; update estimates only from future authorized formal member completions.",
            "Member runtime may not capture all dispatch/setup costs; 1.0-1.5 is a planning allowance only."
        ]
    }
    ledger, matrices = [], []
    for p in sorted(population, key=lambda x:x["lia_unique_id"]):
        sid = p["lia_unique_id"]
        role = ("provisional-candidates-retained-for-separate-reproduction" if sid in passed else
                "new-melt-grid-no-baseline-stable" if sid in no_stable else
                "new-melt-grid-baseline-forward-nonpass")
        ledger.append({**p, "design_role":role, "baseline_params":model_params[sid],
                       "baseline_temperature_cells_reused":21,
                       "proposed_new_spinup_cells":0 if sid in passed else 42,
                       "new_execution_authorized":False})
        if sid not in unresolved:
            continue
        for multiplier in MULTIPLIERS:
            for delta in DELTAS:
                source = baseline[(sid, delta)]
                matrices.append({
                    "proposal_key":f"p{len(matrices):04d}",
                    "lia_unique_id":sid, "rgi_id":p["rgi_id"],"delta_t_eq_degC":delta,
                    **model_params[sid],"melt_f":model_params[sid]["melt_f"]*multiplier,
                    "melt_f_multiplier":multiplier,
                    "baseline_reference":source["result"],
                    "baseline_parameters_reference":source["checkpoint_parameters_source"],
                    "spinup_years":900,
                    "source_year900_geometry_is_not_new_equilibrium":True,
                    "physical_initial_geometry_binding":"original fixed prepared geometry; exact launch input binding pending",
                    "launch_authorized":False,
                    "stage":"prospective-spinup-only"
                })
    assert len(matrices) == 7644
    assert Counter(m["lia_unique_id"] for m in matrices) == Counter({sid:42 for sid in unresolved})
    signature={(m["lia_unique_id"],m["melt_f_multiplier"],m["delta_t_eq_degC"]) for m in matrices}
    assert len(signature) == len(matrices)
    rules = {
        "status":"prospective-design-not-launchable",
        "design_authorization":"user ok 开始 approved preparing next182-system values/count/runtime plan, not unknown grid launch",
        "clarified_initialization":"Stable initial geometry under spinup forcing is required; forward steady state or zero initial MB is not required. Initial growth is not a new rejection gate.",
        "proposed_melt_f_multipliers":[0.7,1.0,1.3],
        "multiplier_status":"0.7 and1.3 are newly proposed values for review, not authorized simulation values",
        "baseline_one_multiplier":"reuse all matching completed results; never schedule1.0 again",
        "delta_t_eq_grid_degC":DELTAS,
        "candidate_limit":None,"drop_nonbracketing_systems":False,
        "all42_new_combinations_per_unresolved_system":True,
        "spinup":{"duration_years":900,"forced_full_duration":True,
                  "final_same_phase_years":[810,840,870,900],
                  "volume_cycle_and_90year_relative_tolerance":0.03,
                  "length_cycle_and_90year_absolute_tolerance_m":300,
                  "use_exact_existing_relative_difference_operator":True,
                  "checkpoint_interval_years":30,"annual_diagnostics":True,
                  "initial_LIA_length_gate":False,"extend_to3000_or6000":False},
        "handoff":{"physical_melt_f_identical_across_stages":True,
                   "spinup_temperature":"inherited b + proposed delta_t_eq",
                   "forward_temperature":"inherited b; remove spinup delta_t_eq; q=0",
                   "preserve_all_stable_new_states":True,
                   "forward_only_after_state_hash_identity_and_stability_validated":True},
        "forward":{"start":1470,"end":2000,"climate":"unchanged ensemble_rec_ref, precipitation anomalies off",
                   "checkpoint_interval_years":30,"additional_final_checkpoint":2000,
                   "initial_zero_MB_or_steady_gate":False,"new_warmup_years":0,"initial_years_removed":False,
                   "policy":"CPU-direction-v2-retrospective six existing gates unchanged",
                   "gate_reference":ref(POL/"terminal_evidence.json")},
        "fixed_parameters":["prcp_fac","glen_a","fs","temp_melt","base_temp_bias","bed","widths","grid","solver family"],
        "later_glen_a_stage":"not expanded, not authorized; review after proposed melt-only batch",
        "runtime":{"proposed_CPU_workers":20,"internal_threads":1,"GPU":False,
                   "member_failure_isolation":True,"status_write_failure_cannot_stop_models":True,
                   "scientific_nonconvergence_is_not_technical_failure":True,
                   "no_new_automatic_batch_stop":True},
        "execution_pending":["explicit approval of this proposed grid",
                             "future immutable launch transaction, final source/hash/driver/geometry bindings and global duplicate/closed-mechanism reconciliation",
                             "reuse applicable existing numerical/recovery evidence; no new scientific success-rate preflight gate",
                             "bind separate short output root and collision-free process/resource ownership"],
        "model_trajectories_created":0,"launch_authorized":False,"accepted_ledger_effect":0
    }
    forward_template = {
        "stage":"prospective-forward-template-not-a-runnable-queue",
        "parents":"all newly stable output states from proposed spinup matrix; none exist yet",
        "counts":{"minimum":0,"maximum":7644,"actual":None},
        "filter":"technical state validity and inherited stability only; no initial LIA-length or zero-MB gate",
        "parameter_identity":"same modified melt_f in spinup and forward; no automatic extra temperature correction",
        "parent_hash_binding":"materialize after actual stable checkpoints exist",
        "rules":rules["forward"],"launch_authorized":False
    }
    sources = [
        ref(staging_path),
        ref(ROOT/"LIA_PROJECT_FACTS.md"), ref(ROOT/"LIA_EXECUTION_CONTRACT.md"),
        ref(ROOT/"LIA_CURRENT_STATE.json"), ref(ROOT/state["authoritative_route"]["path"]),
        ref(POL/"population_191_v2.json"),ref(POL/"rescored_cells_1393.json"),ref(POL/"summary.json"),
        ref(RES/"lia_cpu_stage2_handoff_design_v1/固定重建气候后续计划_v2.md"),
        ref(RES/"lia_191_all191_fullgrid_finalcell_retry_v10_273_terminal_v1/summary.json"),
        ref(RES/"lia_191_all191_fullgrid_completion_v10_267_transaction_v1/input_binding.json"),
        ref(RES/"lia_cpu_fixed_climate_forward_v10_275_transaction_v1/transaction_manifest.json"),
        ref(Path(__file__))
    ]
    OUT.mkdir()
    artifacts = {}
    for name, value in [
        ("rules-proposed.json",rules),("population_191.json",ledger),
        ("baseline_reuse_4011.json",list(baseline.values())),
        ("spinup_matrix_proposed_7644.json",matrices),
        ("forward_template.json",forward_template),
        ("retained_provisional_candidates_77.json",[c for c in scores if c["provisional_curve_pass_v2"]]),
        ("runtime_evidence.json",timing),("runtime_forward_source_inventory.json",forward_inventories)
    ]:
        artifacts[name]=save(name,value)
    summary = {
        "schema":"lia-cpu-melt-factor-prospective-design-v10.278",
        "status":"design-complete-not-launchable",
        "created_at":datetime.now(timezone.utc).isoformat(),
        "population":191,"provisional_systems_retained":9,"provisional_cells_retained":77,
        "baseline_forward_nonpass_systems":164,"baseline_no_stable_systems":18,
        "proposed_new_systems":182,"proposed_new_multipliers":MULTIPLIERS,
        "temperature_values_per_multiplier":21,"proposed_new_spinup_cells":7644,
        "baseline_spinup_results_reused_all191":4011,
        "baseline_spinup_results_for182":3822,
        "new_forward_count":"unknown; all newly stable combinations, max7644",
        "new_checkpoints_count_spinup_if_all_complete":7644*31,
        "new_checkpoints_count_forward_max":7644*19,
        "proposal_fingerprint_not_launch_fingerprint":hashlib.sha256(
            json.dumps({"rules":rules,"matrix_hash":artifacts["spinup_matrix_proposed_7644.json"]["sha256"]},
                       sort_keys=True).encode()).hexdigest().upper(),
        "runtime_planning":{k:v for k,v in timing.items() if k in (
            "ideal_cpu20_spinup_hours","ideal_cpu20_forward_all_stable_hours",
            "scenario_total_hours","planning_overhead_multiplier_range","old_unresolved_stable_fraction")},
        "sources":sources,"artifacts":artifacts,
        "validation":{"all191_unique":True,"all182_have42_new_cells":True,
                      "all18_no_stable_included":True,"all77_provisional_preserved":True,
                      "baseline4011_complete":True,"baseline_parameters_consistent":True,
                      "no_new_dynamics":True,"global_older_physical_cache_audit":"pending before any launch"},
        "launch_authorized":False,"simulation_launched":False,
        "accepted_ledger_effect":0,"old_outputs_modified":False
    }
    save("summary.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("sources","artifacts")},ensure_ascii=False))

if __name__ == "__main__":
    main()
