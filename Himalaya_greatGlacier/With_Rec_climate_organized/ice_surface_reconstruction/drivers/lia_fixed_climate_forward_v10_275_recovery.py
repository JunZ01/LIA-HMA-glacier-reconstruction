"""Explicitly authorized recovery only; never a fresh or automatic retry."""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
from . import lia_fixed_climate_forward_v10_275 as run
os.environ.update(run.THREADS)


def restore(checkpoint, row, work):
    from . import lia_all191_flowline_forward_v10_237 as base
    cp, fls, history = run.load_checkpoint(checkpoint, row)
    base._configure_oggm(work, work)
    mb = run.mb_for(row, run.read(run.TX / "climate_inventory.json"))
    model = base.make_model(fls, mb, row["glen_a_pa_3_s_1"], row["fs"], cp["calendar_year"])
    import numpy as np
    for left, right in zip(model.fls, fls):
        np.testing.assert_allclose(left.section, right.section, rtol=1e-7, atol=0)
    return model, cp, history


def test():
    """Test the exact future recovery entry using the already saved static fixture."""
    _, rows = run.contract()
    row = min(rows, key=lambda r: int(r["nx"]))
    cp = run.OUT / "test" / row["key"] / "cp/1470"
    model, payload, history = restore(cp, row, run.OUT / "test/recovery_w")
    assert model.yr == 1470 and len(history) == 1
    report = {"status": "pass", "dynamic_steps": 0, "code": run.evidence(Path(__file__)),
              "saved_checkpoint": run.evidence(cp / "c.json"), "restored_calendar_year": model.yr,
              "state_and_history_synchronized": True, "requires_new_explicit_recovery_authorization": True,
              "already_completed_member_recomputation_forbidden": True}
    run.write_new(run.TX / "recovery_readiness.json", report)
    print(json.dumps(report))


def recover(authorization):
    """Future recovery receipt must be explicitly bound by a new current route."""
    report = run.validate()
    state = run.read(run.ROOT / "LIA_CURRENT_STATE.json")
    auth = run.read(authorization)
    if (not report["launch_authorized"] or state["next_action"].get("recovery_authorization_sha256") != run.sha(authorization)
            or not auth.get("explicit_user_recovery_approval") or auth["recovery_driver_sha256"] != run.sha(Path(__file__))):
        raise ValueError("new-explicit-recovery-transaction-required")
    _, rows = run.contract()
    row = next(r for r in rows if r["fingerprint"] == auth["fingerprint"])
    original = run.OUT / "m" / row["key"]
    if (original / "result.json").exists():
        raise ValueError("terminal-member-cannot-be-recomputed-by-recovery")
    cp_path = Path(auth["checkpoint_path"]).resolve()
    if run.sha(cp_path / "c.json") != auth["checkpoint_manifest_sha256"]:
        raise ValueError("recovery-checkpoint-binding-mismatch")
    output = Path(auth["new_output_root"]).resolve()
    output.relative_to(run.ROOT)
    output.mkdir(exist_ok=False)
    model, cp, history = restore(cp_path, row, output / "w")
    counters = dict(cp["counters"])
    step = model.step
    def measured(dt):
        value = float(step(dt))
        counters["accepted_steps"] += 1
        prior = counters["minimum_dt_seconds"]
        counters["minimum_dt_seconds"] = value if prior is None else min(prior, value)
        return value
    model.step = measured
    started = time.perf_counter()
    for year in range(cp["calendar_year"] + 1, 2001):
        model.run_until(float(year))
        history.append(run.validate_state(model, year))
        if year in run.YEARS:
            saved = run.save_checkpoint(output, model, year, row, history, counters)
            run.load_checkpoint(saved, row)
            run.status_write(output / "progress.json", {"year": year, "counters": counters})
    run.write_new(output / "result.json", {"status": "completed", "terminal": True, "technical_failure": False,
                  "fingerprint": row["fingerprint"], "key": row["key"], "lia_unique_id": row["lia_unique_id"],
                  "source_checkpoint": run.evidence(cp_path / "c.json"), "resume_year": cp["calendar_year"],
                  "end_year": 2000, "annual_rows": len(history), "new_checkpoints": [y for y in run.YEARS if y > cp["calendar_year"]],
                  "prior_checkpoints_readonly": True, "checkpoint_contract_pass": True, "counters": counters,
                  "runtime_seconds": time.perf_counter() - started, "accepted_ledger_effect": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("test", "recover"))
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    test() if args.action == "test" else recover(args.authorization)
