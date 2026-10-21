"""Five-year gate for the registered system-2005 topology-adaptive split."""

from __future__ import annotations

import argparse
import csv
import json
import traceback

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_topology_adaptive_split_system2005_zero_mb_v4 as solver


PASS_0P1 = "zero-mb-0p1y-pass-topology-adaptive-split"
PASS_5Y = "zero-mb-5y-pass-topology-adaptive-split"


def run(config):
    preflight_path = base.resolve(config["registered_0p1y_pass"])
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != PASS_0P1 or float(preflight.get("years", -1)) != 0.1:
        raise RuntimeError("registered-0p1y-gate-not-passed")
    if not all(preflight.get("checks", {}).values()):
        raise RuntimeError("registered-0p1y-checks-not-all-passed")
    if float(config["years"]) != 5.0:
        raise RuntimeError("five-year-duration-contract-mismatch")

    summary, rows = solver.run(config)
    if summary.get("status") != PASS_0P1:
        summary.update({
            "schema": "lia-191-network-topology-adaptive-split-system2005-zero-mb-5y-v1",
            "status": "zero-mb-5y-fail-mechanism-closed",
            "zero_mb_5y_run": True,
            "steady_50y_run": False,
            "historical_run": False,
            "next_action": "close mechanism",
        })
        return summary, rows

    output = base.resolve(config["output_root"])
    old_state = output / "model_flowlines_0p1y.pkl.gz"
    state = output / "model_flowlines_5y.pkl.gz"
    if not old_state.exists():
        raise RuntimeError("five-year-state-missing")
    old_state.rename(state)
    summary.update({
        "schema": "lia-191-network-topology-adaptive-split-system2005-zero-mb-5y-v1",
        "status": PASS_5Y,
        "saved_state": str(state.relative_to(base.ROOT)),
        "saved_state_sha256": base.sha256(state),
        "zero_mb_5y_run": True,
        "steady_50y_run": False,
        "historical_run": False,
        "next_action": "run one registered 50y steady screen using the five-year-passed geometry contract",
        "registered_0p1y_pass_sha256": base.sha256(preflight_path),
    })
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = base.resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = base.resolve(config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    try:
        summary, rows = run(config)
    except Exception as exc:
        summary = {
            "schema": "lia-191-network-topology-adaptive-split-system2005-zero-mb-5y-v1",
            "status": "zero-mb-5y-technical-or-scientific-fail",
            "system_id": 2005,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "source_modified": False,
            "zero_mb_5y_run": True,
            "steady_50y_run": False,
            "historical_run": False,
        }
        rows = []
    (output / "config_snapshot.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if rows:
        with (output / "checkpoints.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary))
    if summary.get("status") != PASS_5Y:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

