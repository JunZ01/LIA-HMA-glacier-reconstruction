"""Gated 50-year steady runner for the converged six-hour topology solver."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import traceback

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_topology_adaptive_split_system2005_climate_v2 as climate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(base.resolve(args.config).read_text(encoding="utf-8"))
    output = base.resolve(config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "config_snapshot.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    try:
        convergence = json.loads(base.resolve(config["timestep_convergence_summary"]).read_text(encoding="utf-8"))
        if convergence.get("status") != "six-hour-converged-authorized-for-50y":
            raise RuntimeError("six-hour-convergence-gate-not-passed")
        if float(config["years"]) != 50.0 or float(config["maximum_timestep_seconds"]) != 21600.0:
            raise RuntimeError("registered-steady-duration-or-timestep-mismatch")
        summary, rows, flowlines = climate.run(config)
        summary["timestep_convergence_summary_sha256"] = base.sha256(base.resolve(config["timestep_convergence_summary"]))
    except Exception as exc:
        summary = {"schema": "lia-191-network-topology-adaptive-split-system2005-steady-50y-v1", "status": "steady-50y-technical-or-scientific-fail", "system_id": 2005, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "historical_run": False}
        rows, flowlines = [], None
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if rows:
        with (output / "timeseries.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if summary.get("status") == "steady-50y-pass" and flowlines is not None:
        with gzip.open(output / "lia_equilibrium_state_50y.pkl.gz", "wb") as stream:
            pickle.dump(flowlines, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps(summary))
    if summary.get("status") != "steady-50y-pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
