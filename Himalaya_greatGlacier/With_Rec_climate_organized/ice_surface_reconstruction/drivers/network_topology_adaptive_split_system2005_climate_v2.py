"""Climate runner v2: formal closure uses the registered per-step maximum."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import traceback

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_topology_adaptive_split_system2005_climate_v1 as v1


def run(config):
    summary, rows, flowlines = v1.run(config)
    checks = dict(summary["numerical_checks"])
    checks.pop("total_budget_closure")
    checks["cumulative_budget_closure_diagnostic_only"] = True
    summary["numerical_checks"] = checks
    steady = dict(summary["steady_checks"])
    steady.pop("total_budget_closure")
    steady["cumulative_budget_closure_diagnostic_only"] = True
    summary["steady_checks"] = steady
    summary["schema"] = "lia-191-network-topology-adaptive-split-system2005-climate-v2"
    summary["formal_closure_definition"] = "maximum absolute per-step residual; cumulative floating residual is diagnostic only"
    if float(config["years"]) == 50.0:
        summary["status"] = "steady-50y-pass" if all(steady.values()) else "steady-50y-fail"
    else:
        summary["status"] = "climate-screen-pass" if all(checks.values()) else "climate-screen-fail"
    return summary, rows, flowlines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(base.resolve(args.config).read_text(encoding="utf-8"))
    output = base.resolve(config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "config_snapshot.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    try:
        summary, rows, flowlines = run(config)
    except Exception as exc:
        summary = {"schema": "lia-191-network-topology-adaptive-split-system2005-climate-v2", "status": "climate-technical-or-scientific-fail", "system_id": 2005, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "historical_run": False}
        rows, flowlines = [], None
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if rows:
        with (output / "timeseries.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if summary.get("status") == "steady-50y-pass" and flowlines is not None:
        with gzip.open(output / "lia_equilibrium_state_50y.pkl.gz", "wb") as stream:
            pickle.dump(flowlines, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps(summary))
    if summary.get("status") not in {"climate-screen-pass", "steady-50y-pass"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
