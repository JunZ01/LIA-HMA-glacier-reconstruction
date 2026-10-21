"""Final v3 runner after quantified 1.15e-10 m3 reserve roundoff."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import traceback

from . import network_connected_support_system2005_climate_v1 as v1
from .network_capacity_connected_support_step_v3 import connected_support_step


def run(config):
    original = v1.connected_support_step
    v1.connected_support_step = connected_support_step
    try:
        summary, rows, flowlines = v1.run(config)
    finally:
        v1.connected_support_step = original
    summary["schema"] = "lia-191-network-connected-support-system2005-climate-v3"
    summary["implementation_correction"] = "quantified-reserve-roundoff-up-to-1e-8-m3-canonicalized"
    return summary, rows, flowlines


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args()
    config = json.loads(v1.resolve(args.config).read_text(encoding="utf-8")); output = v1.resolve(config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "config_snapshot.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        summary, rows, flowlines = run(config)
    except Exception as exc:
        summary = {"schema": "lia-191-network-connected-support-system2005-climate-v3", "status": "climate-technical-or-scientific-fail", "system_id": 2005, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "historical_run": False}
        rows, flowlines = [], None
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    if rows:
        with (output / "timeseries.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if summary["status"] == "steady-50y-pass" and flowlines is not None:
        with gzip.open(output / "lia_equilibrium_state_50y.pkl.gz", "wb") as stream: pickle.dump(flowlines, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps(summary, ensure_ascii=False))
    if summary["status"] in {"climate-technical-or-scientific-fail", "climate-screen-fail", "steady-50y-fail"}: raise SystemExit(1)


if __name__ == "__main__": main()
