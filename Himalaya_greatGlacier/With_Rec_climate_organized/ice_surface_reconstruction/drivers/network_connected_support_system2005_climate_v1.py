"""System-2005 climate run with conservative connected-support constraints."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import traceback
from pathlib import Path

import numpy as np

from . import network_connected_front_system2005_climate_v2 as registered
from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from .network_capacity_connected_support_step_v1 import (
    MINIMUM_SUPPORT_THICKNESS_M,
    connected_support_step,
    minimum_section_for_support,
)
from .network_connected_support_projection_v1 import project_lower_bounded_simplex


PROJECT = Path(__file__).resolve().parents[4]


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT / candidate


def connected_support_climate_source(flowlines, mb_model, model_year: float, dt_seconds: float):
    accepted_volume = 0.0
    redistributed_volume = 0.0
    remote_zero_nodes = 0
    for branch, fl in enumerate(flowlines):
        before = np.asarray(fl.section, dtype=float).copy()
        active = np.asarray(fl.thick) > base.ACTIVE_TOLERANCE_M
        remote_zero_nodes += int(np.count_nonzero(~active))
        mb = np.asarray(
            mb_model.get_monthly_mb(np.asarray(fl.surface_h), year=model_year, fl_id=branch, fls=flowlines),
            dtype=float,
        )
        after = before.copy()
        if np.any(active):
            idx = np.flatnonzero(active)
            requested = before[idx] + float(dt_seconds) * mb[idx] * np.asarray(fl.widths_m)[idx]
            lower = np.asarray([minimum_section_for_support(fl, int(node)) for node in idx], dtype=float)
            projected = project_lower_bounded_simplex(requested, lower)
            after[idx] = projected
            redistributed_volume += 0.5 * float(np.sum(np.abs(projected - requested))) * fl.dx_meter
        if np.any(after[~active] != before[~active]):
            raise RuntimeError("remote-zero-node-received-local-mb")
        accepted_volume += float(np.sum(after - before)) * fl.dx_meter
        with np.errstate(invalid="raise", over="raise"):
            fl.section = after
        if np.any(np.asarray(fl.thick)[active] <= base.ACTIVE_TOLERANCE_M):
            raise RuntimeError(f"mb-projection-lost-active-support:{branch}")
    return accepted_volume, redistributed_volume, remote_zero_nodes


def run(config: dict):
    old_source = registered.connected_ice_climate_source
    old_step = registered.connected_front_step_v2
    registered.connected_ice_climate_source = connected_support_climate_source
    registered.connected_front_step_v2 = connected_support_step
    try:
        summary, rows, flowlines = registered.run(config)
    finally:
        registered.connected_ice_climate_source = old_source
        registered.connected_front_step_v2 = old_step
    summary["schema"] = "lia-191-network-connected-support-system2005-climate-v1"
    summary["mechanism"] = "network-connected-support-conservative-mb-and-transport-v1"
    summary["minimum_support_thickness_m"] = MINIMUM_SUPPORT_THICKNESS_M
    summary["cumulative_mb_redistributed_m3"] = summary.pop("cumulative_prevented_overmelt_m3")
    summary["local_mb_total_preserved"] = True
    summary["posthoc_section_clip_used"] = False
    for row in rows:
        row["cumulative_mb_redistributed_m3"] = row.pop("cumulative_prevented_overmelt_m3")
    return summary, rows, flowlines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(resolve(args.config).read_text(encoding="utf-8"))
    output = resolve(config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "config_snapshot.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        summary, rows, flowlines = run(config)
    except Exception as exc:
        summary = {
            "schema": "lia-191-network-connected-support-system2005-climate-v1",
            "status": "climate-technical-or-scientific-fail",
            "system_id": 2005,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "historical_run": False,
        }
        rows, flowlines = [], None
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    if rows:
        with (output / "timeseries.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if summary["status"] == "steady-50y-pass" and flowlines is not None:
        with gzip.open(output / "lia_equilibrium_state_50y.pkl.gz", "wb") as stream:
            pickle.dump(flowlines, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps(summary, ensure_ascii=False))
    if summary["status"] in {"climate-technical-or-scientific-fail", "climate-screen-fail", "steady-50y-fail"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
