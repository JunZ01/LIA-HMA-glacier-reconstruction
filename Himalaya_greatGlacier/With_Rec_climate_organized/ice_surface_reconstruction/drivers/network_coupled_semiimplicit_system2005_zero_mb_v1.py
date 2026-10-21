"""Short zero-MB integration for the system-2005 network semiimplicit solver."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import pickle
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

from oggm import cfg
from oggm.cfg import G

from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.network_coupled_semiimplicit_system2005_step_v3 import (
    _mixedbed_face_conductance,
)


PROJECT = Path(__file__).resolve().parents[4]
SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0
ACTIVE_TOLERANCE_M = 1e-6
REVERSE_FLUX_TOLERANCE_M3_S = 1e-12


class Coefficients:
    def __init__(self, glen_a: float, fs: float):
        self.glen_n = float(cfg.PARAMS["glen_n"])
        self.glen_a = float(glen_a)
        self.fs = float(fs)
        self.rho = float(cfg.PARAMS["ice_density"])
        self._fd = 2.0 / (self.glen_n + 2.0) * self.glen_a
        self.rhog = (self.rho * float(G)) ** self.glen_n


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT / candidate


def load_flowlines(path: Path):
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def total_volume(flowlines) -> float:
    return float(sum(np.sum(fl.section) * fl.dx_meter for fl in flowlines))


def total_area(flowlines) -> float:
    return float(sum(np.sum(fl.widths_m) * fl.dx_meter for fl in flowlines))


def build_active_graph(flowlines, coefficients):
    branch_by_object = {id(fl): index for index, fl in enumerate(flowlines)}
    active_local = []
    global_of = {}
    active_by_branch = []
    for branch, fl in enumerate(flowlines):
        active = np.flatnonzero(np.asarray(fl.thick, dtype=float) > ACTIVE_TOLERANCE_M)
        active_by_branch.append(active)
        for node in active:
            global_of[(branch, int(node))] = len(active_local)
            active_local.append((branch, int(node)))
    edges = []
    for branch, fl in enumerate(flowlines):
        active = active_by_branch[branch]
        aset = set(map(int, active))
        for node in active[:-1]:
            node = int(node)
            if node + 1 not in aset:
                continue
            edges.append((
                global_of[(branch, node)],
                global_of[(branch, node + 1)],
                _mixedbed_face_conductance(coefficients, fl, node, fl, node + 1),
                "within",
            ))
        if fl.flows_to is not None and len(active) and int(active[-1]) == len(fl.thick) - 1:
            receiver = branch_by_object[id(fl.flows_to)]
            receiver_node = int(fl.flows_to_indice)
            terminal = int(active[-1])
            if (receiver, receiver_node) not in global_of:
                raise RuntimeError(
                    f"active tributary {branch} targets inactive receiver {receiver}/{receiver_node}"
                )
            edges.append((
                global_of[(branch, terminal)],
                global_of[(receiver, receiver_node)],
                _mixedbed_face_conductance(
                    coefficients, fl, terminal, flowlines[receiver], receiver_node
                ),
                "junction",
            ))
    return active_local, edges


def one_step(flowlines, coefficients, dt_seconds: float, closure_limit_m3: float) -> dict:
    active_local, edges = build_active_graph(flowlines, coefficients)
    node_count = len(active_local)
    surface_before = np.asarray(
        [flowlines[b].surface_h[n] for b, n in active_local], dtype=float
    )
    storage = np.asarray(
        [flowlines[b].widths_m[n] * flowlines[b].dx_meter for b, n in active_local],
        dtype=float,
    )
    if np.any(~np.isfinite(storage)) or np.any(storage <= 0):
        raise RuntimeError("nonfinite-or-nonpositive-active-storage")
    row, col, data = [], [], []
    for u, v, conductance, _kind in edges:
        value = float(dt_seconds) * float(conductance)
        row.extend([u, v, u, v])
        col.extend([u, v, v, u])
        data.extend([value, value, -value, -value])
    graph = coo_matrix((data, (row, col)), shape=(node_count, node_count)).tocsr()
    matrix = diags(storage, format="csr") + graph
    if np.any(~np.isfinite(matrix.data)) or np.any(matrix.diagonal() <= 0):
        raise RuntimeError("invalid-network-matrix")
    surface_after = np.asarray(spsolve(matrix, storage * surface_before), dtype=float)
    if np.any(~np.isfinite(surface_after)):
        raise RuntimeError("nonfinite-network-solution")

    flux_divergence_volume = np.zeros(node_count, dtype=float)
    reverse_within = 0
    reverse_junction = 0
    for u, v, conductance, kind in edges:
        flux = float(conductance * (surface_after[u] - surface_after[v]))
        transfer = flux * float(dt_seconds)
        flux_divergence_volume[u] += transfer
        flux_divergence_volume[v] -= transfer
        if flux < -REVERSE_FLUX_TOLERANCE_M3_S:
            if kind == "junction":
                reverse_junction += 1
            else:
                reverse_within += 1

    linear_residual = storage * (surface_after - surface_before) + flux_divergence_volume
    before_volume = total_volume(flowlines)
    sections = [np.asarray(fl.section, dtype=float).copy() for fl in flowlines]
    for global_index, (branch, node) in enumerate(active_local):
        sections[branch][node] -= flux_divergence_volume[global_index] / float(
            flowlines[branch].dx_meter
        )
    minimum_section = min(float(np.min(section)) for section in sections)
    if minimum_section < -1e-10:
        raise RuntimeError(f"negative-section-required:{minimum_section}")
    for fl, section in zip(flowlines, sections):
        fl.section = section
    after_volume = total_volume(flowlines)
    actual_closure = after_volume - before_volume
    minimum_thickness = min(float(np.min(fl.thick)) for fl in flowlines)
    if abs(actual_closure) > closure_limit_m3:
        raise RuntimeError(
            f"per-step-volume-closure:{actual_closure}>{closure_limit_m3}"
        )
    if reverse_within or reverse_junction:
        raise RuntimeError(
            f"reverse-flux:within={reverse_within},junction={reverse_junction}"
        )
    if minimum_thickness < -1e-10:
        raise RuntimeError(f"negative-thickness:{minimum_thickness}")
    return {
        "active_node_count": node_count,
        "edge_count": len(edges),
        "junction_edge_count": sum(kind == "junction" for *_rest, kind in edges),
        "actual_closure_m3": actual_closure,
        "maximum_absolute_linear_residual_m3": float(np.max(np.abs(linear_residual))),
        "minimum_section_m2": minimum_section,
        "minimum_thickness_m": minimum_thickness,
    }


def run(config: dict) -> tuple[dict, list[dict]]:
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_kcalving_for_run"] = False
    fixture = json.loads(resolve(config["fixture_summary"]).read_text(encoding="utf-8"))
    step_gate = json.loads(resolve(config["real_single_step_summary"]).read_text(encoding="utf-8"))
    if fixture.get("status") != "fixture-pass-real-system-not-authorized":
        raise RuntimeError("fixture-gate-not-passed")
    if step_gate.get("status") != "real-single-step-pass":
        raise RuntimeError("real-single-step-gate-not-passed")
    source_path = resolve(config["model_flowlines"])
    flowlines = load_flowlines(source_path)
    coefficients = Coefficients(config["glen_a"], config["fs"])
    initial_volume = total_volume(flowlines)
    initial_area = total_area(flowlines)
    closure_limit = max(1e-6, initial_volume * 1e-12)
    years = float(config["years"])
    target_seconds = years * SECONDS_PER_YEAR
    maximum_dt = float(config["maximum_timestep_seconds"])
    minimum_dt = float(config["minimum_timestep_seconds"])
    if maximum_dt < minimum_dt or minimum_dt < 30.0:
        raise ValueError("invalid-timestep-contract")
    elapsed = 0.0
    step_index = 0
    maximum_step_closure = 0.0
    maximum_linear_residual = 0.0
    minimum_used_dt = float("inf")
    maximum_used_dt = 0.0
    final_step = {}
    rows = []
    started = time.perf_counter()
    while elapsed < target_seconds - 1e-9:
        dt = min(maximum_dt, target_seconds - elapsed)
        if dt < minimum_dt - 1e-9:
            raise RuntimeError(f"required-final-step-below-floor:{dt}")
        final_step = one_step(flowlines, coefficients, dt, closure_limit)
        elapsed += dt
        step_index += 1
        minimum_used_dt = min(minimum_used_dt, dt)
        maximum_used_dt = max(maximum_used_dt, dt)
        maximum_step_closure = max(
            maximum_step_closure, abs(final_step["actual_closure_m3"])
        )
        maximum_linear_residual = max(
            maximum_linear_residual,
            final_step["maximum_absolute_linear_residual_m3"],
        )
        if step_index == 1 or elapsed >= target_seconds - 1e-9:
            rows.append({
                "step": step_index,
                "year": elapsed / SECONDS_PER_YEAR,
                "dt_seconds": dt,
                "volume_m3": total_volume(flowlines),
                "area_m2": total_area(flowlines),
                **final_step,
            })
    final_volume = total_volume(flowlines)
    final_area = total_area(flowlines)
    drift = (final_volume - initial_volume) / initial_volume
    root = next(fl for fl in flowlines if fl.flows_to is None)
    terminal = float(root.thick[-1])
    gates = config["gates"]
    checks = {
        "completed_target": abs(elapsed - target_seconds) <= 1e-6,
        "volume_drift": abs(drift) <= float(gates["volume_drift_fraction_max"]),
        "per_step_closure": maximum_step_closure <= closure_limit,
        "terminal_thickness": terminal <= float(gates["terminal_thickness_m_max"]),
        "finite_geometry": all(
            np.isfinite(fl.thick).all()
            and np.isfinite(fl.section).all()
            and np.isfinite(fl.widths_m).all()
            for fl in flowlines
        ),
        "nonnegative_geometry": all(
            np.all(fl.thick >= -1e-10) and np.all(fl.section >= -1e-10)
            for fl in flowlines
        ),
        "minimum_timestep_floor": minimum_used_dt >= minimum_dt - 1e-9,
    }
    status = "zero-mb-pass" if all(checks.values()) else "zero-mb-fail-mechanism-closed"
    summary = {
        "schema": "lia-191-network-coupled-semiimplicit-system2005-zero-mb-v1",
        "status": status,
        "system_id": 2005,
        "years": years,
        "step_count": step_index,
        "minimum_timestep_seconds": minimum_used_dt,
        "maximum_timestep_seconds": maximum_used_dt,
        "initial_volume_m3": initial_volume,
        "final_volume_m3": final_volume,
        "volume_drift_fraction": drift,
        "initial_area_m2": initial_area,
        "final_area_m2": final_area,
        "maximum_absolute_step_closure_m3": maximum_step_closure,
        "maximum_absolute_linear_residual_m3": maximum_linear_residual,
        "closure_limit_m3": closure_limit,
        "root_terminal_thickness_m": terminal,
        "checks": checks,
        "runtime_seconds": time.perf_counter() - started,
        "solver": "network-coupled-semiimplicit-finite-volume-v1",
        "mass_balance": "zero",
        "runtime_clip_used": False,
        "source_model_flowlines_sha256": sha256(source_path),
        "source_modified": False,
        "steady_run": False,
        "historical_run": False,
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = resolve(config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    try:
        summary, rows = run(config)
    except Exception as exc:
        summary = {
            "schema": "lia-191-network-coupled-semiimplicit-system2005-zero-mb-v1",
            "status": "zero-mb-technical-or-scientific-fail",
            "system_id": 2005,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "source_modified": False,
            "steady_run": False,
            "historical_run": False,
        }
        rows = []
    (output / "config_snapshot.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if rows:
        with (output / "checkpoints.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False))
    if summary["status"] != "zero-mb-pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

