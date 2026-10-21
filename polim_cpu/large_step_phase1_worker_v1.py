"""One registered large-step GPU-BP/implicit-thickness phase-1 worker."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

from .gpu_energy_worker_v1 import np, tensor, torch
from .historical_climate_v1 import annual
from .large_step_gpu_bridge_v1 import ProjectedAdaptiveBPSolver
from .large_step_gpu_coupling_v1 import solve_gpu_coupled_step
from .large_step_transport_client_v1 import IsolatedTransportService
from . import packed_storage_v1 as st


def run(root: Path) -> None:
    started = time.perf_counter()
    cfg = st.read(root / "config.json")
    contract = st.read(root / "contract.json")["contract"]
    if not cfg.get("launch_authorized", False):
        raise RuntimeError("case configuration does not authorize launch")
    source = Path(cfg["source_root"])
    source_contract = st.read(source / "contract.json")["contract"]
    z = st.arrays(source / "input.npz")
    climate = st.arrays(source / "climate.npz")
    state, source_meta = st.restore(source / cfg["source_frame"], source_contract)
    if abs(float(source_meta["year"]) - float(cfg["start_year"])) > 1e-10:
        raise RuntimeError("source checkpoint year does not match the frozen case")
    if float(cfg["end_year"]) > np.floor(float(cfg["start_year"])) + 1 + 1e-12:
        raise RuntimeError("a phase-1 step may not cross the climate-year boundary")
    torch.cuda.set_per_process_memory_fraction(float(cfg["allocator_fraction"]))
    h_old = np.asarray(state["h"], dtype=np.float64)
    u_old = np.asarray(state["u"], dtype=np.float64)
    backend = SimpleNamespace(xp=torch, array=tensor)
    smb = annual(
        tensor(z["bed"]) + tensor(h_old),
        climate,
        cfg["parameters"],
        int(np.floor(float(cfg["start_year"]))),
        backend,
    ).cpu().numpy()
    initial_meta = {
        "year": float(cfg["start_year"]),
        "source": str(source / cfg["source_frame"]),
        "source_state_sha256": st.digest(source / cfg["source_frame"] / "state.npz"),
        "simulation_advanced": False,
    }
    st.save(root, "frame-000000", h_old, u_old, initial_meta, contract)
    unavailable = st.status(
        root,
        {"status": "running", "stage": "combined-technical-check", "pid": os.getpid()},
        inject=True,
    )
    if unavailable:
        raise RuntimeError("injected unavailable progress target unexpectedly succeeded")
    try:
        with IsolatedTransportService(z, root / "transport-service") as transport:
            with ProjectedAdaptiveBPSolver(z, root / "amg-cache") as bp_solver:
                result = solve_gpu_coupled_step(
                    h_old,
                    u_old,
                    z["bed"],
                    smb,
                    float(cfg["end_year"]) - float(cfg["start_year"]),
                    transport,
                    bp_solver,
                    use_diffusion_predictor=bool(cfg["use_diffusion_predictor"]),
                    damping=float(cfg["outer_damping"]),
                    maximum_bp_calls=int(cfg["maximum_bp_calls"]),
                    bp_tolerance=float(cfg["gates"]["BP_relative"]),
                    thickness_equation_tolerance=float(
                        cfg["gates"]["thickness_equation_relative"]
                    ),
                )
                result["amg_rebuilds"] = bp_solver.rebuilds
        accepted_meta = {
            key: value
            for key, value in result.items()
            if key not in {"thickness", "velocity", "actual_smb"}
        }
        accepted_meta.update(
            year=float(cfg["end_year"]),
            actual_smb_sha256=st.array_hash(result["actual_smb"]),
            simulation_advanced=True,
        )
        st.save(
            root,
            "frame-000001",
            result["thickness"],
            result["velocity"],
            accepted_meta,
            contract,
        )
        restored, restored_meta = st.restore(root / "frame-000001", contract)
        if not np.array_equal(restored["h"], result["thickness"]):
            raise RuntimeError("accepted thickness failed immediate checkpoint restore")
        if not np.array_equal(restored["u"], result["velocity"]):
            raise RuntimeError("accepted velocity failed immediate checkpoint restore")
        status = {
            "status": "completed",
            "year": float(cfg["end_year"]),
            "BP_relative": result["bp_relative"],
            "thickness_equation_relative": result[
                "thickness_equation_relative"
            ],
            "mass_relative": result["mass_relative"],
            "bp_calls": result["bp_calls"],
            "outer_iterations": result["outer_iterations"],
            "wall_seconds": time.perf_counter() - started,
            "checkpoint": "frame-000001",
            "progress_unavailable_survived": True,
            "checkpoint_immediate_restore": True,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        }
        st.write(root / "completion.json", status)
        if not st.status(root, status):
            raise RuntimeError("final replaceable progress status could not be written")
        print(json.dumps(status), flush=True)
    except Exception as exc:
        failure = {
            "status": "failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "wall_seconds": time.perf_counter() - started,
        }
        if hasattr(exc, "history"):
            failure["outer_history"] = exc.history
        st.write(root / "failure.json", failure)
        st.status(root, {key: value for key, value in failure.items() if key != "traceback"})
        print(json.dumps({key: value for key, value in failure.items() if key != "traceback"}), flush=True)
        raise


if __name__ == "__main__":
    run(Path(sys.argv[1]))
