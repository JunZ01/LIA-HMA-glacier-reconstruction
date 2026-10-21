"""Register, launch and close the unique phase-1 large-step BP trial."""
from __future__ import annotations

from .solver_acceleration_v1 import *
from . import packed_storage_v1 as st
import subprocess


NAME = "polim_v37_large_step_phase1_v1"
E = PROJECT / "validation_data/results/paleo_surface_validation_v0" / NAME
RUN = R / "runs/v37-large-step-phase1-v1/system2387-1482-quarter"
SOURCE = R / "runs/v32r4-historical-1470-1570-v1"
CODE = [
    "large_step_transport_v1.py",
    "large_step_coupling_v1.py",
    "large_step_gpu_bridge_v1.py",
    "large_step_gpu_coupling_v1.py",
    "large_step_transport_client_v1.py",
    "large_step_transport_service_v1.py",
    "large_step_phase1_worker_v1.py",
    "large_step_phase1_verify_v1.py",
    "large_step_phase1_trial_v1.py",
    "adaptive_bp_solve_v1.py",
    "projected_amg_v1.py",
    "packed_storage_v1.py",
    "gpu_allocator_safe_v1.py",
    "gpu_chunked_v1.py",
    "gpu_cached_v1.py",
    "gpu_bp_v1.py",
    "historical_climate_v1.py",
]


def code_hashes():
    return {
        (R / name).relative_to(PROJECT).as_posix(): digest(R / name) for name in CODE
    }


def transition(stage, evidence, fact):
    validate()
    state_path = PROJECT / "LIA_CURRENT_STATE.json"
    state_raw = state_path.read_bytes()
    state = json.loads(state_raw)
    facts_path = PROJECT / "LIA_PROJECT_FACTS.md"
    facts_raw = facts_path.read_bytes()
    assert state["current_focus"]["state"] != "decision-required"
    assert digest(facts_path).upper() == state["scientific_contract"]["sha256"].upper()
    authorized = stage == "authorized"
    if authorized:
        assert NAME not in state["research_branches"]
    else:
        assert state["research_branches"][NAME]["launch_authorized"]
    branch = {
        "status": stage,
        "launch_authorized": authorized,
        "allowed_stages": [
            "one-quarter-year-large-step",
            "combined-technical-check",
        ]
        if authorized
        else [],
        "plan": ref(E / "plan.frozen.json"),
        "evidence": ref(evidence),
        "accepted_count_effect": 0,
        "production_replacement_authorized": False,
        "phase2_authorized": False,
        "baseline_replay_authorized": False,
        "full_stokes_authorized": False,
    }
    number = max(
        map(int, re.findall(r"^## (\d+)\.", facts_raw.decode("utf-8-sig"), re.M))
    ) + 1
    addition = (
        f"\n\n## {number}. System2387 BP large-step phase1 {stage}\n\n"
        f"- {fact} Evidence `{ref(evidence)['path']}`.\n"
    )
    facts_hash = hashlib.sha256(facts_raw + addition.encode()).hexdigest().upper()
    route = read(PROJECT / state["authoritative_route"]["path"])
    suffix = f"+{NAME}-{stage}"
    route["version"] += suffix
    route[NAME] = branch
    if "scientific_contract" in route:
        route["scientific_contract"] = {
            **state["scientific_contract"],
            "sha256": facts_hash,
        }
    route_path = E / f"route-{stage}.json"
    write_json(route_path, route)
    new_state = copy.deepcopy(state)
    new_state["research_branches"][NAME] = branch
    new_state["authoritative_route"] = {
        **ref(route_path),
        "version": route["version"],
        "supersedes": state["authoritative_route"]["version"],
    }
    new_state["scientific_contract"]["sha256"] = facts_hash
    new_state["state_id"] += suffix
    new_state["updated_at"] = datetime.now(timezone.utc).isoformat()
    assert state_path.read_bytes() == state_raw and facts_path.read_bytes() == facts_raw
    temporary = state_path.with_name(f".LIA_CURRENT_STATE.large-step-{os.getpid()}.tmp")
    write_json(temporary, new_state)
    with facts_path.open("ab") as stream:
        stream.write(addition.encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, state_path)
    with (PROJECT / "LIA_DECISION_LOG.md").open("a", encoding="utf-8") as stream:
        stream.write(
            f"\n\n## {new_state['updated_at']} — BP large-step phase1 {stage}\n\n{fact}\n"
        )
    write_json(E / f"{stage}-validation.json", validate())


def prepare():
    validate()
    if E.exists() or RUN.parent.exists():
        raise RuntimeError("phase-1 evidence or output root already exists")
    E.mkdir()
    RUN.mkdir(parents=True)
    source_contract = read(SOURCE / "contract.json")["contract"]
    source_year = read(SOURCE / "year-1482.json")
    source_frame = f"frame-{source_year['step']:06d}"
    st.restore(SOURCE / source_frame, source_contract)
    parameters = read(SOURCE / "config.json")["parameters"]
    gates = {
        "BP_relative": 1e-6,
        "mass_relative": 1e-10,
        "thickness_equation_relative": 1e-8,
        "nonnegative_thickness": True,
        "maximum_bp_calls": 8,
    }
    config = {
        "schema": "polim-large-step-phase1-case-v1",
        "case_id": "system2387-1482-quarter",
        "source_root": str(SOURCE),
        "source_frame": source_frame,
        "start_year": 1482.0,
        "end_year": 1482.25,
        "parameters": parameters,
        "use_diffusion_predictor": True,
        "outer_damping": 0.7,
        "maximum_bp_calls": 8,
        "allocator_fraction": 0.10,
        "gates": gates,
        "launch_authorized": True,
    }
    st.write(RUN / "config.json", config)
    synthetic = subprocess.run(
        [
            sys.executable,
            "-m",
            "polim_cpu.large_step_phase1_verify_v1",
            "--json",
            str(E / "synthetic-verification.json"),
        ],
        cwd=PROJECT,
        check=False,
    )
    if synthetic.returncode != 0:
        raise RuntimeError("phase-1 synthetic verification failed")
    sources = [
        SOURCE / "input.npz",
        SOURCE / "climate.npz",
        SOURCE / "config.json",
        SOURCE / "contract.json",
        SOURCE / "year-1482.json",
        SOURCE / source_frame / "state.npz",
        SOURCE / source_frame / "metadata.json",
        SOURCE / source_frame / "seal.json",
    ]
    plan = {
        "schema": NAME,
        "status": "frozen-before-launch",
        "user_authorization": "Implement phase1; register concrete numerical trials before running and do not repeat completed baselines.",
        "system": 2387,
        "grid_m": 50,
        "scope": "One unique 1482.00-1482.25 backward-Euler thickness/BP coupled step. This is the phase-1 combined numerical check and may be retained as the quarter-step member of a later registered comparison; it is not a long run.",
        "source_state": ref(SOURCE / source_frame / "state.npz"),
        "source_year": 1482.0,
        "end_year": 1482.25,
        "baseline": "Read-only completed v32r4 trajectory; no baseline dynamics replay.",
        "physics": "Unchanged fixed-bed/fixed-A/no-slip float64 BP and original 1482 annual historical TI SMB; BP terminal residual 1e-6.",
        "algorithm": "Local-diffusion initial H guess; exact projected-AMG BP; fixed-velocity backward-Euler upwind finite volume; complementarity non-negativity and ablation limiter; damped outer correction.",
        "isolation": "SciPy nonsymmetric transport runs in a persistent Anaconda helper. The CUDA worker never imports SciPy or CPU BLAS; BP AMG hierarchy keeps its existing isolated helper.",
        "gates": gates,
        "resources": {
            "GPU_workers": 1,
            "allocator_fraction": 0.10,
            "maximum_seconds": 900,
            "estimate_minutes": [2, 10],
        },
        "recovery": "Immutable source adoption and one accepted packed checkpoint. Inject unavailable mutable status and continue. Parent process restores the accepted checkpoint after worker exit. No failed attempt is retried under this fingerprint.",
        "phase2_authorized": False,
        "long_run_authorized": False,
        "baseline_replay_authorized": False,
        "output_root": str(RUN),
        "config": ref(RUN / "config.json"),
        "synthetic_verification": ref(E / "synthetic-verification.json"),
        "implementation": code_hashes(),
        "sources": {
            path.relative_to(PROJECT).as_posix(): digest(path) for path in sources
        },
    }
    write_json(E / "plan.frozen.json", plan)
    contract = seal(plan)
    st.write(RUN / "contract.json", {"contract": contract, "plan": ref(E / "plan.frozen.json")})
    transition(
        "authorized",
        E / "plan.frozen.json",
        "User-authorized phase1 implementation frozen before the first concrete large-step trial. One unique50m system2387 step1482.00-1482.25 reuses the sealed completed1482 state and annual climate without replaying the30day baseline. Implicit conservative upwind thickness, complementarity non-negativity/ablation, optional diffusion initial guess and exact projected-AMG BP correction; BP1e-6,mass1e-10,local equation1e-8,maximum8BP calls. OneGPU10percent allocator,15minute hard bound, isolated SciPy transport helper, immutable accepted checkpoint and status-survival check. No phase2 matrix,long run,default,count or physics change.",
    )


def launch():
    validate()
    plan = read(E / "plan.frozen.json")
    if plan["implementation"] != code_hashes():
        raise RuntimeError("phase-1 implementation changed after registration")
    branch = read(PROJECT / "LIA_CURRENT_STATE.json")["research_branches"][NAME]
    if not branch["launch_authorized"]:
        raise RuntimeError("phase-1 route is not launch-authorized")
    for path, expected in plan["sources"].items():
        if digest(PROJECT / path) != expected:
            raise RuntimeError(f"registered source changed: {path}")
    for item in [plan["config"], plan["synthetic_verification"]]:
        if digest(PROJECT / item["path"]).upper() != item["sha256"].upper():
            raise RuntimeError(f"registered input changed: {item['path']}")
    site = PROJECT / ".runtime/torch-cu128-v1/site-packages"
    windows = Path(os.environ.get("SystemRoot", "C:/Windows"))
    environment = os.environ.copy()
    environment.pop("KMP_DUPLICATE_LIB_OK", None)
    environment.pop("PYTHONHOME", None)
    environment["PATH"] = os.pathsep.join(
        map(str, [site / "torch/lib", windows / "System32", windows])
    )
    environment["PYTHONPATH"] = os.pathsep.join([str(site), str(PROJECT)])
    environment["POLIM_TORCH_SITE"] = str(site)
    command = [
        "D:/Anaconda/envs/SS4/python.exe",
        "-m",
        "polim_cpu.large_step_phase1_worker_v1",
        str(RUN),
    ]
    tag = time.time_ns()
    with (RUN / f"worker-{tag}.log").open("x") as log:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        st.write(
            RUN / f"launch-{tag}.json",
            {
                "pid": process.pid,
                "command": command,
                "start": datetime.now(timezone.utc).isoformat(),
                "output_root": str(RUN),
                "checkpoint": "one immutable accepted large-step state",
                "estimate_minutes": [2, 10],
                "maximum_seconds": 900,
                "stop_conditions": "BP/mass/local-equation/nonnegative gate failure or8BP-call budget",
            },
        )
        process.wait(timeout=960)
    return process.returncode


def finish(returncode):
    plan = read(E / "plan.frozen.json")
    contract = read(RUN / "contract.json")["contract"]
    if (RUN / "completion.json").exists():
        worker = read(RUN / "completion.json")
        state, meta = st.restore(RUN / "frame-000001", contract)
        cross_process = {
            "status": "pass",
            "h_sha256": st.array_hash(state["h"]),
            "u_sha256": st.array_hash(state["u"]),
            "year": meta["year"],
        }
        passed = (
            returncode == 0
            and worker["status"] == "completed"
            and worker["BP_relative"] <= plan["gates"]["BP_relative"]
            and worker["mass_relative"] <= plan["gates"]["mass_relative"]
            and worker["thickness_equation_relative"]
            <= plan["gates"]["thickness_equation_relative"]
            and worker["bp_calls"] <= plan["gates"]["maximum_bp_calls"]
        )
        result = {
            "schema": NAME + "-completion",
            "status": "completed" if passed else "technical-failure",
            "all_gates_passed": passed,
            "simulation_launched": True,
            "unique_step": [1482.0, 1482.25],
            "worker": worker,
            "cross_process_checkpoint_restore": cross_process,
            "baseline_replayed": False,
            "phase2_authorized": False,
            "production_replacement": False,
            "accepted_count_effect": 0,
        }
    else:
        failure = read(RUN / "failure.json") if (RUN / "failure.json").exists() else None
        result = {
            "schema": NAME + "-completion",
            "status": "technical-failure",
            "all_gates_passed": False,
            "simulation_launched": True,
            "returncode": returncode,
            "failure": failure,
            "baseline_replayed": False,
            "phase2_authorized": False,
            "production_replacement": False,
            "accepted_count_effect": 0,
        }
    write_json(E / "completion.json", result)
    if result["status"] == "completed":
        fact = (
            "The registered phase1 unique quarter-year large-step completed and all frozen engineering gates passed. "
            f"BP={result['worker']['BP_relative']:.12g},mass={result['worker']['mass_relative']:.12g},"
            f"local-equation={result['worker']['thickness_equation_relative']:.12g},"
            f"BPcalls={result['worker']['bp_calls']},wall={result['worker']['wall_seconds']:.6f}s. "
            "Synthetic kernels, isolated transport helper, unavailable-status survival and cross-process checkpoint restore passed. "
            "No completed30day baseline replay,phase2 launch,production promotion,count or physics change."
        )
    else:
        fact = "The registered phase1 unique quarter-year large-step ended in a technical failure. Preserve outputs; no automatic retry,phase2 launch,production promotion,count or physics change."
    transition(result["status"], E / "completion.json", fact)
    print(json.dumps(result, indent=2), flush=True)


def main():
    action = sys.argv[-1]
    if action == "prepare":
        prepare()
        return
    if action == "launch":
        finish(launch())
        return
    raise SystemExit("usage: python -m polim_cpu.large_step_phase1_trial_v1 prepare|launch")


if __name__ == "__main__":
    main()
