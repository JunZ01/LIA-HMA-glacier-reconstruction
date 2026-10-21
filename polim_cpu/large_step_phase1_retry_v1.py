"""Single registered technical retry for the phase-1 large-step trial."""
from __future__ import annotations

from .solver_acceleration_v1 import *
from . import packed_storage_v1 as st
from . import large_step_phase1_trial_v1 as base
import subprocess


NAME = "polim_v37r1_large_step_phase1_v1"
E = PROJECT / "validation_data/results/paleo_surface_validation_v0" / NAME
RUN = R / "runs/v37r1-large-step-phase1-v1/system2387-1482-quarter"
SOURCE = R / "runs/v32r4-historical-1470-1570-v1"
PREVIOUS = PROJECT / "validation_data/results/paleo_surface_validation_v0/polim_v37_large_step_phase1_v1"
CODE = list(base.CODE) + ["large_step_phase1_retry_v1.py"]


def _configure_base():
    base.NAME = NAME
    base.E = E
    base.RUN = RUN
    base.SOURCE = SOURCE
    base.CODE = CODE


def _failure_review():
    review = {
        "schema": "polim-v37-large-step-phase1-failure-review-v1",
        "classification": "identified-outer-iteration-configuration-failure",
        "previous_status": "technical-failure",
        "previous_completion": ref(PREVIOUS / "completion.json"),
        "observed": {
            "wall_seconds": 11.797793999998248,
            "error": "GPUCouplingConvergenceError after the frozen 8 BP calls",
            "accepted_large_steps": 0,
            "baseline_replayed": False,
        },
        "cause": "The frozen damping was 0.7 while the fixed-point tolerance was 1e-8 and the budget was eight BP calls. Even for a constant transport map, the remaining error factor at the eighth evaluation is 0.3^7 = 2.187e-4, so this configuration does not generally permit the frozen tolerance.",
        "technical_repair": "Use a full Picard update (damping 1.0) for the single retry. Physics, source, BP tolerance, mass/local-equation gates, step length and eight-call budget remain unchanged. Persist outer history on any failure.",
        "scientific_contract_changed": False,
        "same_failed_fingerprint_relaunch": False,
        "retry_ordinal": 1,
        "automatic_further_retry": False,
    }
    write_json(PREVIOUS / "failure-review-v1.json", review)
    return review


def prepare():
    _configure_base()
    validate()
    if E.exists() or RUN.parent.exists():
        raise RuntimeError("phase-1 retry evidence or output root already exists")
    if read(PROJECT / "LIA_CURRENT_STATE.json")["research_branches"][
        "polim_v37_large_step_phase1_v1"
    ]["status"] != "technical-failure":
        raise RuntimeError("the reviewed predecessor is not closed as technical-failure")
    review = _failure_review()
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
        "case_id": "system2387-1482-quarter-r1",
        "source_root": str(SOURCE),
        "source_frame": source_frame,
        "start_year": 1482.0,
        "end_year": 1482.25,
        "parameters": parameters,
        "use_diffusion_predictor": True,
        "outer_damping": 1.0,
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
        raise RuntimeError("phase-1 retry synthetic verification failed")
    sources = [
        SOURCE / "input.npz",
        SOURCE / "climate.npz",
        SOURCE / "config.json",
        SOURCE / "contract.json",
        SOURCE / "year-1482.json",
        SOURCE / source_frame / "state.npz",
        SOURCE / source_frame / "metadata.json",
        SOURCE / source_frame / "seal.json",
        PREVIOUS / "completion.json",
        PREVIOUS / "failure-review-v1.json",
    ]
    plan = {
        "schema": NAME,
        "status": "frozen-before-single-technical-retry",
        "technical_retry_ordinal": 1,
        "supersedes": "polim_v37_large_step_phase1_v1",
        "identified_failure": review,
        "system": 2387,
        "grid_m": 50,
        "scope": "One and only technical retry of the 1482.00-1482.25 phase-1 step; no accepted prefix existed in v37.",
        "source_state": ref(SOURCE / source_frame / "state.npz"),
        "baseline": "Read-only completed v32r4 trajectory; no baseline dynamics replay.",
        "unchanged_physics_and_gates": True,
        "repair": "outer_damping 0.7 -> 1.0 and failure-history persistence only",
        "gates": gates,
        "resources": {
            "GPU_workers": 1,
            "allocator_fraction": 0.10,
            "maximum_seconds": 900,
            "estimate_minutes": [2, 10],
        },
        "automatic_further_retry": False,
        "phase2_authorized": False,
        "long_run_authorized": False,
        "baseline_replay_authorized": False,
        "output_root": str(RUN),
        "config": ref(RUN / "config.json"),
        "synthetic_verification": ref(E / "synthetic-verification.json"),
        "implementation": base.code_hashes(),
        "sources": {
            path.relative_to(PROJECT).as_posix(): digest(path) for path in sources
        },
    }
    write_json(E / "plan.frozen.json", plan)
    contract = seal(plan)
    st.write(RUN / "contract.json", {"contract": contract, "plan": ref(E / "plan.frozen.json")})
    base.transition(
        "authorized",
        E / "plan.frozen.json",
        "One identified phase1 technical retry authorized after v37 exhausted8corrections because damping0.7 leaves a constant-map error factor0.3^7=2.187e-4 against the frozen1e-8 gate. Repair uses full Picard update damping1.0 and persists failure history; source1482 state,quarter-year step,physics,BP1e-6,mass1e-10,local equation1e-8,eight-call budget,GPU10percent and isolated helpers unchanged. No accepted prefix or30day baseline is replayed. No further automatic retry,phase2,long run,default,count or physics change.",
    )


def main():
    _configure_base()
    action = sys.argv[-1]
    if action == "prepare":
        prepare()
        return
    if action == "launch":
        base.finish(base.launch())
        return
    raise SystemExit("usage: python -m polim_cpu.large_step_phase1_retry_v1 prepare|launch")


if __name__ == "__main__":
    main()
