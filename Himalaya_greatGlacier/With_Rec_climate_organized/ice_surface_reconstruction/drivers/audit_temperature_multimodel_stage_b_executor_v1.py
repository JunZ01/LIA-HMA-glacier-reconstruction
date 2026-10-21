from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from run_temperature_multimodel_stage_b_pseudoproxy_v2 import SharedObservationNoise


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
PREP = LOOP / "temperature_multimodel_evolving_prior_stage_b_pseudoproxy_preregistration_v2"
INPUT = PREP / "stage_b_pseudoproxy_input_v1.npz"
RUNNER_V1 = ROOT / (
    "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/"
    "drivers/run_temperature_multimodel_stage_b_pseudoproxy_v1.py"
)
RUNNER_V2 = ROOT / (
    "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/"
    "drivers/run_temperature_multimodel_stage_b_pseudoproxy_v2.py"
)
PLANNED_OUTPUT = LOOP / "temperature_multimodel_evolving_prior_stage_b_pseudoproxy_v2"
OUT = LOOP / "temperature_multimodel_evolving_prior_stage_b_executor_audit_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    sources = [INPUT, PREP / "summary.json", PREP / "STAGE_B_CONTRACT.md", RUNNER_V1, RUNNER_V2]
    before = {str(path): sha256(path) for path in sources}
    ast.parse(RUNNER_V1.read_text(encoding="utf-8"), filename=str(RUNNER_V1))
    ast.parse(RUNNER_V2.read_text(encoding="utf-8"), filename=str(RUNNER_V2))

    data = np.load(INPUT, allow_pickle=False)
    keys = set(data.files)
    years = np.asarray(data["evaluation_years"], dtype=int)
    forbidden_key_fragments = ("observation", "instrument", "station", "cru", "berkeley")
    forbidden_keys = sorted(
        key for key in keys if any(fragment in key.lower() for fragment in forbidden_key_fragments)
    )

    rng = SharedObservationNoise(20260827)
    draws = [rng.normal(0.0, np.ones(5), (3, 5)) for _ in range(6)]
    shared_noise_exact = bool(
        np.array_equal(draws[0], draws[2])
        and np.array_equal(draws[0], draws[4])
        and np.array_equal(draws[1], draws[3])
        and np.array_equal(draws[1], draws[5])
    )

    array_shapes = {key: list(np.asarray(data[key]).shape) for key in sorted(keys)}
    total_uncompressed_bytes = int(sum(np.asarray(data[key]).nbytes for key in keys))
    runner_text = RUNNER_V1.read_text(encoding="utf-8") + RUNNER_V2.read_text(encoding="utf-8")
    gates = {
        "preparation_summary_pass": bool(
            json.loads((PREP / "summary.json").read_text(encoding="utf-8"))["preparation_pass"]
        ),
        "runner_v1_ast_valid": True,
        "runner_v2_ast_valid": True,
        "planned_output_absent_before_authorization": not PLANNED_OUTPUT.exists(),
        "evaluation_years_exactly_1515_1833": np.array_equal(years, np.arange(1515, 1834)),
        "sealed_years_absent": not np.any((years >= 1921) & (years <= 1939)),
        "no_real_observation_input_keys": len(forbidden_keys) == 0,
        "truth_state_shape_4x319x333": tuple(data["truth_temperature_anomaly"].shape) == (4, 319, 333),
        "proxy6_truth_operator_shape": tuple(data["proxy6_truth_operator_anomaly"].shape) == (4, 319, 201),
        "proxy9_truth_operator_shape": tuple(data["proxy9_truth_operator_anomaly"].shape) == (4, 319, 290),
        "shared_observation_noise_exact_across_three_priors": shared_noise_exact,
        "no_matlab_M_update_call": "M_update" not in runner_text,
        "no_localization_branch": "localization" not in runner_text.lower(),
        "no_production_path_in_runner": "out_data/climate_anomaly_05deg" not in runner_text.replace("\\", "/"),
    }
    audit_pass = all(gates.values())
    summary = {
        "experiment": OUT.name,
        "status": "complete",
        "audit_pass": audit_pass,
        "gates": gates,
        "input_array_shapes": array_shapes,
        "input_uncompressed_bytes": total_uncompressed_bytes,
        "forbidden_input_keys": forbidden_keys,
        "runner_authorized": False,
        "runner_executed": False,
        "enkf_run": False,
        "next_action": "explicit_user_authorization_required_for_stage_b_enkf",
        "sealed_1921_1939_values_used": False,
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUT / "FINAL_RESULT.md").write_text(
        "# Stage B executor static audit\n\n"
        f"Audit pass: `{audit_pass}`.\n\n"
        + "\n".join(f"- `{name}`: `{value}`" for name, value in gates.items())
        + "\n\nThe runner was not executed. Explicit user authorization remains required.\n",
        encoding="utf-8",
    )
    after = {str(path): sha256(path) for path in sources}
    if before != after:
        raise RuntimeError("source-changed")
    if not audit_pass:
        raise RuntimeError("executor-static-audit-failed")


if __name__ == "__main__":
    main()
