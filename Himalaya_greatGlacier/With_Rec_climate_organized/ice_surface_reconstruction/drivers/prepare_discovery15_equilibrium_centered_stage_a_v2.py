"""Run the one authorized technical retry for the 15 static-admissible members."""

from . import prepare_discovery16_equilibrium_centered_stage_a_v1 as stage_a


stage_a.RUN_VERSION = "v2"
stage_a.RUN_SYSTEMS = [
    988,
    2086,
    3185,
    801,
    4959,
    2534,
    2570,
    705,
    2727,
    4491,
    3016,
    3660,
    2177,
    5770,
    2559,
]
stage_a.OUT = stage_a.BASE / "lia_191_discovery15_equilibrium_centered_stage_a_v2"
stage_a.RUNTIME_ROOT = stage_a.ROOT / "validation_data/lia_runtime/discovery15_stage_a_v2"
stage_a.EXPECTED_STATE_ID = "2026-09-01-v10.185-discovery15-stage-a-v2-prelaunch-binding"
stage_a.EXPECTED_NEXT_ACTION = "retry-discovery15-equilibrium-centered-planform2d-stage-a-v2"


if __name__ == "__main__":
    raise SystemExit(stage_a.main())
