"""Single authorized 30 s continuation after the pure 60 s CFL stop on 334."""

from . import run_system334_zero_mb_connectors_v1 as v1


v1.MINIMUM_TIMESTEP_SECONDS = 30.0
v1.OUTPUT = (
    v1.VALIDATION
    / "lia_191_invalid_outline334_terminal_control_volume_condensation_v1/"
    "zero_mb_connectors_mindt30_v2"
)


if __name__ == "__main__":
    v1.main()
