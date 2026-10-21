"""Build the missed system-551 standard-inversion config."""

from . import build_synthetic_inversion_batch36_wave1_v1 as base


if __name__ == "__main__":
    base.CASES = {
        551: (
            "w625reggeom551_recovery_v1/system_551/geometry_summary_formal.json",
            "w625reggeom551_recovery_v1/system_551/synthetic_gdir",
            "w626inv551_recovery_v1",
        )
    }
    raise SystemExit(base.main())
