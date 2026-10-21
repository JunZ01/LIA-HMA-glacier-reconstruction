"""Build standard-inversion configs for the final batch-38 static passes."""
from . import build_synthetic_inversion_batch36_wave1_v1 as base

if __name__ == "__main__":
    base.CASES = {
        2689: ("w584reggeom2689_batch38_v1/system_2689/geometry_summary_formal.json", "w584reggeom2689_batch38_v1/system_2689/synthetic_gdir", "w586inv2689v1"),
        553: ("w585reggeom553_batch38_v1/system_553/geometry_summary_formal.json", "w585reggeom553_batch38_v1/system_553/synthetic_gdir", "w587inv553v1"),
    }
    raise SystemExit(base.main())
