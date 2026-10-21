"""Build standard-inversion configs for batch-36 wave-2 static passes."""

from . import build_synthetic_inversion_batch36_wave1_v1 as base


if __name__ == "__main__":
    base.CASES = {
        4607: ("w534reggeom_batch36_wave2_v1/system_4607/geometry_summary_formal.json", "w534reggeom_batch36_wave2_v1/system_4607/synthetic_gdir", "w535inv4607v1"),
        5062: ("w534reggeom_batch36_wave2_v1/system_5062/geometry_summary_formal.json", "w534reggeom_batch36_wave2_v1/system_5062/synthetic_gdir", "w536inv5062v1"),
        4492: ("w534reggeom_batch36_wave2_v1/system_4492/geometry_summary_formal.json", "w534reggeom_batch36_wave2_v1/system_4492/synthetic_gdir", "w537inv4492v1"),
    }
    raise SystemExit(base.main())
