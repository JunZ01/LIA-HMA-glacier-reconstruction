"""Build standard-inversion configs for available batch-38 static passes."""

from . import build_synthetic_inversion_batch36_wave1_v1 as base


if __name__ == "__main__":
    base.CASES = {
        4464: ("w563geom4464v1/geometry_summary.json", "w563geom4464v1/synthetic_gdir", "w576inv4464v1"),
        3654: ("w574reggeom_batch38_wave1_v1/system_3654/geometry_summary_formal.json", "w574reggeom_batch38_wave1_v1/system_3654/synthetic_gdir", "w577inv3654v1"),
        3244: ("w574reggeom_batch38_wave1_v1/system_3244/geometry_summary_formal.json", "w574reggeom_batch38_wave1_v1/system_3244/synthetic_gdir", "w578inv3244v1"),
        902: ("w574reggeom_batch38_wave1_v1/system_902/geometry_summary_formal.json", "w574reggeom_batch38_wave1_v1/system_902/synthetic_gdir", "w579inv902v1"),
        1880: ("w574reggeom_batch38_wave1_v1/system_1880/geometry_summary_formal.json", "w574reggeom_batch38_wave1_v1/system_1880/synthetic_gdir", "w580inv1880v1"),
        1645: ("w574reggeom_batch38_wave1_v1/system_1645/geometry_summary_formal.json", "w574reggeom_batch38_wave1_v1/system_1645/synthetic_gdir", "w581inv1645v1"),
        2611: ("w575reggeom_batch38_wave2_v1/system_2611/geometry_summary_formal.json", "w575reggeom_batch38_wave2_v1/system_2611/synthetic_gdir", "w582inv2611v1"),
        1743: ("w575reggeom_batch38_wave2_v1/system_1743/geometry_summary_formal.json", "w575reggeom_batch38_wave2_v1/system_1743/synthetic_gdir", "w583inv1743v1"),
    }
    raise SystemExit(base.main())
