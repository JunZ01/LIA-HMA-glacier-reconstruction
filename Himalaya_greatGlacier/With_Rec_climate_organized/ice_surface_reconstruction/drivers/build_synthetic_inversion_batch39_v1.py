"""Build standard-inversion configs for all batch-39 static passes."""

from . import build_synthetic_inversion_batch36_wave1_v1 as base


if __name__ == "__main__":
    base.CASES = {
        3906: ("w600reggeom_batch39_wave1_v1/system_3906/geometry_summary_formal.json", "w600reggeom_batch39_wave1_v1/system_3906/synthetic_gdir", "w602inv3906v1"),
        3907: ("w600reggeom_batch39_wave1_v1/system_3907/geometry_summary_formal.json", "w600reggeom_batch39_wave1_v1/system_3907/synthetic_gdir", "w603inv3907v1"),
        2727: ("w600reggeom_batch39_wave1_v1/system_2727/geometry_summary_formal.json", "w600reggeom_batch39_wave1_v1/system_2727/synthetic_gdir", "w604inv2727v1"),
        558: ("w600reggeom_batch39_wave1_v1/system_558/geometry_summary_formal.json", "w600reggeom_batch39_wave1_v1/system_558/synthetic_gdir", "w605inv558v1"),
        2538: ("w600reggeom_batch39_wave1_v1/system_2538/geometry_summary_formal.json", "w600reggeom_batch39_wave1_v1/system_2538/synthetic_gdir", "w606inv2538v1"),
        2141: ("w600reggeom_batch39_wave1_v1/system_2141/geometry_summary_formal.json", "w600reggeom_batch39_wave1_v1/system_2141/synthetic_gdir", "w607inv2141v1"),
        3185: ("w601reggeom_batch39_wave2_v1/system_3185/geometry_summary_formal.json", "w601reggeom_batch39_wave2_v1/system_3185/synthetic_gdir", "w608inv3185v1"),
        2603: ("w601reggeom_batch39_wave2_v1/system_2603/geometry_summary_formal.json", "w601reggeom_batch39_wave2_v1/system_2603/synthetic_gdir", "w609inv2603v1"),
        2177: ("w601reggeom_batch39_wave2_v1/system_2177/geometry_summary_formal.json", "w601reggeom_batch39_wave2_v1/system_2177/synthetic_gdir", "w610inv2177v1"),
        1642: ("w601reggeom_batch39_wave2_v1/system_1642/geometry_summary_formal.json", "w601reggeom_batch39_wave2_v1/system_1642/synthetic_gdir", "w611inv1642v1"),
        1871: ("w595geom1871v1/geometry_summary.json", "w595geom1871v1/synthetic_gdir", "w612inv1871v1"),
    }
    raise SystemExit(base.main())
