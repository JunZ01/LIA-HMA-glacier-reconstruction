"""Build corrected batch-39 inversion configs after formal geometry promotion."""

from . import build_synthetic_inversion_batch36_wave1_v1 as base


if __name__ == "__main__":
    base.CASES = {
        3906: ("w600reggeom_batch39_wave1_v1/system_3906/geometry_summary_formal.json", "w600reggeom_batch39_wave1_v1/system_3906/synthetic_gdir", "w613inv3906v2"),
        3907: ("w600reggeom_batch39_wave1_v1/system_3907/geometry_summary_formal.json", "w600reggeom_batch39_wave1_v1/system_3907/synthetic_gdir", "w614inv3907v2"),
        2727: ("w600reggeom_batch39_wave1_v1/system_2727/geometry_summary_formal.json", "w600reggeom_batch39_wave1_v1/system_2727/synthetic_gdir", "w615inv2727v2"),
        558: ("w600reggeom_batch39_wave1_v1/system_558/geometry_summary_formal.json", "w600reggeom_batch39_wave1_v1/system_558/synthetic_gdir", "w616inv558v2"),
        2538: ("w600reggeom_batch39_wave1_v1/system_2538/geometry_summary_formal.json", "w600reggeom_batch39_wave1_v1/system_2538/synthetic_gdir", "w617inv2538v2"),
        2141: ("w600reggeom_batch39_wave1_v1/system_2141/geometry_summary_formal.json", "w600reggeom_batch39_wave1_v1/system_2141/synthetic_gdir", "w618inv2141v2"),
        3185: ("w601reggeom_batch39_wave2_v1/system_3185/geometry_summary_formal.json", "w601reggeom_batch39_wave2_v1/system_3185/synthetic_gdir", "w619inv3185v2"),
        2603: ("w601reggeom_batch39_wave2_v1/system_2603/geometry_summary_formal.json", "w601reggeom_batch39_wave2_v1/system_2603/synthetic_gdir", "w620inv2603v2"),
        2177: ("w601reggeom_batch39_wave2_v1/system_2177/geometry_summary_formal.json", "w601reggeom_batch39_wave2_v1/system_2177/synthetic_gdir", "w621inv2177v2"),
        1642: ("w601reggeom_batch39_wave2_v1/system_1642/geometry_summary_formal.json", "w601reggeom_batch39_wave2_v1/system_1642/synthetic_gdir", "w622inv1642v2"),
        1871: ("w595geom1871v1/geometry_summary.json", "w595geom1871v1/synthetic_gdir", "w623inv1871v2"),
    }
    raise SystemExit(base.main())
