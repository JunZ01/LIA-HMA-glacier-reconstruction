"""Build standard-inversion configs for batch-37 static passes."""

from . import build_synthetic_inversion_batch36_wave1_v1 as base


if __name__ == "__main__":
    base.CASES = {
        5127: ("w550reggeom_batch37_v1/system_5127/geometry_summary_formal.json", "w550reggeom_batch37_v1/system_5127/synthetic_gdir", "w552inv5127v1"),
        4369: ("w550reggeom_batch37_v1/system_4369/geometry_summary_formal.json", "w550reggeom_batch37_v1/system_4369/synthetic_gdir", "w553inv4369v1"),
        3912: ("w550reggeom_batch37_v1/system_3912/geometry_summary_formal.json", "w550reggeom_batch37_v1/system_3912/synthetic_gdir", "w554inv3912v1"),
        2793: ("w550reggeom_batch37_v1/system_2793/geometry_summary_formal.json", "w550reggeom_batch37_v1/system_2793/synthetic_gdir", "w555inv2793v1"),
        801: ("w550reggeom_batch37_v1/system_801/geometry_summary_formal.json", "w550reggeom_batch37_v1/system_801/synthetic_gdir", "w556inv801v1"),
        2619: ("w550reggeom_batch37_v1/system_2619/geometry_summary_formal.json", "w550reggeom_batch37_v1/system_2619/synthetic_gdir", "w557inv2619v1"),
        1326: ("w550reggeom_batch37_v1/system_1326/geometry_summary_formal.json", "w550reggeom_batch37_v1/system_1326/synthetic_gdir", "w558inv1326v1"),
        709: ("w550reggeom_batch37_v1/system_709/geometry_summary_formal.json", "w550reggeom_batch37_v1/system_709/synthetic_gdir", "w559inv709v1"),
        2541: ("w550reggeom_batch37_v1/system_2541/geometry_summary_formal.json", "w550reggeom_batch37_v1/system_2541/synthetic_gdir", "w560inv2541v1"),
        520: ("w551reggeom520_batch37_v1/system_520/geometry_summary_formal.json", "w551reggeom520_batch37_v1/system_520/synthetic_gdir", "w561inv520v1"),
    }
    raise SystemExit(base.main())
