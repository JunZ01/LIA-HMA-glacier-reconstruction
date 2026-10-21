from pathlib import Path

root = Path(__file__).resolve().parent
router = root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/build_lia_191_iterative_loop.py"
text = router.read_text(encoding="utf-8")
old = '''        ScoreSource(
            base / "lia_191_curve_loop_delta_refine_1077_r1_v1" / "scores.csv",
            "delta_refine_1077_r1",
            "layer1_delta",
            2,
        ),
        ScoreSource(
            base / "lia_191_tradeoff_pilot_v1" / "scores.csv",'''
new = '''        ScoreSource(
            base / "lia_191_curve_loop_delta_refine_1077_r1_v1" / "scores.csv",
            "delta_refine_1077_r1",
            "layer1_delta",
            2,
        ),
        ScoreSource(
            base / "lia_191_delta_boundary_extension_v1" / "scores.csv",
            "delta_boundary_extension_v1",
            "layer1_delta",
            3,
        ),
        ScoreSource(
            base / "lia_191_tradeoff_pilot_v1" / "scores.csv",'''
if text.count(old) != 1:
    raise RuntimeError(f"expected one router insertion, found {text.count(old)}")
router.write_text(text.replace(old, new), encoding="utf-8")
