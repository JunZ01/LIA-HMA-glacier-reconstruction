from pathlib import Path

root = Path(__file__).resolve().parent
router = root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/build_lia_191_iterative_loop.py"
text = router.read_text(encoding="utf-8")
old = '''    layer1_stable = layer1[layer1["stable_chain"]].copy()
    delta_boundary_hit = False
'''
new = '''    layer1_stable = layer1[layer1["stable_chain"]].copy()
    layer1_extension_complete = bool(
        layer1["score_source"].astype(str).eq("delta_boundary_extension_v1").any()
    )
    delta_boundary_hit = False
'''
if text.count(old) != 1:
    raise RuntimeError(f"extension-complete insertion count={text.count(old)}")
text = text.replace(old, new)

old = '''    elif delta_boundary_hit:
        route = "layer1_delta_extension"
        reason = "best stable layer-1 delta-T sits on the tested boundary"
    else:
        route = "hold_after_layer1"
        reason = "current layer-1 results show no direct pass, no explicit two-endpoint bracket, and no boundary case"
'''
new = '''    elif delta_boundary_hit and not layer1_extension_complete:
        route = "layer1_delta_extension"
        reason = "best stable layer-1 delta-T sits on the tested boundary and the one bounded extension is not yet used"
    elif layer1_extension_complete:
        route = "hold_after_layer1"
        reason = "the one bounded layer-1 delta-T extension is complete with no direct pass or explicit two-endpoint bracket"
    else:
        route = "hold_after_layer1"
        reason = "current layer-1 results show no direct pass, no explicit two-endpoint bracket, and no boundary case"
'''
if text.count(old) != 1:
    raise RuntimeError(f"route replacement count={text.count(old)}")
text = text.replace(old, new)

old = '''        "delta_boundary_hit": bool(delta_boundary_hit),
        "direct_candidate_exists": bool(not direct.empty),
'''
new = '''        "delta_boundary_hit": bool(delta_boundary_hit),
        "layer1_extension_complete": bool(layer1_extension_complete),
        "direct_candidate_exists": bool(not direct.empty),
'''
if text.count(old) != 1:
    raise RuntimeError(f"output insertion count={text.count(old)}")
router.write_text(text.replace(old, new), encoding="utf-8")
