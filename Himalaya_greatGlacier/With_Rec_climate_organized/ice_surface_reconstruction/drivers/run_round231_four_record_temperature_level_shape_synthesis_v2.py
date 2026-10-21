from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
DRIVER_DIR = (
    ROOT
    / "Himalaya_greatGlacier/With_Rec_climate_organized"
    / "ice_surface_reconstruction/drivers"
)
V1_DRIVER = DRIVER_DIR / "run_round231_four_record_temperature_level_shape_synthesis_v1.py"
V2_CONTRACT = DRIVER_DIR / "round231_four_record_temperature_level_shape_synthesis_v2_contract.md"
V2_OUT = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
    / "round_231_four_record_temperature_level_shape_synthesis_v2"
)
ROUND227_SOURCE = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
    / "round_227_yang2010_pages2k_source_lineage_v3/summary.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    V2_OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(ROUND227_SOURCE.read_text(encoding="utf-8"))
    decision = source.get("evidence", {}).get("decision")
    if decision != "independent_tree_core_collections_supported_same_regional_investigator":
        raise RuntimeError("Unexpected authoritative Round 227 lineage decision")
    bridge = dict(source)
    bridge["decision"] = decision
    bridge["derived_bridge"] = {
        "authoritative_source_path": str(ROUND227_SOURCE),
        "authoritative_source_sha256": sha256(ROUND227_SOURCE),
        "schema_change_only": "evidence.decision copied to root decision",
    }
    bridge_path = V2_OUT / "round227_lineage_schema_bridge.json"
    bridge_path.write_text(json.dumps(bridge, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    spec = importlib.util.spec_from_file_location("round231_v1_for_v2", V1_DRIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load Round 231 v1 implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = V2_OUT
    module.CONTRACT = V2_CONTRACT
    module.ROUND227 = bridge_path
    module.main()


if __name__ == "__main__":
    main()
