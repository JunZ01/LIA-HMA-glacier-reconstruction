from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
DRIVERS = Path(__file__).resolve().parent
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
OUT = LOOP / "round_224_central_himalaya_mxd2026_public_recovery_v3"


def main() -> None:
    source = DRIVERS / "run_round224_central_himalaya_mxd2026_public_recovery_v2.py"
    spec = importlib.util.spec_from_file_location("round224_v2_for_v3", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT
    module.RAW = OUT / "official_sources_unmodified"
    module.CONTRACT = OUT / "ROUND_CONTRACT.md"
    module.DRIVER = Path(__file__).resolve()
    module.URLS = dict(module.URLS)
    module.URLS["zenodo"] = "https://zenodo.org/api/records?q=Central%20Himalayan%20tree-ring%20density%20network&size=25"
    module.main()

    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["version"] = 3
    summary["v1_stop"] = "missing_pypdf_before_network_access"
    summary["v2_stop"] = "zenodo_anonymous_page_size_100_rejected"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    result_path = OUT / "FINAL_RESULT.md"
    result_path.write_text(result_path.read_text(encoding="utf-8").replace("# Round 224 V2 Result", "# Round 224 V3 Result", 1), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
