from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "prepare_imd_ghcn_overlap_station_page_panel_v1.py"
SPEC = importlib.util.spec_from_file_location("imd_ghcn_overlap_v1", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SOURCE}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

MODULE.OUT = MODULE.LOOP / "imd_ghcn_overlap_pages_v2"
MODULE.DRIVER = Path(__file__).resolve()


def main() -> None:
    MODULE.main()
    summary_path = MODULE.OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["experiment"] = "imd_ghcn_overlap_pages_v2"
    summary["supersedes"] = (
        "temperature_historical_imd_ghcn_overlap_page_panel_v1_interrupted_"
        "before_download_due_to_windows_path_length"
    )
    summary["path_length_fix"] = True
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
