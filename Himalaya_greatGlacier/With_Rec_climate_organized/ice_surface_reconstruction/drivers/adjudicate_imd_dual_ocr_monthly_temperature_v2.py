from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
V1_PATH = HERE / "adjudicate_imd_dual_ocr_monthly_temperature_v1.py"
SPEC = importlib.util.spec_from_file_location("imd_adjudication_v1", V1_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {V1_PATH}")
V1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V1)

V1.IA = V1.LOOP / "imd_ia_fulltext_monthly_values_v5/monthly_extraction.csv"
V1.OUT = V1.LOOP / "imd_dual_ocr_monthly_temperature_adjudication_v2"
V1.DRIVER = Path(__file__).resolve()


def main() -> None:
    V1.main()
    path = V1.OUT / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["experiment"] = "imd_dual_ocr_monthly_temperature_adjudication_v2"
    summary["supersedes"] = "imd_dual_ocr_monthly_temperature_adjudication_v1"
    summary["ia_input"] = "imd_ia_fulltext_monthly_values_v5"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
