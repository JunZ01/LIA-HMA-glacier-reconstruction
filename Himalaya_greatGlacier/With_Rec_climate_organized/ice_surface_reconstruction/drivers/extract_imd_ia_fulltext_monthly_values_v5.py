from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
V4_PATH = HERE / "extract_imd_ia_fulltext_monthly_values_v4.py"
SPEC = importlib.util.spec_from_file_location("imd_ia_v4", V4_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {V4_PATH}")
V4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V4)
V1 = V4.V1

V1.OUT = V1.LOOP / "imd_ia_fulltext_monthly_values_v5"
V1.DRIVER = Path(__file__).resolve()
FALLBACK = V1.LOOP / "temperature_historical_imd_direct_station_table_leaf_audit_v4/raw_api"
PRIMARY_TABLE_TEXT = V1.table_text


def table_text(item_id: str, station: str, leaf: int):
    text, status = PRIMARY_TABLE_TEXT(item_id, station, leaf)
    if text:
        return text, status
    path = FALLBACK / f"{item_id}_{station.lower()}_station.json"
    if not path.exists():
        return "", status + ";fallback_missing"
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = []
    for match in payload.get("matches", []):
        leaves = {int(par.get("page", -1)) for par in (match.get("par") or [])}
        candidate = str(match.get("text", ""))
        if leaf in leaves and sum(month in candidate.lower() for month in V1.MONTHS) >= 8:
            matches.append(candidate)
    if not matches:
        return "", status + ";fallback_table_match_missing"
    return max(matches, key=len), "located_leaf_audit_fallback"


V1.table_text = table_text
V1.parse_temperatures = V4.parse_temperatures
V1.split_station_group = V4.V3.split_station_group


def main() -> None:
    V1.main()
    path = V1.OUT / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["experiment"] = "imd_ia_fulltext_monthly_values_v5"
    summary["supersedes"] = "imd_ia_fulltext_monthly_values_v4"
    summary["leaf_audit_api_fallback"] = str(FALLBACK)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
