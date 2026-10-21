"""Audit western/eastern independent annual mean-bias signs in degrees C."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import build_temperature_regional_reference_field_nested_cv_v1 as field_tools
import validate_temperature_external_low_frequency_v1 as base


PUBLIC = field_tools.PUBLIC
OUTPUT = base.RESULTS / "climate_reconstruction_temperature_annual_anchor_bias_audit_v1"
CHONGCE = field_tools.LAKE_ICE / "chongce_zenodo_5766235.xlsx"
HAIZI = field_tools.RECENT / "41598_2024_79096_MOESM1_ESM.xlsx"


def audit(project: np.ndarray, external: pd.DataFrame, modern_start: int, modern_end: int) -> tuple[pd.DataFrame, float]:
    frame = pd.DataFrame({"year": field_tools.YEARS, "project": project})
    ext = external.copy(); ext["year"] = ext.year.astype(int)
    merged = frame.merge(ext, on="year")
    modern = merged.year.between(modern_start, modern_end)
    merged["project_aligned"] = merged.project - merged.loc[modern, "project"].mean()
    merged["external_aligned"] = merged.external - merged.loc[modern, "external"].mean()
    rows = []
    for start in range(1550, 1900, 50):
        sub = merged.loc[merged.year.between(start, start + 49)]
        if len(sub) < 35:
            continue
        p = float(sub.project_aligned.mean()); e = float(sub.external_aligned.mean())
        rows.append({"window_start": start, "window_end": start + 49, "project_degC": p, "external_degC": e, "project_minus_external_degC": p-e})
    table = pd.DataFrame(rows)
    return table, float(table.project_minus_external_degC.mean())


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    paths = {"public": PUBLIC, "chongce": CHONGCE, "haizi": HAIZI, "contract": contract}
    before = {name: base.sha256(path) for name, path in paths.items()}
    chongce = field_tools.read_xlsx(CHONGCE, 1, 3, 6, 1524, 2012)
    haizi = field_tools.read_xlsx(HAIZI, 1, 3, 2, 1115, 2017)
    with h5py.File(PUBLIC, "r") as handle:
        lat = np.asarray(handle["lat"], dtype=float); lon = np.asarray(handle["lon"], dtype=float)
        project_field = field_tools.allowed_field(handle["tas_ensemble_mean_rec_result"])
    project_chongce = field_tools.local_box(project_field, lat, lon, (80.75,81.75,34.75,35.75))
    project_haizi = field_tools.local_box(project_field, lat, lon, (99.75,101.25,28.75,30.25))
    west, west_bias = audit(project_chongce, chongce, 1940, 2000)
    east, east_bias = audit(project_haizi, haizi, 1959, 2000)
    west.insert(0, "record", "Chongce_annual"); east.insert(0, "record", "Haizi_annual")
    pd.concat([west, east], ignore_index=True).to_csv(OUTPUT / "annual_anchor_50year_biases.csv", index=False)
    opposite = bool(np.sign(west_bias) != np.sign(east_bias))
    result = {
        "status": "complete",
        "chongce_project_minus_external_mean_degC": west_bias,
        "haizi_project_minus_external_mean_degC": east_bias,
        "opposite_bias_signs": opposite,
        "uniform_same_sign_correction_supported": not opposite,
        "locked_1921_1939_accessed": False,
        "phase_1901_1920_used": False,
        "correction_fitted": False,
    }
    after = {name: base.sha256(path) for name, path in paths.items()}
    (OUTPUT / "source_manifest.json").write_text(json.dumps({"inputs":{name:{"path":str(paths[name]),"sha256_before":before[name],"sha256_after":after[name]} for name in paths},"all_input_hashes_unchanged":before==after},indent=2),encoding="utf-8")
    (OUTPUT / "summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text("# Independent Annual-Anchor Bias Audit\n\n"+f"- Chongce project-minus-record: `{west_bias:+.3f} degC`.\n"+f"- Haizi project-minus-record: `{east_bias:+.3f} degC`.\n"+f"- Opposite signs: `{str(opposite).lower()}`.\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
