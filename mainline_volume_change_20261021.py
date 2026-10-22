"""Ice volume and area change from the accepted mainline dynamic runs.

Read-only. For every system in the accepted-32 source manifest:
  * verify the curve file against its registered SHA256;
  * use forward rows only (calendar years 1470-2000; rows without a phase
    column are forward by construction);
  * LIA endpoint (primary): the first forward year at which
    observation_length_m reaches its maximum -- the same rule the production
    scorer uses for forward_peak_year, i.e. the state whose length was
    accepted against the LIA moraine target;
  * modern endpoint: forward year 2000;
  * guard: e_lia and e_rgi are recomputed from these endpoints and the
    registry targets and must match the retained-32 registry, so the
    endpoints are the ones that were accepted;
  * sensitivity: LIA endpoint at the forward volume maximum instead.
Optionally, the modern model volume is divided by the modern reference volume
(descriptive only, never used to tune anything).

No model run, recalibration or reconstruction.

Local schema compatibility (2026-10-22): historical exports that lack
observation_length_m use their saved length_m column. The unchanged registry
endpoint guard still applies; the selected column is reported per system.

Usage (repo root):
    python mainline_volume_change_20261021.py MANIFEST_CSV REGISTRY_CSV OUT_DIR [REFERENCE_PAIRING_CSV]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODERN_YEAR = 2000
GUARD_TOLERANCE = 1e-6


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def forward_rows(table: pd.DataFrame) -> pd.DataFrame:
    if "phase" in table.columns:
        table = table[table["phase"].astype(str) == "forward"]
    return table.sort_values("year").reset_index(drop=True)


def analyse(row: pd.Series, registry: pd.Series | None, root: Path | None) -> dict:
    out = {"system_id": int(row.system_id), "rgi_id": row.rgi_id, "route": row.route,
           "source_filename": row.source_filename}
    path = Path(row.curve_source_absolute)
    if not path.exists() and root is not None:
        path = root / str(row.curve_source).replace("\\", "/")
    if not path.exists():
        return {**out, "status": "missing-file"}
    if sha256(path) != str(row.curve_source_sha256).upper():
        return {**out, "status": "sha256-mismatch"}
    f = forward_rows(pd.read_csv(path))
    if "observation_length_m" in f.columns:
        out["length_source_column"] = "observation_length_m"
    elif "length_m" in f.columns:
        out["length_source_column"] = "length_m"
        f = f.assign(observation_length_m=f["length_m"])
    required = {"year", "volume_m3", "area_m2", "observation_length_m"}
    missing = sorted(required - set(f.columns))
    if missing:
        return {**out, "status": "missing-columns:" + ",".join(missing)}
    if "observation_domain_overflow" in f.columns and f["observation_domain_overflow"].astype(bool).any():
        return {**out, "status": "observation-domain-overflow"}
    if int(f.year.iloc[-1]) != MODERN_YEAR:
        return {**out, "status": f"last-forward-year-{int(f.year.iloc[-1])}"}
    length = pd.to_numeric(f.observation_length_m, errors="coerce")
    lia = f.loc[length.idxmax()]
    modern = f.iloc[-1]
    vmax = f.loc[pd.to_numeric(f.volume_m3).idxmax()]
    out.update({
        "status": "ok",
        "lia_year_length_peak": int(lia.year),
        "lia_length_m": float(lia.observation_length_m),
        "lia_volume_km3": float(lia.volume_m3) / 1e9,
        "lia_area_km2": float(lia.area_m2) / 1e6,
        "modern_length_m": float(modern.observation_length_m),
        "modern_volume_km3": float(modern.volume_m3) / 1e9,
        "modern_area_km2": float(modern.area_m2) / 1e6,
        "volume_max_year": int(vmax.year),
        "volume_max_km3": float(vmax.volume_m3) / 1e9,
    })
    out["dV_km3"] = out["modern_volume_km3"] - out["lia_volume_km3"]
    out["dV_percent"] = 100 * out["dV_km3"] / out["lia_volume_km3"]
    out["dA_km2"] = out["modern_area_km2"] - out["lia_area_km2"]
    out["dA_percent"] = 100 * out["dA_km2"] / out["lia_area_km2"]
    out["lia_mean_thickness_m"] = 1000 * out["lia_volume_km3"] / out["lia_area_km2"]
    out["modern_mean_thickness_m"] = 1000 * out["modern_volume_km3"] / out["modern_area_km2"]
    out["dV_km3_if_lia_at_volume_max"] = out["modern_volume_km3"] - out["volume_max_km3"]
    out["volume_peak_minus_length_peak_years"] = out["volume_max_year"] - out["lia_year_length_peak"]
    if registry is not None:
        lia_m, rgi_m = 1000 * float(registry.lia_km), 1000 * float(registry.rgi_km)
        sep = abs(lia_m - rgi_m)
        e_lia = abs(out["lia_length_m"] - lia_m) / sep
        e_rgi = abs(out["modern_length_m"] - rgi_m) / sep
        out["e_lia_recomputed"], out["e_rgi_recomputed"] = e_lia, e_rgi
        out["endpoint_guard_pass"] = bool(abs(e_lia - float(registry.e_lia)) <= GUARD_TOLERANCE
                                          and abs(e_rgi - float(registry.e_rgi)) <= GUARD_TOLERANCE)
    else:
        out["endpoint_guard_pass"] = False
    return out


def main() -> None:
    manifest = pd.read_csv(sys.argv[1], encoding="utf-8-sig")
    registry = pd.read_csv(sys.argv[2], encoding="utf-8-sig").set_index("system_id")
    out_dir = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=False)
    reference = None
    if len(sys.argv) > 4:
        ref = pd.read_csv(sys.argv[4], encoding="utf-8-sig")
        reference = ref.set_index("lia_unique_id")["modern_reference_native_sum_km3"]
    root = Path("E:/3spyderProject/1_LIA_himalaya_oggm")
    rows = []
    for _, row in manifest.iterrows():
        reg = registry.loc[int(row.system_id)] if int(row.system_id) in registry.index else None
        result = analyse(row, reg, root)
        if reference is not None and result.get("status") == "ok" and result["system_id"] in reference.index:
            result["modern_reference_volume_km3"] = float(reference.loc[result["system_id"]])
            result["modern_model_over_reference"] = result["modern_volume_km3"] / result["modern_reference_volume_km3"]
        rows.append(result)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "mainline_volume_change_by_system.csv", index=False, encoding="utf-8-sig")
    ok = table[(table.status == "ok") & table.endpoint_guard_pass]
    summary = {
        "status": "completed",
        "systems_in_manifest": int(len(table)),
        "systems_used": int(len(ok)),
        "length_source_columns": table.get("length_source_column", pd.Series(dtype=str)).value_counts().to_dict(),
        "excluded": table.loc[~table.index.isin(ok.index), ["system_id", "status", "endpoint_guard_pass"]].to_dict("records"),
        "lia_endpoint": "first forward year of maximum observation_length_m (accepted LIA length peak)",
        "modern_endpoint": f"forward year {MODERN_YEAR}",
        "sum_lia_volume_km3": float(ok.lia_volume_km3.sum()),
        "sum_modern_volume_km3": float(ok.modern_volume_km3.sum()),
        "sum_dV_km3": float(ok.dV_km3.sum()),
        "sum_dV_percent_of_lia": float(100 * ok.dV_km3.sum() / ok.lia_volume_km3.sum()) if len(ok) else np.nan,
        "sum_dA_km2": float(ok.dA_km2.sum()),
        "sum_dA_percent_of_lia": float(100 * ok.dA_km2.sum() / ok.lia_area_km2.sum()) if len(ok) else np.nan,
        "median_dV_percent": float(ok.dV_percent.median()) if len(ok) else np.nan,
        "systems_gaining_volume": [int(s) for s in ok.loc[ok.dV_km3 > 0, "system_id"]],
        "sensitivity_sum_dV_km3_if_lia_at_volume_max": float(ok.dV_km3_if_lia_at_volume_max.sum()),
        "median_modern_model_over_reference": (float(ok.modern_model_over_reference.median())
                                               if "modern_model_over_reference" in ok else None),
        "scope": "accepted mainline dynamic states only; same model and bed at both endpoints; length-calibrated, not volume-calibrated; descriptive, no new run",
    }
    (out_dir / "mainline_volume_change_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "excluded"}, indent=2, ensure_ascii=False))
    if summary["excluded"]:
        print("EXCLUDED:", json.dumps(summary["excluded"], ensure_ascii=False))


if __name__ == "__main__":
    main()
