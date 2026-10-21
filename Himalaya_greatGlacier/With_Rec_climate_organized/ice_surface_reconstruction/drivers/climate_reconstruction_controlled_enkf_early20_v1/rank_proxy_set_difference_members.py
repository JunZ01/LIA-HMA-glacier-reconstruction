"""Rank individual proxy-set additions and shared-proxy reweighting terms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, center=True, min_periods=window).mean().to_numpy()


def _trend(years: np.ndarray, values: np.ndarray) -> float:
    x = years.astype(float) - float(np.mean(years))
    return float(100.0 * np.polyfit(x, values, 1)[0])


def _projection(values: np.ndarray, reference: np.ndarray) -> float:
    x = _rolling(values, 15)
    y = _rolling(reference, 15)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid] - np.mean(x[valid])
    y = y[valid] - np.mean(y[valid])
    return float(x @ y / (y @ y))


def _add_match_key(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy().sort_values("proxy_index_1based")
    columns = ["latitude", "longitude", "proxy_type", "psm_method", "b0", "b1", "b2"]
    normalized = frame[columns].copy()
    for column in ("latitude", "longitude", "b0", "b1", "b2"):
        normalized[column] = pd.to_numeric(normalized[column]).round(8).astype(str)
    signature = normalized.astype(str).agg("|".join, axis=1)
    frame["match_key"] = signature + "|occ=" + signature.groupby(signature).cumcount().astype(str)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=None)
    args = parser.parse_args()
    root = _project_root()
    output = args.result_root or root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_controlled_enkf_early20_v1"
    ranking = pd.read_csv(output / "proxy_contribution_ranking_1901_1950.csv")
    arrays = np.load(output / "controlled_enkf_attribution_arrays.npz")
    years = arrays["years"]
    selected = (years >= 1901) & (years <= 1950)
    xyears = years[selected]
    reference = arrays["rec_ref_target"][selected]
    c6 = arrays["proxy6_target_contributions"][:, selected]
    c9 = arrays["proxy9_target_contributions"][:, selected]
    proxy6 = _add_match_key(ranking[ranking.branch.eq("proxy6")])
    proxy9 = _add_match_key(ranking[ranking.branch.eq("proxy9")])
    rows6 = {row.match_key: row for row in proxy6.itertuples()}
    rows9 = {row.match_key: row for row in proxy9.itertuples()}
    keys6 = set(rows6)
    keys9 = set(rows9)
    rows = []

    def append(category: str, key: str, values: np.ndarray, row6: object | None, row9: object | None) -> None:
        metadata = row9 if row9 is not None else row6
        rows.append({
            "category": category,
            "proxy6_index_1based": int(row6.proxy_index_1based) if row6 is not None else np.nan,
            "proxy9_index_1based": int(row9.proxy_index_1based) if row9 is not None else np.nan,
            "latitude": float(metadata.latitude),
            "longitude": float(metadata.longitude),
            "distance_to_3584_km": float(metadata.distance_to_3584_km),
            "proxy_type": str(metadata.proxy_type),
            "psm_method": str(metadata.psm_method),
            "trend_component_c_per_century": _trend(xyears, values),
            "rolling15_reference_projection": _projection(values, reference),
            "component_std_c": float(np.std(values)),
            "proxy6_mean_innovation": float(row6.mean_innovation_1901_1950) if row6 is not None else np.nan,
            "proxy9_mean_innovation": float(row9.mean_innovation_1901_1950) if row9 is not None else np.nan,
            "b0": float(metadata.b0),
            "b1": float(metadata.b1),
            "b2": float(metadata.b2),
        })

    for key in sorted(keys6 & keys9):
        row6 = rows6[key]
        row9 = rows9[key]
        append("shared_proxy_reweighting", key, c9[int(row9.proxy_index_1based) - 1] - c6[int(row6.proxy_index_1based) - 1], row6, row9)
    for key in sorted(keys9 - keys6):
        row9 = rows9[key]
        append("proxy9_added_record", key, c9[int(row9.proxy_index_1based) - 1], None, row9)
    for key in sorted(keys6 - keys9):
        row6 = rows6[key]
        append("proxy6_only_removed", key, -c6[int(row6.proxy_index_1based) - 1], row6, None)

    table = pd.DataFrame(rows)
    table["absolute_trend_component"] = table.trend_component_c_per_century.abs()
    table.sort_values(["category", "absolute_trend_component"], ascending=[True, False]).to_csv(output / "proxy_set_difference_member_ranking.csv", index=False)
    summary = {
        category: group.nlargest(10, "absolute_trend_component").drop(columns="absolute_trend_component").to_dict(orient="records")
        for category, group in table.groupby("category")
    }
    (output / "proxy_set_difference_member_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
