"""Decompose proxy9-minus-proxy6 into added-proxy and reweighting terms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def _projection(values: np.ndarray, reference: np.ndarray, window: int) -> float:
    x = _rolling(values, window)
    y = _rolling(reference, window)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid] - np.mean(x[valid])
    y = y[valid] - np.mean(y[valid])
    return float(x @ y / (y @ y))


def _signature(frame: pd.DataFrame) -> pd.Series:
    columns = ["latitude", "longitude", "proxy_type", "psm_method", "b0", "b1", "b2"]
    normalized = frame[columns].copy()
    for column in ("latitude", "longitude", "b0", "b1", "b2"):
        normalized[column] = pd.to_numeric(normalized[column]).round(8).astype(str)
    signature = normalized.astype(str).agg("|".join, axis=1)
    occurrence = signature.groupby(signature).cumcount().astype(str)
    return signature + "|occ=" + occurrence


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
    proxy6 = ranking[ranking.branch.eq("proxy6")].copy().sort_values("proxy_index_1based")
    proxy9 = ranking[ranking.branch.eq("proxy9")].copy().sort_values("proxy_index_1based")
    proxy6["match_key"] = _signature(proxy6)
    proxy9["match_key"] = _signature(proxy9)
    keys6 = set(proxy6.match_key)
    keys9 = set(proxy9.match_key)
    shared = keys6 & keys9
    only6 = keys6 - keys9
    only9 = keys9 - keys6
    index6 = {row.match_key: int(row.proxy_index_1based) - 1 for row in proxy6.itertuples()}
    index9 = {row.match_key: int(row.proxy_index_1based) - 1 for row in proxy9.itertuples()}
    c6 = arrays["proxy6_target_contributions"]
    c9 = arrays["proxy9_target_contributions"]

    shared6 = np.sum(c6[[index6[key] for key in sorted(shared)]], axis=0)
    shared9 = np.sum(c9[[index9[key] for key in sorted(shared)]], axis=0)
    unique6 = np.sum(c6[[index6[key] for key in sorted(only6)]], axis=0) if only6 else np.zeros_like(years, dtype=float)
    unique9 = np.sum(c9[[index9[key] for key in sorted(only9)]], axis=0) if only9 else np.zeros_like(years, dtype=float)
    total6 = np.sum(c6, axis=0)
    total9 = np.sum(c9, axis=0)
    reweighting = shared9 - shared6
    reconstructed_difference = reweighting + unique9 - unique6
    actual_difference = total9 - total6
    closure = float(np.max(np.abs(actual_difference - reconstructed_difference)))

    components = {
        "proxy6_total_update": total6,
        "proxy9_total_update": total9,
        "shared_proxy_update_under_proxy6": shared6,
        "shared_proxy_update_under_proxy9": shared9,
        "shared_proxy_reweighting_proxy9_minus_proxy6": reweighting,
        "proxy9_added_records_direct_term": unique9,
        "proxy6_only_records_term": -unique6,
        "proxy9_minus_proxy6": actual_difference,
    }
    rows = []
    for name, values in components.items():
        rows.append({
            "component": name,
            "trend_c_per_century_1901_1950": _trend(xyears, values[selected]),
            "rolling15_reference_projection_1901_1950": _projection(values[selected], reference, 15),
            "std_c_1901_1950": float(np.std(values[selected])),
        })
    table = pd.DataFrame(rows)
    table.to_csv(output / "proxy_set_difference_components.csv", index=False)

    plot_selected = (years >= 1871) & (years <= 1970)
    figure, axes = plt.subplots(2, 1, figsize=(12.5, 8.5), sharex=True, constrained_layout=True)
    plot_components = {
        "proxy9 - proxy6": actual_difference,
        "shared-proxy reweighting": reweighting,
        "proxy9-added records": unique9,
        "proxy6-only records (subtracted)": -unique6,
    }
    colors = {"proxy9 - proxy6": "#111111", "shared-proxy reweighting": "#d95f02", "proxy9-added records": "#008837", "proxy6-only records (subtracted)": "#7b3294"}
    for name, values in plot_components.items():
        centered = values - np.mean(values[selected])
        axes[0].plot(years[plot_selected], centered[plot_selected], color=colors[name], lw=2.2 if name == "proxy9 - proxy6" else 1.5, label=name)
        axes[1].plot(years[plot_selected], _rolling(centered, 15)[plot_selected], color=colors[name], lw=2.2, label=name)
    for axis in axes:
        axis.axvspan(1901, 1950, color="#d9d9d9", alpha=0.25)
        axis.axhline(0.0, color="#777777", lw=0.7)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, ncol=2)
        axis.set_ylabel("3584 target increment (degC)")
    axes[0].set_title("Proxy-set difference: annual additive components")
    axes[1].set_title("Proxy-set difference: centered 15-year mean")
    axes[1].set_xlabel("Year")
    figure.savefig(output / "target_3584_proxy_set_difference_curves.png", dpi=220)
    plt.close(figure)

    summary = {
        "shared_proxy_count": len(shared),
        "proxy6_only_count": len(only6),
        "proxy9_added_count": len(only9),
        "difference_closure_max_abs_c": closure,
        "components": table.to_dict(orient="records"),
        "interpretation_boundary": "Terms are an exact simultaneous-update algebraic decomposition. Large opposing terms reflect covariance and cannot be interpreted as independent physical causal effects without holdout tests.",
    }
    (output / "proxy_set_difference_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    if closure > 1e-10:
        raise RuntimeError("proxy-set-difference-closure-failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
