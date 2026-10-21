"""Diagnose the 48 Layer-1 systems that reach RGI but not the LIA peak.

This driver is read-only with respect to OGGM.  It merges saved score tables,
uses only numerically clean and converged Layer-1 chains, and selects for each
system the RGI-compatible chain with the smallest LIA peak error.  It then
extracts the saved 1470-2000 length, area, and volume trajectory.

The mechanism labels are descriptive routing aids, not new acceptance gates:

* ``peak_at_start_below_lia``: the chosen curve gains no more than one 50 m
  model node after 1470;
* ``low_start_limited_growth``: the curve starts below the LIA target and
  recovers less than 25% of that initial length deficit;
* ``partial_growth_peak_short``: it recovers at least 25% of the initial
  deficit but still cannot enter the existing 10% LIA total-length gate.

The cold-side diagnostic asks whether the best stable LIA-reaching direction
is bounded by non-converged colder chains, by the tested search edge, or by a
non-monotonic/interior stable response.  It does not authorize new runs.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .build_lia_191_iterative_loop import load_scores


NODE_M = 50.0
RGI_GATE = 0.10
LIA_GATE = 0.10


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_input() -> Path:
    return (
        project_root()
        / "validation_data"
        / "results"
        / "paleo_surface_validation_v0"
        / "lia_191_controller_v1"
        / "diagnostics"
        / "candidate_curve_catalog_v3"
        / "unresolved_system_diagnosis.csv"
    )


def default_output() -> Path:
    return (
        project_root()
        / "validation_data"
        / "results"
        / "paleo_surface_validation_v0"
        / "lia_191_controller_v1"
        / "diagnostics"
        / "rgi_only_layer1_mechanisms_v1"
    )


def _load_forward(output_dir: str) -> pd.DataFrame:
    path = Path(output_dir) / "timeseries.csv"
    frame = pd.read_csv(path)
    required = {"phase", "year", "length_m", "area_m2", "volume_m3"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing-fields:{sorted(missing)}:{path}")
    frame = frame[frame["phase"].astype(str).eq("forward")].copy()
    for column in ("year", "length_m", "area_m2", "volume_m3"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["year", "length_m", "area_m2", "volume_m3"])
    frame["year"] = frame["year"].astype(int)
    return frame.sort_values("year").drop_duplicates("year", keep="last")


def _pick_rgi_anchor(stable: pd.DataFrame) -> pd.Series:
    eligible = stable[stable["e_rgi_total"].le(RGI_GATE)].copy()
    if eligible.empty:
        raise ValueError("no-rgi-compatible-stable-chain")
    return eligible.sort_values(
        ["e_lia_total", "e_rgi_total", "score_priority", "delta_t"],
        na_position="last",
    ).iloc[0]


def _pick_peak_anchor(stable: pd.DataFrame) -> pd.Series:
    return stable.sort_values(
        ["e_lia_total", "e_rgi_total", "score_priority", "delta_t"],
        na_position="last",
    ).iloc[0]


def _endpoint(frame: pd.DataFrame, year: int, column: str) -> float:
    exact = frame.loc[frame["year"].eq(year), column]
    if exact.empty:
        raise ValueError(f"missing-year-{year}")
    return float(exact.iloc[-1])


def _curve_metrics(frame: pd.DataFrame, lia_m: float, rgi_m: float) -> dict:
    start_m = float(frame.iloc[0]["length_m"])
    peak_index = frame["length_m"].idxmax()
    peak = frame.loc[peak_index]
    peak_m = float(peak["length_m"])
    final_m = _endpoint(frame, 2000, "length_m")
    separation_m = lia_m - rgi_m
    initial_deficit_m = lia_m - start_m
    peak_gain_m = peak_m - start_m
    recovery = peak_gain_m / initial_deficit_m if initial_deficit_m > 0 else np.nan
    peak_to_final_m = peak_m - final_m
    peak_area = float(peak["area_m2"])
    peak_volume = float(peak["volume_m3"])
    final_area = _endpoint(frame, 2000, "area_m2")
    final_volume = _endpoint(frame, 2000, "volume_m3")
    return {
        "forward_start_year": int(frame.iloc[0]["year"]),
        "forward_start_m": start_m,
        "forward_peak_year": int(peak["year"]),
        "forward_peak_m": peak_m,
        "forward_2000_m": final_m,
        "initial_lia_deficit_m": initial_deficit_m,
        "peak_lia_deficit_m": lia_m - peak_m,
        "peak_gain_from_start_m": peak_gain_m,
        "initial_deficit_recovery_fraction": recovery,
        "peak_to_2000_amplitude_m": peak_to_final_m,
        "amplitude_ratio_to_target": peak_to_final_m / separation_m,
        "initial_deficit_ratio_to_target": initial_deficit_m / separation_m,
        "peak_deficit_ratio_to_target": (lia_m - peak_m) / separation_m,
        "year_2000_offset_ratio_to_target": (final_m - rgi_m) / separation_m,
        "e_lia_total_from_series": abs(peak_m - lia_m) / lia_m,
        "e_rgi_total_from_series": abs(final_m - rgi_m) / rgi_m,
        "peak_to_2000_area_change_fraction": (final_area - peak_area) / peak_area,
        "peak_to_2000_volume_change_fraction": (final_volume - peak_volume) / peak_volume,
    }


def _trajectory_class(metrics: dict) -> str:
    if metrics["peak_gain_from_start_m"] <= NODE_M:
        return "peak_at_start_below_lia"
    recovery = metrics["initial_deficit_recovery_fraction"]
    if np.isfinite(recovery) and recovery < 0.25:
        return "low_start_limited_growth"
    return "partial_growth_peak_short"


def _cold_side_class(layer1: pd.DataFrame, peak_anchor: pd.Series) -> str:
    stable = layer1[layer1["stable_chain"]].copy()
    stable_min = float(stable["delta_t"].min())
    anchor_dt = float(peak_anchor["delta_t"])
    if not np.isclose(anchor_dt, stable_min):
        return "interior_or_nonmonotonic_stable_peak"
    colder = layer1[layer1["delta_t"].lt(anchor_dt - 1e-9)]
    if colder.empty:
        return "cold_side_search_edge"
    if not colder["stable_chain"].any():
        return "cold_side_stability_boundary"
    return "nonmonotonic_cold_side_response"


def _plot_one(
    row: pd.Series,
    stable_rows: pd.DataFrame,
    rgi_frame: pd.DataFrame,
    peak_frame: pd.DataFrame,
    output_dir: Path,
) -> Path:
    uid = int(row["lia_unique_id"])
    fig, axes = plt.subplots(3, 1, figsize=(10.8, 9.6), sharex=True, constrained_layout=True)
    for score in stable_rows.sort_values("delta_t").itertuples(index=False):
        try:
            frame = _load_forward(str(score.output_dir))
        except Exception:
            continue
        axes[0].plot(frame["year"], frame["length_m"] / 1000.0, color="0.72", lw=1.0, alpha=0.65)

    same_anchor = str(row["rgi_anchor_task_id"]) == str(row["peak_anchor_task_id"])
    axes[0].plot(
        rgi_frame["year"],
        rgi_frame["length_m"] / 1000.0,
        lw=2.4,
        color="#1f77b4",
        label="Best LIA peak while retaining RGI ≤10%",
    )
    if not same_anchor:
        axes[0].plot(
            peak_frame["year"],
            peak_frame["length_m"] / 1000.0,
            lw=2.0,
            color="#ff7f0e",
            label="Closest stable LIA peak (RGI may fail)",
        )
    lia_km = float(row["lia_target_m"]) / 1000.0
    rgi_km = float(row["rgi_target_m"]) / 1000.0
    axes[0].axhline(lia_km, color="#d62728", lw=1.4, label=f"LIA {lia_km:.2f} km")
    axes[0].axhline(rgi_km, color="#2ca02c", lw=1.4, label=f"RGI {rgi_km:.2f} km")
    axes[0].set_ylabel("Length (km)")
    axes[0].legend(frameon=False, fontsize=8, loc="best")

    axes[1].plot(rgi_frame["year"], rgi_frame["area_m2"] / 1e6, color="#1f77b4", lw=2.0)
    axes[1].set_ylabel("Area (km²)")
    axes[2].plot(rgi_frame["year"], rgi_frame["volume_m3"] / 1e9, color="#1f77b4", lw=2.0)
    axes[2].set_ylabel("Volume (km³)")
    axes[2].set_xlabel("Year")
    for axis in axes:
        axis.set_xlim(1470, 2000)
        axis.grid(alpha=0.22)
    fig.suptitle(
        f"LIA system {uid} | {row['trajectory_class']} | {row['cold_side_class']}\n"
        f"RGI anchor ΔT={row['rgi_anchor_delta_t']:+.3f} K; "
        f"peak deficit={row['peak_lia_deficit_m']/1000:.2f} km; "
        f"simulated/target amplitude={row['amplitude_ratio_to_target']:.2f}",
        fontsize=12.5,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"lia_{uid:04d}_rgi_only_diagnosis.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_summary(diagnosis: pd.DataFrame, output_dir: Path) -> Path:
    classes = sorted(diagnosis["trajectory_class"].unique())
    colors = dict(zip(classes, plt.cm.tab10(np.linspace(0, 0.75, len(classes)))))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)

    for name, group in diagnosis.groupby("trajectory_class", sort=True):
        axes[0].scatter(
            group["initial_deficit_ratio_to_target"],
            group["peak_deficit_ratio_to_target"],
            s=42,
            alpha=0.78,
            color=colors[name],
            label=f"{name} (n={len(group)})",
        )
        axes[1].scatter(
            group["amplitude_ratio_to_target"],
            -100.0 * group["peak_to_2000_volume_change_fraction"],
            s=42,
            alpha=0.78,
            color=colors[name],
            label=name,
        )
    axes[0].axhline(0, color="0.25", lw=1.0)
    axes[0].set_xlabel("1470 LIA deficit / target LIA–RGI separation")
    axes[0].set_ylabel("Peak LIA deficit / target LIA–RGI separation")
    axes[0].set_title("State level: how far below LIA the curve begins and peaks")
    axes[1].axvline(1.0, color="0.25", lw=1.0, ls="--")
    axes[1].set_xlabel("Simulated peak-to-2000 amplitude / target separation")
    axes[1].set_ylabel("Peak-to-2000 volume loss (%)")
    axes[1].set_title("Length amplitude versus stored volume response")
    for axis in axes:
        axis.grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    path = output_dir / "rgi_only_mechanism_summary.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_gallery(diagnosis: pd.DataFrame, plot_paths: dict[int, Path], output_dir: Path) -> None:
    cards = []
    for row in diagnosis.sort_values(["trajectory_class", "lia_unique_id"]).itertuples(index=False):
        uid = int(row.lia_unique_id)
        image = html.escape(plot_paths[uid].name)
        cards.append(
            f"<a class='item' href='plots/{image}'><img loading='lazy' src='plots/{image}' alt='LIA {uid} diagnosis'>"
            f"<span>{uid} · {html.escape(str(row.trajectory_class))}</span></a>"
        )
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>RGI-only Layer-1 diagnosis</title>
<style>body{font-family:system-ui,sans-serif;margin:20px;background:#f5f5f5;color:#222}h1{font-size:22px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(285px,1fr));gap:14px}.item{display:block;color:inherit;text-decoration:none;background:#fff;border:1px solid #ddd;padding:8px}.item img{width:100%;height:auto;display:block}.item span{display:block;margin-top:6px;font-weight:600}</style>
</head><body><h1>48 Layer-1 RGI-only systems</h1><div class="grid">__CARDS__</div></body></html>""".replace("__CARDS__", "".join(cards))
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def run(unresolved_path: Path, output_dir: Path) -> dict:
    unresolved = pd.read_csv(unresolved_path)
    selected = unresolved[unresolved["failure_group"].eq("rgi_endpoint_only_after_layer1")].copy()
    if len(selected) != 48:
        raise ValueError(f"expected-48-rgi-only-systems:found-{len(selected)}")
    scores = load_scores()
    rows: list[dict] = []
    series_rows: list[pd.DataFrame] = []
    plot_payloads: dict[int, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    issues: list[dict] = []

    for source in selected.sort_values("lia_unique_id").itertuples(index=False):
        uid = int(source.lia_unique_id)
        layer1 = scores[
            scores["lia_unique_id"].eq(uid) & scores["score_layer"].eq("layer1_delta")
        ].copy()
        stable = layer1[layer1["stable_chain"]].copy()
        try:
            rgi_anchor = _pick_rgi_anchor(stable)
            peak_anchor = _pick_peak_anchor(stable)
            rgi_frame = _load_forward(str(rgi_anchor["output_dir"]))
            peak_frame = _load_forward(str(peak_anchor["output_dir"]))
            metrics = _curve_metrics(rgi_frame, float(source.lia_target_m), float(source.rgi_target_m))
        except Exception as exc:
            issues.append({"lia_unique_id": uid, "issue": f"{type(exc).__name__}:{exc}"})
            continue
        result = {
            "lia_unique_id": uid,
            "rgi_id": str(source.rgi_id),
            "lia_target_m": float(source.lia_target_m),
            "rgi_target_m": float(source.rgi_target_m),
            "lia_rgi_separation_m": float(source.lia_rgi_separation_m),
            "tested_layer1_count": int(len(layer1)),
            "stable_layer1_count": int(len(stable)),
            "rgi_anchor_task_id": str(rgi_anchor["task_id"]),
            "rgi_anchor_source": str(rgi_anchor["score_source"]),
            "rgi_anchor_output_dir": str(rgi_anchor["output_dir"]),
            "rgi_anchor_delta_t": float(rgi_anchor["delta_t"]),
            "peak_anchor_task_id": str(peak_anchor["task_id"]),
            "peak_anchor_source": str(peak_anchor["score_source"]),
            "peak_anchor_output_dir": str(peak_anchor["output_dir"]),
            "peak_anchor_delta_t": float(peak_anchor["delta_t"]),
            "peak_anchor_e_lia_total": float(peak_anchor["e_lia_total"]),
            "peak_anchor_e_rgi_total": float(peak_anchor["e_rgi_total"]),
            "tested_delta_t_min": float(layer1["delta_t"].min()),
            "tested_delta_t_max": float(layer1["delta_t"].max()),
            "stable_delta_t_min": float(stable["delta_t"].min()),
            "stable_delta_t_max": float(stable["delta_t"].max()),
        }
        result.update(metrics)
        result["trajectory_class"] = _trajectory_class(metrics)
        result["cold_side_class"] = _cold_side_class(layer1, peak_anchor)
        rows.append(result)
        rgi_series = rgi_frame.copy()
        rgi_series["lia_unique_id"] = uid
        rgi_series["trajectory_class"] = result["trajectory_class"]
        rgi_series["cold_side_class"] = result["cold_side_class"]
        rgi_series["selected_task_id"] = result["rgi_anchor_task_id"]
        series_rows.append(rgi_series)
        plot_payloads[uid] = (stable, rgi_frame, peak_frame)

    diagnosis = pd.DataFrame(rows).sort_values("lia_unique_id").reset_index(drop=True)
    if len(diagnosis) != 48:
        raise ValueError(f"incomplete-diagnosis:{len(diagnosis)};issues={issues}")
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plot_paths: dict[int, Path] = {}
    for row in diagnosis.itertuples(index=False):
        uid = int(row.lia_unique_id)
        stable, rgi_frame, peak_frame = plot_payloads[uid]
        plot_paths[uid] = _plot_one(pd.Series(row._asdict()), stable, rgi_frame, peak_frame, plots_dir)
    summary_plot = _plot_summary(diagnosis, output_dir)
    _write_gallery(diagnosis, plot_paths, output_dir)

    diagnosis.to_csv(output_dir / "rgi_only_mechanism_diagnosis.csv", index=False)
    pd.concat(series_rows, ignore_index=True).to_csv(
        output_dir / "rgi_anchor_forward_timeseries.csv", index=False
    )
    pd.DataFrame(issues).to_csv(output_dir / "diagnostic_issues.csv", index=False)
    trajectory_counts = diagnosis["trajectory_class"].value_counts().sort_index()
    cold_counts = diagnosis["cold_side_class"].value_counts().sort_index()
    representatives = {
        "largest_peak_deficit": int(diagnosis.loc[diagnosis["peak_deficit_ratio_to_target"].idxmax(), "lia_unique_id"]),
        "smallest_peak_deficit": int(diagnosis.loc[diagnosis["peak_deficit_ratio_to_target"].idxmin(), "lia_unique_id"]),
        "lowest_amplitude_ratio": int(diagnosis.loc[diagnosis["amplitude_ratio_to_target"].idxmin(), "lia_unique_id"]),
        "largest_volume_loss_with_length_deficit": int(diagnosis.loc[diagnosis["peak_to_2000_volume_change_fraction"].idxmin(), "lia_unique_id"]),
    }
    manifest = {
        "launched_oggm": False,
        "system_count": int(len(diagnosis)),
        "plot_count": int(len(plot_paths)),
        "issue_count": int(len(issues)),
        "trajectory_class_counts": {str(k): int(v) for k, v in trajectory_counts.items()},
        "cold_side_class_counts": {str(k): int(v) for k, v in cold_counts.items()},
        "median_initial_deficit_ratio_to_target": float(diagnosis["initial_deficit_ratio_to_target"].median()),
        "median_peak_deficit_ratio_to_target": float(diagnosis["peak_deficit_ratio_to_target"].median()),
        "median_amplitude_ratio_to_target": float(diagnosis["amplitude_ratio_to_target"].median()),
        "median_peak_to_2000_volume_change_fraction": float(diagnosis["peak_to_2000_volume_change_fraction"].median()),
        "representative_systems": representatives,
        "summary_plot": summary_plot.name,
        "gallery": "index.html",
        "interpretation_limit": "descriptive routing diagnosis only; no parameter family or new OGGM batch is authorized",
    }
    (output_dir / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unresolved", type=Path, default=default_input())
    parser.add_argument("--output-dir", type=Path, default=default_output())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run(args.unresolved, args.output_dir)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
