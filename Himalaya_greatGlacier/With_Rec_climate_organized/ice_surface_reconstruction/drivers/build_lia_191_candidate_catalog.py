"""Build reusable curve and diagnosis products from saved LIA-191 chains.

This is a read-only post-processing driver.  It does not initialize OGGM and
does not start a glacier simulation.  It uses the current three-layer router
to locate the 51 systems routed to ``freeze_provisional_candidate``, extracts
their saved 1470-2000 forward length/area/volume series, writes one plot per
system, and classifies the remaining systems from their declared routing
state and best saved stable chain.

The manual 1044 B-window candidate is explicit because it predates the
controller score archive.  All other candidate selections use the router's
stable direct-candidate rule and the same deterministic row ordering.
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


TIME_SERIES_FIELDS = {"phase", "year", "length_m", "area_m2", "volume_m3"}
DIRECT_ROUTE = "freeze_provisional_candidate"
MANUAL_CANDIDATES = {
    1044: {
        "source": "manual_1044_B_window_candidate",
        "task_id": "mf_4p666_dt_p1p192",
        "output_dir": (
            "validation_data/results/paleo_surface_validation_v0/"
            "layered_coordinate_search_v1/"
            "layer_B_melt_delta_boundary_refine_1044_round2_v1/"
            "system_1044/mf_4p666_dt_p1p192"
        ),
    }
}
SCORE_MATCH_TOLERANCE_M = 50.0


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def controller_root() -> Path:
    return (
        project_root()
        / "validation_data"
        / "results"
        / "paleo_surface_validation_v0"
        / "lia_191_controller_v1"
    )


def default_routing_path() -> Path:
    return (
        controller_root()
        / "diagnostics"
        / "iterative_loop_v5_after_layer3_close"
        / "core191_iterative_loop_table.csv"
    )


def default_output_dir() -> Path:
    return controller_root() / "diagnostics" / "candidate_curve_catalog_v1"


def _best_direct_row(rows: pd.DataFrame) -> pd.Series:
    ranked = rows.sort_values(
        ["e_lia_total", "e_rgi_total", "score_priority", "objective_sep"],
        na_position="last",
    )
    return ranked.iloc[0]


def _best_stable_row(rows: pd.DataFrame) -> pd.Series | None:
    stable = rows[rows["stable_chain"]].copy()
    if stable.empty:
        return None
    stable["objective_total"] = stable["e_lia_total"] + stable["e_rgi_total"]
    return stable.sort_values(
        ["objective_total", "score_priority", "objective_sep"],
        na_position="last",
    ).iloc[0]


def _load_forward(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(TIME_SERIES_FIELDS.difference(frame.columns))
    if missing:
        raise ValueError(f"missing-timeseries-fields:{','.join(missing)}")
    forward = frame[frame["phase"].astype(str).eq("forward")].copy()
    if forward.empty:
        raise ValueError("no-forward-rows")
    for column in ("year", "length_m", "area_m2", "volume_m3"):
        forward[column] = pd.to_numeric(forward[column], errors="coerce")
    forward = forward.dropna(subset=["year", "length_m", "area_m2", "volume_m3"])
    forward["year"] = forward["year"].astype(int)
    return forward.sort_values("year").drop_duplicates("year", keep="last")


def _candidate_rows(routing: pd.DataFrame, scores: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    rows: list[dict] = []
    issues: list[dict] = []
    for route_row in routing[routing["next_route"].eq(DIRECT_ROUTE)].itertuples(index=False):
        uid = int(route_row.lia_unique_id)
        result = {
            "lia_unique_id": uid,
            "rgi_id": str(route_row.rgi_id),
            "lia_target_m": float(route_row.lia_length_m),
            "rgi_target_m": float(route_row.rgi_length_m),
            "lia_rgi_separation_m": float(route_row.lia_rgi_separation_m),
            "route": str(route_row.next_route),
            "route_reason": str(route_row.route_reason),
        }
        if uid in MANUAL_CANDIDATES:
            manual = MANUAL_CANDIDATES[uid]
            result.update(manual)
            result["selection_kind"] = "manual_explicit_candidate"
            result["score_fingerprint"] = ""
            result["delta_t"] = np.nan
            result["melt_f"] = np.nan
            result["prcp_fac"] = np.nan
            result["glen_a"] = np.nan
            result["temp_melt"] = np.nan
            result["e_lia_total_score"] = np.nan
            result["e_rgi_total_score"] = np.nan
        else:
            direct = scores[
                scores["lia_unique_id"].eq(uid) & scores["direct_candidate"]
            ].copy()
            if direct.empty:
                issues.append(
                    {
                        "lia_unique_id": uid,
                        "issue": "route-freeze-without-direct-score-or-manual-map",
                    }
                )
                continue
            row = _best_direct_row(direct)
            result.update(
                {
                    "source": str(row["score_source"]),
                    "task_id": str(row["task_id"]),
                    "output_dir": str(row["output_dir"]),
                    "selection_kind": "router_best_direct_candidate",
                    "score_fingerprint": str(row["fingerprint"]),
                    "delta_t": float(row["delta_t"]),
                    "melt_f": float(row["melt_f"]),
                    "prcp_fac": float(row["prcp_fac"]),
                    "glen_a": float(row["glen_a"]),
                    "temp_melt": float(row["temp_melt"]),
                    "e_lia_total_score": float(row["e_lia_total"]),
                    "e_rgi_total_score": float(row["e_rgi_total"]),
                }
            )
        result["output_dir"] = str(project_root() / result["output_dir"])
        rows.append(result)
    return pd.DataFrame(rows).sort_values("lia_unique_id").reset_index(drop=True), issues


def _append_curve_metrics(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    candidate_rows: list[dict] = []
    long_rows: list[pd.DataFrame] = []
    issues: list[dict] = []
    for candidate in candidates.to_dict(orient="records"):
        uid = int(candidate["lia_unique_id"])
        source_path = Path(candidate["output_dir"]) / "timeseries.csv"
        try:
            forward = _load_forward(source_path)
        except Exception as exc:
            issues.append({"lia_unique_id": uid, "issue": f"timeseries:{type(exc).__name__}:{exc}", "path": str(source_path)})
            continue
        peak_m = float(forward["length_m"].max())
        final_m = float(forward.loc[forward["year"].eq(2000), "length_m"].iloc[-1]) if (forward["year"] == 2000).any() else np.nan
        if not np.isfinite(final_m):
            issues.append({"lia_unique_id": uid, "issue": "missing-forward-year-2000", "path": str(source_path)})
            continue
        candidate["timeseries_path"] = str(source_path)
        candidate["forward_first_year"] = int(forward["year"].min())
        candidate["forward_last_year"] = int(forward["year"].max())
        candidate["forward_peak_m"] = peak_m
        candidate["forward_2000_m"] = final_m
        candidate["e_lia_total_from_series"] = abs(peak_m - candidate["lia_target_m"]) / candidate["lia_target_m"]
        candidate["e_rgi_total_from_series"] = abs(final_m - candidate["rgi_target_m"]) / candidate["rgi_target_m"]
        candidate["peak_year"] = int(forward.loc[forward["length_m"].idxmax(), "year"])
        candidate["peak_to_2000_amplitude_m"] = peak_m - final_m
        if np.isfinite(candidate["e_lia_total_score"]):
            peak_score_delta = abs(candidate["e_lia_total_from_series"] - candidate["e_lia_total_score"]) * candidate["lia_target_m"]
            final_score_delta = abs(candidate["e_rgi_total_from_series"] - candidate["e_rgi_total_score"]) * candidate["rgi_target_m"]
            candidate["score_match_peak_delta_m"] = peak_score_delta
            candidate["score_match_2000_delta_m"] = final_score_delta
            if max(peak_score_delta, final_score_delta) > SCORE_MATCH_TOLERANCE_M:
                issues.append(
                    {
                        "lia_unique_id": uid,
                        "issue": "saved-series-does-not-match-router-score",
                        "peak_difference_m": peak_score_delta,
                        "year_2000_difference_m": final_score_delta,
                        "path": str(source_path),
                    }
                )
        else:
            candidate["score_match_peak_delta_m"] = np.nan
            candidate["score_match_2000_delta_m"] = np.nan
        for key, value in candidate.items():
            forward[key] = value
        long_rows.append(forward)
        candidate_rows.append(candidate)
    metrics = pd.DataFrame(candidate_rows).sort_values("lia_unique_id").reset_index(drop=True)
    long = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()
    return metrics, long, issues


def _plot_candidate(candidate: pd.Series, series: pd.DataFrame, output: Path) -> Path:
    uid = int(candidate["lia_unique_id"])
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.5), sharex=True, constrained_layout=True)
    panels = [
        ("length_m", 1000.0, "Length (km)"),
        ("area_m2", 1.0e6, "Area (km²)"),
        ("volume_m3", 1.0e9, "Volume (km³)"),
    ]
    for axis, (column, scale, ylabel) in zip(axes, panels):
        values = series[column] / scale
        axis.plot(series["year"], values, color="#1f77b4", lw=2.0)
        axis.set_ylabel(ylabel)
        axis.set_xlim(1470, 2000)
        axis.grid(alpha=0.22)
    length_axis = axes[0]
    lia = float(candidate["lia_target_m"]) / 1000.0
    rgi = float(candidate["rgi_target_m"]) / 1000.0
    length_axis.axhline(lia, color="#d95f02", lw=1.4, label=f"LIA target {lia:.2f} km")
    length_axis.axhline(rgi, color="#1b9e77", lw=1.4, label=f"RGI target {rgi:.2f} km")
    length_axis.legend(frameon=False, loc="best", fontsize=9)
    axes[-1].set_xlabel("Year")
    title_delta = "" if not np.isfinite(candidate["delta_t"]) else f" | ΔT={candidate['delta_t']:+.3f} K"
    fig.suptitle(
        f"LIA system {uid} | saved provisional candidate{title_delta}\n"
        f"peak error={candidate['e_lia_total_from_series']:.1%}; 2000 error={candidate['e_rgi_total_from_series']:.1%}; "
        f"peak year={int(candidate['peak_year'])}",
        fontsize=13,
    )
    path = output / f"lia_{uid:04d}_forward_metrics.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_candidate_gallery(metrics: pd.DataFrame, plots: dict[int, Path], output: Path) -> None:
    cards = []
    for row in metrics.sort_values("lia_unique_id").itertuples(index=False):
        uid = int(row.lia_unique_id)
        if uid not in plots:
            continue
        image_name = html.escape(plots[uid].name)
        cards.append(
            f"<a class='item' href='plots/{image_name}'><img loading='lazy' src='plots/{image_name}' alt='LIA {uid} forward metrics'><span>LIA {uid}</span></a>"
        )
    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LIA-191 provisional candidate curves</title>
<style>body{font-family:system-ui,sans-serif;margin:20px;background:#f5f5f5;color:#222}h1{font-size:22px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}.item{display:block;color:inherit;text-decoration:none;background:#fff;border:1px solid #ddd;padding:8px}.item img{display:block;width:100%;height:auto}.item span{display:block;margin-top:6px;font-weight:600}</style>
</head><body><h1>Saved 1470-2000 provisional candidate curves</h1><div class="grid">__CARDS__</div></body></html>
""".replace("__CARDS__", "".join(cards))
    (output / "index.html").write_text(document, encoding="utf-8")


def _failure_group(route_row: pd.Series, score_rows: pd.DataFrame) -> tuple[str, str]:
    route = str(route_row["next_route"])
    if route == "hold_after_layer1":
        stable = score_rows[score_rows["stable_chain"]]
        if stable.empty:
            return "no_stable_saved_chain", "no stable numerical chain in merged score archive"
        lia = bool(score_rows["lia_reachable_total"].any())
        rgi = bool(score_rows["rgi_reachable_total"].any())
        if lia and rgi:
            return "separate_endpoint_branches_after_layer1", "LIA and RGI are each reachable, but not by the same stable chain"
        if lia:
            return "lia_endpoint_only_after_layer1", "a stable LIA-peak match exists but no stable RGI endpoint match"
        if rgi:
            return "rgi_endpoint_only_after_layer1", "a stable RGI endpoint match exists but no stable LIA-peak match"
        return "neither_endpoint_after_layer1", "stable chains exist but neither endpoint enters the 10% total-length rule"
    if route == "layer3_closed":
        lia = bool(score_rows["lia_reachable_total"].any())
        rgi = bool(score_rows["rgi_reachable_total"].any())
        if lia and rgi:
            return "layer3_closed_separate_endpoint_branches", "the declared Layer-3 block did not produce a joint stable endpoint match"
        if lia:
            return "layer3_closed_lia_only", "the declared Layer-3 block retained a LIA-peak-only match"
        if rgi:
            return "layer3_closed_rgi_only", "the declared Layer-3 block retained a RGI-only match"
        return "layer3_closed_neither_endpoint", "the declared Layer-3 block did not produce either endpoint match"
    labels = {
        "small_separation_review": ("small_endpoint_separation", "LIA-RGI separation is below the declared 500 m routing threshold"),
        "target_direction_review": ("target_direction_review", "LIA and RGI targets do not currently define a retreat target"),
        "blocked_dynamic_review": ("blocked_zero_mb_dynamic_review", "explicit zero-MB dynamic review gate remains unresolved"),
        "named_configuration_closed": ("named_configuration_closed", "bounded parameter blocks are closed for the named configuration"),
    }
    return labels.get(route, (route, str(route_row["route_reason"])))


def _build_unresolved(routing: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for route_row in routing[~routing["next_route"].eq(DIRECT_ROUTE)].itertuples(index=False):
        row = pd.Series(route_row._asdict())
        uid = int(row["lia_unique_id"])
        local = scores[scores["lia_unique_id"].eq(uid)].copy()
        best = _best_stable_row(local)
        group, mechanism = _failure_group(row, local)
        result = {
            "lia_unique_id": uid,
            "rgi_id": str(row["rgi_id"]),
            "route": str(row["next_route"]),
            "failure_group": group,
            "mechanism_record": mechanism,
            "lia_target_m": float(row["lia_length_m"]),
            "rgi_target_m": float(row["rgi_length_m"]),
            "lia_rgi_separation_m": float(row["lia_rgi_separation_m"]),
            "tested_chain_count": int(len(local)),
            "stable_chain_count": int(local["stable_chain"].sum()),
        }
        if best is None:
            result.update({
                "best_stable_source": "",
                "best_stable_task_id": "",
                "best_stable_delta_t": np.nan,
                "best_stable_e_lia_total": np.nan,
                "best_stable_e_rgi_total": np.nan,
                "best_stable_objective_total": np.nan,
            })
        else:
            result.update({
                "best_stable_source": str(best["score_source"]),
                "best_stable_task_id": str(best["task_id"]),
                "best_stable_delta_t": float(best["delta_t"]),
                "best_stable_e_lia_total": float(best["e_lia_total"]),
                "best_stable_e_rgi_total": float(best["e_rgi_total"]),
                "best_stable_objective_total": float(best["e_lia_total"] + best["e_rgi_total"]),
            })
        rows.append(result)
    return pd.DataFrame(rows).sort_values(["route", "lia_unique_id"]).reset_index(drop=True)


def _plot_unresolved(unresolved: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    counts = unresolved["failure_group"].value_counts().sort_values(ascending=True)
    fig, axis = plt.subplots(figsize=(10.5, 7.2), constrained_layout=True)
    axis.barh(counts.index, counts.values, color="#4c78a8")
    axis.set_xlabel("System count")
    axis.set_title("Unresolved systems: current bounded-loop diagnosis")
    axis.grid(axis="x", alpha=0.25)
    for index, value in enumerate(counts.values):
        axis.text(value + max(0.3, counts.max() * 0.012), index, str(int(value)), va="center")
    count_path = output / "unresolved_failure_group_counts.png"
    fig.savefig(count_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    finite = unresolved.dropna(subset=["best_stable_e_lia_total", "best_stable_e_rgi_total"]).copy()
    fig, axis = plt.subplots(figsize=(8.5, 7.0), constrained_layout=True)
    for route, group in finite.groupby("route", sort=True):
        axis.scatter(
            group["best_stable_e_lia_total"],
            group["best_stable_e_rgi_total"],
            s=34,
            alpha=0.78,
            label=route,
        )
    axis.axvline(0.10, color="#d95f02", lw=1.2, ls="--")
    axis.axhline(0.10, color="#1b9e77", lw=1.2, ls="--")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Best stable LIA peak error / LIA length")
    axis.set_ylabel("Best stable year-2000 error / RGI length")
    axis.set_title("Unresolved systems: best saved stable endpoint errors")
    axis.grid(alpha=0.25, which="both")
    axis.legend(frameon=False, fontsize=8, loc="best")
    scatter_path = output / "unresolved_best_stable_endpoint_errors.png"
    fig.savefig(scatter_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return count_path, scatter_path


def build_catalog(routing_path: Path, output_dir: Path) -> dict:
    routing = pd.read_csv(routing_path)
    routing["lia_unique_id"] = pd.to_numeric(routing["lia_unique_id"], errors="raise").astype(int)
    if len(routing) != 191 or routing["lia_unique_id"].nunique() != 191:
        raise ValueError("routing-is-not-a-unique-191-system-table")
    scores = load_scores()
    candidates, issues = _candidate_rows(routing, scores)
    metrics, forward, metric_issues = _append_curve_metrics(candidates)
    issues.extend(metric_issues)
    if len(metrics) != len(candidates):
        missing_ids = sorted(set(candidates["lia_unique_id"]).difference(metrics["lia_unique_id"]))
        issues.append({"issue": "candidate-timeseries-not-complete", "lia_unique_ids": missing_ids})
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: dict[int, Path] = {}
    for row in metrics.itertuples(index=False):
        subset = forward[forward["lia_unique_id"].eq(int(row.lia_unique_id))].copy()
        plot_paths[int(row.lia_unique_id)] = _plot_candidate(pd.Series(row._asdict()), subset, plots_dir)
    _write_candidate_gallery(metrics, plot_paths, output_dir)
    unresolved = _build_unresolved(routing, scores)
    unresolved_counts = (
        unresolved.groupby(["route", "failure_group"], as_index=False)
        .size()
        .rename(columns={"size": "system_count"})
        .sort_values(["route", "failure_group"])
    )
    count_path, scatter_path = _plot_unresolved(unresolved, output_dir)
    candidates.to_csv(output_dir / "candidate_selection_raw.csv", index=False)
    metrics.to_csv(output_dir / "candidate_selection_verified.csv", index=False)
    forward.to_csv(output_dir / "candidate_forward_timeseries_1470_2000.csv", index=False)
    unresolved.to_csv(output_dir / "unresolved_system_diagnosis.csv", index=False)
    unresolved_counts.to_csv(output_dir / "unresolved_failure_group_counts.csv", index=False)
    pd.DataFrame(issues).to_csv(output_dir / "catalog_issues.csv", index=False)
    manifest = {
        "launched_oggm": False,
        "routing_path": str(routing_path),
        "merged_score_row_count": int(len(scores)),
        "candidate_route_count": int(routing["next_route"].eq(DIRECT_ROUTE).sum()),
        "candidate_timeseries_count": int(len(metrics)),
        "candidate_plot_count": int(len(plot_paths)),
        "unresolved_system_count": int(len(unresolved)),
        "issue_count": int(len(issues)),
        "direct_rule": "stable chain with forward peak within 10% of LIA total length and year-2000 within 10% of RGI total length",
        "manual_candidate": "1044 mf_4p666_dt_p1p192 retained by explicit project decision",
        "plots": {
            "candidate_gallery": "index.html",
            "unresolved_group_counts": count_path.name,
            "unresolved_endpoint_scatter": scatter_path.name,
        },
    }
    (output_dir / "catalog_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routing", type=Path, default=default_routing_path())
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_catalog(args.routing, args.output_dir)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
