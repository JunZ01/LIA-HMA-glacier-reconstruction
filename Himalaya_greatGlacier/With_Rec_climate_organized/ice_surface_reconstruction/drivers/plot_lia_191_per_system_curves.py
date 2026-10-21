"""Plot one diagnostic figure per LIA system from saved controller chains.

The script is read-only with respect to simulations: it reads completed score
and time-series files, launches no OGGM task, and writes PNG diagnostics plus a
small gallery index.  Each figure contains a full-range historical length
panel, a target-window zoom, and the delta-T endpoint-error response.
"""

from __future__ import annotations

import argparse
import html
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SPINUP_STABILITY_LIMIT = 0.02
ENDPOINT_GATE = 0.20


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


def _bool_value(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _prepare_scores(scores: pd.DataFrame) -> pd.DataFrame:
    frame = scores.copy()
    numeric_columns = [
        "lia_unique_id",
        "delta_t",
        "spinup_stability_fraction",
        "lia_target_m",
        "rgi_target_m",
        "lia_rgi_separation_m",
        "e_lia_sep",
        "e_rgi_sep",
        "objective_sep",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["lia_unique_id"] = frame["lia_unique_id"].astype(int)
    frame["stable_chain"] = (
        ~frame["technical_failure"].map(_bool_value)
        & frame["summary_status"].eq("ok")
        & frame["spinup_converged"].map(_bool_value)
        & frame["spinup_stability_fraction"].le(SPINUP_STABILITY_LIMIT)
    )
    return frame


def _highlight_tasks(group: pd.DataFrame) -> dict[str, set[str]]:
    stable = group[group["stable_chain"]].copy()
    labels: dict[str, set[str]] = {}
    if stable.empty:
        return labels
    selections = {
        "best objective": stable.sort_values(["objective_sep", "delta_t"]).iloc[0],
        "closest LIA": stable.sort_values(["e_lia_sep", "objective_sep"]).iloc[0],
        "closest RGI": stable.sort_values(["e_rgi_sep", "objective_sep"]).iloc[0],
    }
    for label, row in selections.items():
        labels.setdefault(str(row["task_id"]), set()).add(label)
    return labels


def _load_forward(path: str) -> pd.DataFrame:
    frame = pd.read_csv(Path(path) / "timeseries.csv", usecols=["phase", "year", "length_m"])
    frame = frame[frame["phase"].eq("forward")].copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["length_km"] = pd.to_numeric(frame["length_m"], errors="coerce") / 1000.0
    return frame.dropna(subset=["year", "length_km"]).sort_values("year")


def _route_label(route: str) -> str:
    labels = {
        "neither": "neither endpoint reachable",
        "rgi_only": "RGI endpoint only",
        "lia_only": "LIA endpoint only",
        "endpoint_tradeoff_melt_f_delta_t_closed": "endpoint trade-off; melt_f-delta_T closed",
        "no_stable_chain": "no stable chain",
        "small_separation_review": "small endpoint separation review",
        "target_direction_review": "target direction review",
    }
    return labels.get(route, route)


def _plot_one(payload: dict) -> dict:
    uid = int(payload["uid"])
    rows = pd.DataFrame(payload["rows"])
    route = str(payload["route"])
    output = Path(payload["output"])
    output.mkdir(parents=True, exist_ok=True)
    highlights = _highlight_tasks(rows)

    curves: list[tuple[pd.Series, pd.DataFrame]] = []
    errors: list[str] = []
    for _, row in rows.sort_values("delta_t").iterrows():
        try:
            forward = _load_forward(str(row["output_dir"]))
        except Exception as exc:
            errors.append(f"{row['task_id']}:{type(exc).__name__}:{exc}")
            continue
        curves.append((row, forward))

    if not curves:
        return {
            "lia_unique_id": uid,
            "status": "no_curve_file",
            "plot_path": "",
            "curve_count": 0,
            "stable_curve_count": int(rows["stable_chain"].sum()),
            "errors": " | ".join(errors),
        }

    first = rows.iloc[0]
    lia_km = float(first["lia_target_m"]) / 1000.0
    rgi_km = float(first["rgi_target_m"]) / 1000.0
    separation_km = abs(float(first["lia_rgi_separation_m"])) / 1000.0
    tolerance_km = ENDPOINT_GATE * separation_km

    fig = plt.figure(figsize=(15.5, 8.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.65, 1.0], height_ratios=[1.0, 1.0])
    full_ax = fig.add_subplot(grid[:, 0])
    zoom_ax = fig.add_subplot(grid[0, 1])
    response_ax = fig.add_subplot(grid[1, 1])

    highlight_style = {
        "best objective": ("#d62728", 2.8),
        "closest LIA": ("#1f77b4", 2.4),
        "closest RGI": ("#2ca02c", 2.4),
    }
    legend_handles: dict[str, object] = {}

    for row, forward in curves:
        task_id = str(row["task_id"])
        stable = bool(row["stable_chain"])
        named = highlights.get(task_id, set())
        if named:
            primary = min(
                named,
                key=lambda name: ["best objective", "closest LIA", "closest RGI"].index(name),
            )
            color, linewidth = highlight_style[primary]
            alpha = 0.98
            linestyle = "-"
            label = ", ".join(sorted(named)) + f" (ΔT={float(row['delta_t']):+.2f} K)"
        elif stable:
            color, linewidth, alpha, linestyle = "#555555", 1.15, 0.58, "-"
            label = None
        else:
            color, linewidth, alpha, linestyle = "#aaaaaa", 1.0, 0.48, "--"
            label = None
        for axis in (full_ax, zoom_ax):
            line, = axis.plot(
                forward["year"],
                forward["length_km"],
                color=color,
                lw=linewidth,
                alpha=alpha,
                ls=linestyle,
                label=label,
                zorder=4 if named else (3 if stable else 2),
            )
            if label and label not in legend_handles:
                legend_handles[label] = line

    for axis in (full_ax, zoom_ax):
        axis.axhspan(
            lia_km - tolerance_km,
            lia_km + tolerance_km,
            color="#d95f02",
            alpha=0.10,
            zorder=0,
        )
        axis.axhspan(
            rgi_km - tolerance_km,
            rgi_km + tolerance_km,
            color="#1b9e77",
            alpha=0.10,
            zorder=0,
        )
        lia_line = axis.axhline(lia_km, color="#d95f02", lw=1.6, zorder=1)
        rgi_line = axis.axhline(rgi_km, color="#1b9e77", lw=1.6, zorder=1)
        axis.set_xlim(1470, 2000)
        axis.set_xlabel("Year")
        axis.set_ylabel("Length (km)")
        axis.grid(alpha=0.20)
    legend_handles.setdefault(f"LIA target {lia_km:.2f} km", lia_line)
    legend_handles.setdefault(f"RGI target {rgi_km:.2f} km", rgi_line)

    zoom_margin = max(0.35 * separation_km, 0.25)
    zoom_ax.set_ylim(
        max(0.0, min(lia_km, rgi_km) - tolerance_km - zoom_margin),
        max(lia_km, rgi_km) + tolerance_km + zoom_margin,
    )
    full_ax.set_title("All 10 delta-T chains: full length range")
    zoom_ax.set_title("Target-window zoom")

    stable = rows[rows["stable_chain"]].sort_values("delta_t")
    unstable = rows[~rows["stable_chain"]].sort_values("delta_t")
    if not stable.empty:
        response_ax.plot(
            stable["delta_t"],
            stable["e_lia_sep"],
            color="#d95f02",
            marker="o",
            lw=1.8,
            label="LIA error / separation",
        )
        response_ax.plot(
            stable["delta_t"],
            stable["e_rgi_sep"],
            color="#1b9e77",
            marker="s",
            lw=1.8,
            label="RGI error / separation",
        )
        finite_max = np.nanmax(
            np.r_[stable["e_lia_sep"].to_numpy(float), stable["e_rgi_sep"].to_numpy(float)]
        )
        response_top = max(0.35, finite_max * 1.10)
    else:
        response_top = 1.0
        response_ax.text(
            0.5,
            0.55,
            "No stable delta-T chain",
            transform=response_ax.transAxes,
            ha="center",
            va="center",
            fontsize=13,
        )
    if not unstable.empty:
        response_ax.scatter(
            unstable["delta_t"],
            np.full(len(unstable), response_top * 0.96),
            marker="x",
            color="#888888",
            label="non-stable / out-of-range",
            zorder=5,
        )
    response_ax.axhline(ENDPOINT_GATE, color="#222222", ls="--", lw=1.2, label="0.20 gate")
    response_ax.set_ylim(0, response_top)
    response_ax.set_xlabel("Delta T (K)")
    response_ax.set_ylabel("Endpoint error / LIA-RGI separation")
    response_ax.set_title("Delta-T endpoint response")
    response_ax.grid(alpha=0.20)
    response_ax.legend(frameon=False, fontsize=8, loc="best")

    full_ax.legend(
        legend_handles.values(),
        legend_handles.keys(),
        frameon=False,
        fontsize=8,
        loc="best",
    )
    stable_count = int(rows["stable_chain"].sum())
    fig.suptitle(
        f"LIA system {uid} | {_route_label(route)}\n"
        f"stable chains {stable_count}/{len(rows)} | "
        f"LIA {lia_km:.2f} km | RGI {rgi_km:.2f} km | "
        f"20% separation tolerance ±{tolerance_km:.2f} km",
        fontsize=14,
    )

    path = output / f"lia_{uid:04d}_delta_t_curves.png"
    fig.savefig(path, dpi=165, bbox_inches="tight")
    plt.close(fig)

    best = stable.sort_values(["objective_sep", "delta_t"]).iloc[0] if not stable.empty else None
    return {
        "lia_unique_id": uid,
        "status": "ok",
        "route": route,
        "plot_path": str(path),
        "curve_count": len(curves),
        "stable_curve_count": stable_count,
        "best_task_id": "" if best is None else str(best["task_id"]),
        "best_delta_t": np.nan if best is None else float(best["delta_t"]),
        "best_e_lia_sep": np.nan if best is None else float(best["e_lia_sep"]),
        "best_e_rgi_sep": np.nan if best is None else float(best["e_rgi_sep"]),
        "errors": " | ".join(errors),
    }


def _write_gallery(index: pd.DataFrame, output_dir: Path) -> None:
    cards = []
    for row in index[index["status"].eq("ok")].sort_values("lia_unique_id").itertuples():
        image_name = Path(row.plot_path).name
        cards.append(
            "<a class='item' href='plots/{name}'><img loading='lazy' src='plots/{name}' "
            "alt='LIA system {uid} curve diagnostic'><span>{uid}</span></a>".format(
                name=html.escape(image_name), uid=int(row.lia_unique_id)
            )
        )
    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Core-191 per-system curve diagnostics</title>
<style>body{font-family:system-ui,sans-serif;margin:20px;background:#f5f5f5;color:#222}h1{font-size:22px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}.item{display:block;color:inherit;text-decoration:none;background:#fff;border:1px solid #ddd;padding:8px}.item img{display:block;width:100%;height:auto}.item span{display:block;margin-top:6px;font-weight:600}</style>
</head><body><h1>Core-191 saved delta-T curves: one figure per searched system</h1><div class="grid">__CARDS__</div></body></html>
""".replace("__CARDS__", "".join(cards))
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def build_payloads(
    scores: pd.DataFrame,
    routing: pd.DataFrame,
    output_dir: Path,
    ids: list[int] | None,
) -> list[dict]:
    route_by_id = routing.set_index("lia_unique_id")["primary_route"].to_dict()
    selected = scores
    if ids:
        selected = selected[selected["lia_unique_id"].isin(ids)]
    payloads = []
    for uid, group in selected.groupby("lia_unique_id", sort=True):
        payloads.append(
            {
                "uid": int(uid),
                "route": route_by_id.get(int(uid), "unclassified"),
                "rows": group.to_dict(orient="records"),
                "output": str(output_dir / "plots"),
            }
        )
    return payloads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = controller_root()
    parser.add_argument("--scores", type=Path, default=root / "scores.csv")
    parser.add_argument(
        "--routing",
        type=Path,
        default=root
        / "diagnostics"
        / "core191_routing_v1"
        / "core191_routing_table.csv",
    )
    parser.add_argument(
        "--master-status",
        type=Path,
        default=root / "lia_191_master_status.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "diagnostics" / "per_system_curves_v1",
    )
    parser.add_argument("--ids", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scores = _prepare_scores(pd.read_csv(args.scores))
    routing = pd.read_csv(args.routing)
    routing["lia_unique_id"] = pd.to_numeric(routing["lia_unique_id"]).astype(int)
    master = pd.read_csv(args.master_status)
    master["lia_unique_id"] = pd.to_numeric(master["lia_unique_id"]).astype(int)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payloads = build_payloads(scores, routing, args.output_dir, args.ids)
    results: list[dict] = []
    if args.workers <= 1:
        for payload in payloads:
            results.append(_plot_one(payload))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_plot_one, payload): payload["uid"] for payload in payloads}
            for number, future in enumerate(as_completed(futures), start=1):
                uid = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(
                        {
                            "lia_unique_id": uid,
                            "status": "plot_failure",
                            "plot_path": "",
                            "curve_count": 0,
                            "stable_curve_count": 0,
                            "errors": f"{type(exc).__name__}:{exc}",
                        }
                    )
                if number % 10 == 0 or number == len(futures):
                    print(f"completed {number}/{len(futures)}", flush=True)

    index = pd.DataFrame(results).sort_values("lia_unique_id").reset_index(drop=True)
    index.to_csv(args.output_dir / "plot_index.csv", index=False)
    _write_gallery(index, args.output_dir)

    plotted_ids = set(index.loc[index["status"].eq("ok"), "lia_unique_id"].astype(int))
    unplotted = master[~master["lia_unique_id"].isin(plotted_ids)].copy()
    keep = [
        "lia_unique_id",
        "rgi_id",
        "workflow_state",
        "existing_decision",
        "existing_decision_evidence",
        "preflight_state",
        "preflight_reasons",
    ]
    unplotted[[column for column in keep if column in unplotted.columns]].to_csv(
        args.output_dir / "unplotted_systems.csv", index=False
    )
    summary = {
        "launched_oggm": False,
        "requested_system_count": len(payloads),
        "plotted_system_count": int(index["status"].eq("ok").sum()),
        "plot_failure_count": int(index["status"].ne("ok").sum()),
        "full_core_system_count": int(len(master)),
        "systems_without_controller_delta_t_curves": sorted(
            set(master["lia_unique_id"].astype(int)) - set(scores["lia_unique_id"].astype(int))
        ),
        "workers": args.workers,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if not index["status"].ne("ok").any() else 1


if __name__ == "__main__":
    raise SystemExit(main())

