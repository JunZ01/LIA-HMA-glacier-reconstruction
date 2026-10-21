"""Consolidate and plot the three-round delta-T bisection experiment.

The script launches no OGGM run. It combines the original boundary pilot,
the system-520 preflight, and three adaptive bisection rounds. One figure per
LIA system shows all forward curves and the delta-T response of the forward
peak and year-2000 endpoint on the same length scale.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


TOLERANCE = 0.20
STABILITY_LIMIT = 0.02


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _inputs(root: Path) -> list[tuple[str, Path]]:
    base = root / "validation_data" / "results" / "paleo_surface_validation_v0"
    return [
        ("boundary", base / "lia_191_curve_loop_delta_boundary_pilot_v1" / "scores.csv"),
        ("preflight", base / "lia_191_curve_loop_delta_boundary_preflight_v1" / "scores.csv"),
        ("round1", base / "lia_191_curve_loop_delta_bisection_r1_v1" / "scores.csv"),
        ("round2", base / "lia_191_curve_loop_delta_bisection_r2_v1" / "scores.csv"),
        ("round3", base / "lia_191_curve_loop_delta_bisection_r3_v1" / "scores.csv"),
    ]


def load_scores(root: Path) -> pd.DataFrame:
    frames = []
    for source, path in _inputs(root):
        frame = pd.read_csv(path)
        frame["bisection_source"] = source
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    numeric = [
        "lia_unique_id", "delta_t", "spinup_stability_fraction",
        "lia_target_m", "rgi_target_m", "lia_rgi_separation_m",
        "forward_peak_m", "forward_2000_m", "e_lia_sep", "e_rgi_sep",
        "objective_sep", "amplitude_ratio_to_target",
        "max_corridor_exceedance_m",
    ]
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["lia_unique_id"] = out["lia_unique_id"].astype(int)
    out["technical_failure_bool"] = out["technical_failure"].map(_bool)
    out["spinup_converged_bool"] = out["spinup_converged"].map(_bool)
    out["stable_chain"] = (
        ~out["technical_failure_bool"]
        & out["summary_status"].eq("ok")
        & out["spinup_converged_bool"]
        & out["spinup_stability_fraction"].le(STABILITY_LIMIT)
    )
    out["joint_endpoint_pass"] = (
        out["stable_chain"]
        & out["e_lia_sep"].le(TOLERANCE)
        & out["e_rgi_sep"].le(TOLERANCE)
    )
    return out.sort_values(["lia_unique_id", "delta_t", "bisection_source"])


def load_forward(output_dir: str) -> pd.DataFrame:
    frame = pd.read_csv(Path(output_dir) / "timeseries.csv")
    frame = frame[frame["phase"].eq("forward")].copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["length_km"] = pd.to_numeric(frame["length_m"], errors="coerce") / 1000.0
    return frame.dropna(subset=["year", "length_km"]).sort_values("year")


def plot_system(group: pd.DataFrame, output_dir: Path) -> Path:
    uid = int(group.iloc[0]["lia_unique_id"])
    lia = float(group.iloc[0]["lia_target_m"]) / 1000.0
    rgi = float(group.iloc[0]["rgi_target_m"]) / 1000.0
    separation = abs(lia - rgi)
    tolerance = TOLERANCE * separation
    stable = group[group["stable_chain"]].copy()
    best = stable.sort_values(["objective_sep", "delta_t"]).iloc[0]

    dmin = float(group["delta_t"].min())
    dmax = float(group["delta_t"].max())
    norm = Normalize(vmin=dmin, vmax=dmax)
    cmap = plt.get_cmap("viridis")
    fig, (curve_ax, response_ax) = plt.subplots(
        2, 1, figsize=(11.8, 9.2), constrained_layout=True,
        gridspec_kw={"height_ratios": [1.55, 1.0]},
    )

    for _, row in group.sort_values("delta_t").iterrows():
        forward = load_forward(str(row["output_dir"]))
        color = cmap(norm(float(row["delta_t"])))
        curve_ax.plot(
            forward["year"], forward["length_km"], color=color,
            lw=2.1 if bool(row["stable_chain"]) else 1.5,
            ls="-" if bool(row["stable_chain"]) else "--",
            alpha=0.95 if bool(row["stable_chain"]) else 0.65,
        )

    corridor_low = min(lia, rgi) - tolerance
    corridor_high = max(lia, rgi) + tolerance
    curve_ax.axhspan(corridor_low, corridor_high, color="0.6", alpha=0.13)
    curve_ax.axhline(lia, color="#d95f02", lw=1.8)
    curve_ax.axhline(rgi, color="#1b9e77", lw=1.8)
    curve_ax.set_xlim(1470, 2000)
    curve_ax.set_xlabel("Year")
    curve_ax.set_ylabel("Length (km)")
    curve_ax.set_title("All original and bisection forward curves")
    curve_ax.grid(alpha=0.22)

    stable = stable.sort_values("delta_t")
    response_ax.plot(
        stable["delta_t"], stable["forward_peak_m"] / 1000.0,
        color="#d95f02", marker="o", lw=1.8, label="Forward peak",
    )
    response_ax.plot(
        stable["delta_t"], stable["forward_2000_m"] / 1000.0,
        color="#1b9e77", marker="s", lw=1.8, label="Year 2000",
    )
    unstable = group[~group["stable_chain"]]
    if not unstable.empty:
        response_ax.scatter(
            unstable["delta_t"], unstable["forward_peak_m"] / 1000.0,
            marker="x", s=65, color="0.35", label="Not stable/out of range",
            zorder=5,
        )
    response_ax.axhspan(lia - tolerance, lia + tolerance, color="#d95f02", alpha=0.08)
    response_ax.axhspan(rgi - tolerance, rgi + tolerance, color="#1b9e77", alpha=0.08)
    response_ax.axhline(lia, color="#d95f02", lw=1.25, ls="--")
    response_ax.axhline(rgi, color="#1b9e77", lw=1.25, ls="--")
    response_ax.set_xlabel("Delta T (K)")
    response_ax.set_ylabel("Length (km)")
    response_ax.set_title("Endpoint response to delta T")
    response_ax.grid(alpha=0.22)
    response_ax.legend(frameon=False, fontsize=8, loc="best")

    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar, ax=curve_ax, pad=0.01, aspect=28)
    colorbar.set_label("Delta T (K)")
    curve_ax.legend(
        handles=[
            Line2D([0], [0], color="0.25", lw=2.1, ls="-", label="Stable spin-up"),
            Line2D([0], [0], color="0.45", lw=1.5, ls="--", label="Not stable/out of range"),
            Line2D([0], [0], color="#d95f02", lw=1.8, label=f"LIA {lia:.3f} km"),
            Line2D([0], [0], color="#1b9e77", lw=1.8, label=f"RGI {rgi:.3f} km"),
        ],
        frameon=False, fontsize=8, loc="best",
    )
    fig.suptitle(
        f"LIA system {uid} | {len(group)} delta-T chains | "
        f"joint passes {int(group['joint_endpoint_pass'].sum())}\n"
        f"best dT={float(best['delta_t']):+.6f} K, "
        f"E_LIA={float(best['e_lia_sep']):.3f}, "
        f"E_RGI={float(best['e_rgi_sep']):.3f}, "
        f"amplitude={float(best['amplitude_ratio_to_target']):.1%} of target",
        fontsize=13,
    )
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"system_{uid:04d}_delta_bisection.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def summarize(scores: pd.DataFrame, plot_paths: dict[int, Path]) -> pd.DataFrame:
    rows = []
    for uid, group in scores.groupby("lia_unique_id", sort=True):
        stable = group[group["stable_chain"]].copy()
        best = stable.sort_values(["objective_sep", "delta_t"]).iloc[0]
        closest_lia = stable.sort_values(["e_lia_sep", "e_rgi_sep"]).iloc[0]
        closest_rgi = stable.sort_values(["e_rgi_sep", "e_lia_sep"]).iloc[0]
        rows.append(
            {
                "lia_unique_id": int(uid),
                "chain_count": int(len(group)),
                "stable_chain_count": int(group["stable_chain"].sum()),
                "technical_failure_count": int(group["technical_failure_bool"].sum()),
                "joint_endpoint_pass_count": int(group["joint_endpoint_pass"].sum()),
                "best_delta_t": float(best["delta_t"]),
                "best_e_lia_sep": float(best["e_lia_sep"]),
                "best_e_rgi_sep": float(best["e_rgi_sep"]),
                "best_amplitude_ratio": float(best["amplitude_ratio_to_target"]),
                "closest_lia_delta_t": float(closest_lia["delta_t"]),
                "closest_lia_e_lia_sep": float(closest_lia["e_lia_sep"]),
                "closest_rgi_delta_t": float(closest_rgi["delta_t"]),
                "closest_rgi_e_rgi_sep": float(closest_rgi["e_rgi_sep"]),
                "plot_path": str(plot_paths[int(uid)]),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    root = project_root()
    output = (
        root / "validation_data" / "results" / "paleo_surface_validation_v0"
        / "lia_191_curve_loop_delta_bisection_r3_v1" / "diagnostics"
        / "consolidated_v1"
    )
    output.mkdir(parents=True, exist_ok=True)
    scores = load_scores(root)
    scores.to_csv(output / "combined_21_chain_diagnostics.csv", index=False)
    paths = {
        int(uid): plot_system(group, output)
        for uid, group in scores.groupby("lia_unique_id", sort=True)
    }
    summary = summarize(scores, paths)
    summary.to_csv(output / "system_summary.csv", index=False)
    record = {
        "launched_oggm": False,
        "chain_count": int(len(scores)),
        "new_bisection_chain_count": int(scores["bisection_source"].str.startswith("round").sum()),
        "technical_failure_count": int(scores["technical_failure_bool"].sum()),
        "stable_chain_count": int(scores["stable_chain"].sum()),
        "joint_endpoint_pass_count": int(scores["joint_endpoint_pass"].sum()),
        "rounds_completed": 3,
        "routing_tolerance": TOLERANCE,
        "system_summary": summary.to_dict(orient="records"),
    }
    (output / "diagnosis.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
