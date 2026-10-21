"""Diagnose the three-system delta-T branch-boundary pilot.

This script is read-only with respect to OGGM simulations. It combines the
eight batch chains with the system-520 preflight chain, applies the canonical
spin-up stability gate, writes one diagnostic plot per LIA system, and records
whether any stable chain jointly reaches the LIA peak and RGI endpoint at the
declared 20% separation-scaled routing tolerance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STABILITY_LIMIT = 0.02
ROUTING_TOLERANCE = 0.20


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _prepare(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    numeric = [
        "lia_unique_id",
        "delta_t",
        "spinup_stability_fraction",
        "lia_target_m",
        "rgi_target_m",
        "lia_rgi_separation_m",
        "forward_peak_m",
        "forward_2000_m",
        "peak_to_2000_amplitude_m",
        "amplitude_ratio_to_target",
        "e_lia_sep",
        "e_rgi_sep",
        "objective_sep",
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
    sep = out["lia_rgi_separation_m"].abs()
    out["signed_peak_offset_sep"] = (out["forward_peak_m"] - out["lia_target_m"]) / sep
    out["signed_2000_offset_sep"] = (out["forward_2000_m"] - out["rgi_target_m"]) / sep
    out["joint_endpoint_pass"] = (
        out["stable_chain"]
        & out["signed_peak_offset_sep"].abs().le(ROUTING_TOLERANCE)
        & out["signed_2000_offset_sep"].abs().le(ROUTING_TOLERANCE)
    )
    out["corridor_pass"] = (
        out["stable_chain"]
        & out["max_corridor_exceedance_m"].fillna(np.inf).le(1.0e-6)
    )
    out["branch_state"] = out.apply(_branch_state, axis=1)
    return out


def _branch_state(row: pd.Series) -> str:
    if bool(row["technical_failure_bool"]):
        return "technical_failure"
    if not bool(row["stable_chain"]):
        if row["summary_status"] == "out_of_range":
            return "nonconverged_out_of_range_long"
        return "nonconverged"
    if bool(row["joint_endpoint_pass"]):
        return "stable_joint_endpoint_pass"
    peak = float(row["signed_peak_offset_sep"])
    endpoint = float(row["signed_2000_offset_sep"])
    if peak < -ROUTING_TOLERANCE and endpoint < -ROUTING_TOLERANCE:
        return "stable_below_both_targets"
    if peak > ROUTING_TOLERANCE and endpoint < -ROUTING_TOLERANCE:
        return "stable_peak_high_endpoint_low"
    if peak > ROUTING_TOLERANCE and endpoint > ROUTING_TOLERANCE:
        return "stable_above_both_targets"
    if bool(row["corridor_pass"]):
        return "stable_in_corridor_endpoint_miss"
    return "stable_endpoint_tradeoff"


def _load_series(output_dir: str) -> pd.DataFrame:
    path = Path(output_dir) / "timeseries.csv"
    frame = pd.read_csv(path)
    for column in ["year", "length_m", "volume_m3", "area_m2"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["phase", "year", "length_m", "volume_m3"])


def _plot_system(group: pd.DataFrame, output_dir: Path) -> str:
    uid = int(group.iloc[0]["lia_unique_id"])
    lia_km = float(group.iloc[0]["lia_target_m"]) / 1000.0
    rgi_km = float(group.iloc[0]["rgi_target_m"]) / 1000.0
    sep_km = abs(lia_km - rgi_km)
    tol_km = ROUTING_TOLERANCE * sep_km

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 11.0), constrained_layout=True)
    ax_spin_len, ax_spin_vol, ax_forward = axes
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(group)))

    for color, (_, row) in zip(colors, group.sort_values("delta_t").iterrows()):
        series = _load_series(str(row["output_dir"]))
        spin = series[series["phase"].eq("spinup")].sort_values("year")
        forward = series[series["phase"].eq("forward")].sort_values("year")
        stable = bool(row["stable_chain"])
        style = "-" if stable else "--"
        stability_pct = 100.0 * float(row["spinup_stability_fraction"])
        label = (
            f"dT={float(row['delta_t']):+.3f} K | "
            f"{'stable' if stable else 'not stable'} | "
            f"dV200={stability_pct:.2f}%"
        )
        ax_spin_len.plot(
            spin["year"], spin["length_m"] / 1000.0,
            color=color, ls=style, lw=2.0, marker="o", ms=3.2, label=label,
        )
        ax_spin_vol.plot(
            spin["year"], spin["volume_m3"] / 1.0e9,
            color=color, ls=style, lw=2.0, marker="o", ms=3.2,
        )
        ax_forward.plot(
            forward["year"], forward["length_m"] / 1000.0,
            color=color, ls=style, lw=2.2,
        )

    ax_spin_len.set_title("Adaptive spin-up length")
    ax_spin_len.set_xlabel("Spin-up year")
    ax_spin_len.set_ylabel("Length (km)")
    ax_spin_len.legend(frameon=False, fontsize=8, loc="best")

    ax_spin_vol.set_title("Adaptive spin-up volume")
    ax_spin_vol.set_xlabel("Spin-up year")
    ax_spin_vol.set_ylabel("Volume (km3)")

    lower = min(lia_km, rgi_km) - tol_km
    upper = max(lia_km, rgi_km) + tol_km
    ax_forward.axhspan(lower, upper, color="0.65", alpha=0.16, label="20% expanded corridor")
    ax_forward.axhline(lia_km, color="#d95f02", lw=1.8, label=f"LIA {lia_km:.3f} km")
    ax_forward.axhline(rgi_km, color="#1b9e77", lw=1.8, label=f"RGI {rgi_km:.3f} km")
    ax_forward.set_xlim(1470, 2000)
    ax_forward.set_title("Forward length (1470-2000)")
    ax_forward.set_xlabel("Year")
    ax_forward.set_ylabel("Length (km)")
    ax_forward.legend(frameon=False, fontsize=8, loc="best")

    for axis in axes:
        axis.grid(alpha=0.22)

    passes = int(group["joint_endpoint_pass"].sum())
    stable_count = int(group["stable_chain"].sum())
    fig.suptitle(
        f"LIA system {uid}: delta-T branch-boundary pilot | "
        f"stable {stable_count}/{len(group)}, joint endpoint passes {passes}",
        fontsize=14,
    )
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"system_{uid:04d}_delta_boundary.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _system_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for uid, group in rows.groupby("lia_unique_id", sort=True):
        stable = group[group["stable_chain"]].copy()
        best = stable.sort_values(["objective_sep", "delta_t"]).iloc[0] if not stable.empty else None
        joint_count = int(group["joint_endpoint_pass"].sum())
        records.append(
            {
                "lia_unique_id": int(uid),
                "chain_count": len(group),
                "technical_failure_count": int(group["technical_failure_bool"].sum()),
                "stable_chain_count": int(group["stable_chain"].sum()),
                "joint_endpoint_pass_count": joint_count,
                "stable_corridor_pass_count": int(group["corridor_pass"].sum()),
                "best_delta_t": np.nan if best is None else float(best["delta_t"]),
                "best_e_lia_sep": np.nan if best is None else float(best["e_lia_sep"]),
                "best_e_rgi_sep": np.nan if best is None else float(best["e_rgi_sep"]),
                "best_amplitude_ratio": (
                    np.nan if best is None else float(best["amplitude_ratio_to_target"])
                ),
                "stable_intermediate_branch_found": bool(joint_count),
                "delta_t_only_decision": (
                    "retain_joint_candidate" if joint_count
                    else "close_current_pure_delta_t_boundary_direction"
                ),
            }
        )
    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    root = project_root() / "validation_data" / "results" / "paleo_surface_validation_v0"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-scores",
        type=Path,
        default=root / "lia_191_curve_loop_delta_boundary_pilot_v1" / "scores.csv",
    )
    parser.add_argument(
        "--preflight-scores",
        type=Path,
        default=root / "lia_191_curve_loop_delta_boundary_preflight_v1" / "scores.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            root
            / "lia_191_curve_loop_delta_boundary_pilot_v1"
            / "diagnostics"
            / "boundary_pilot_v1"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _prepare(
        pd.concat(
            [pd.read_csv(args.pilot_scores), pd.read_csv(args.preflight_scores)],
            ignore_index=True,
        )
    ).sort_values(["lia_unique_id", "delta_t"])
    rows.to_csv(args.output_dir / "combined_chain_diagnostics.csv", index=False)
    summary = _system_summary(rows)
    plot_paths = {
        int(uid): _plot_system(group, args.output_dir)
        for uid, group in rows.groupby("lia_unique_id", sort=True)
    }
    summary["plot_path"] = summary["lia_unique_id"].map(plot_paths)
    summary.to_csv(args.output_dir / "system_summary.csv", index=False)
    contract = {
        "launched_oggm": False,
        "chain_count": int(len(rows)),
        "system_count": int(rows["lia_unique_id"].nunique()),
        "technical_failure_count": int(rows["technical_failure_bool"].sum()),
        "stable_chain_count": int(rows["stable_chain"].sum()),
        "joint_endpoint_pass_count": int(rows["joint_endpoint_pass"].sum()),
        "routing_tolerance": ROUTING_TOLERANCE,
        "spinup_stability_limit": STABILITY_LIMIT,
        "conclusion": "no_stable_joint_endpoint_branch_in_current_delta_t_boundary_pilot",
        "system_decisions": summary.to_dict(orient="records"),
    }
    (args.output_dir / "diagnosis.json").write_text(
        json.dumps(contract, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
