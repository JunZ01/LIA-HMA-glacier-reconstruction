"""Score and plot spatial-downstream reruns under the active curve contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GEOMETRY_VARIANT = "core-lia-plus-spatial-dem-downstream-v2"


def geometry_fingerprint(source_fingerprint: str) -> str:
    payload = f"{source_fingerprint}|{GEOMETRY_VARIANT}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _corridor_exceedance(length: np.ndarray, lia: float, rgi: float) -> float:
    separation = abs(lia - rgi)
    lower = min(lia, rgi) - 0.2 * separation
    upper = max(lia, rgi) + 0.2 * separation
    return float(
        max(
            np.max(np.maximum(lower - length, 0.0)),
            np.max(np.maximum(length - upper, 0.0)),
        )
    )


def _new_output_dir(
    row: pd.Series,
    *,
    rerun_root: Path,
    preflight_2955_root: Path,
) -> Path:
    unique_id = int(row["lia_unique_id"])
    fingerprint = str(row["fingerprint"])
    if unique_id == 2955:
        return preflight_2955_root / f"system_{unique_id}" / fingerprint[:16]
    source_task = json.loads(
        (Path(str(row["output_dir"])) / "task.json").read_text(encoding="utf-8")
    )
    return (
        rerun_root
        / f"system_{unique_id}"
        / str(source_task["block"])
        / fingerprint[:16]
    )


def _score_one(row: pd.Series, output_dir: Path) -> tuple[dict, pd.DataFrame]:
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    series = pd.read_csv(output_dir / "timeseries.csv")
    forward = series[series["phase"].astype(str).eq("forward")].copy()
    observation_length = pd.to_numeric(
        forward["observation_length_m"], errors="coerce"
    )
    if observation_length.isna().any():
        raise ValueError(f"nonfinite-observation-length:{output_dir}")
    forward_peak_m = float(observation_length.max())
    forward_2000_m = float(
        observation_length[forward["year"].astype(int).eq(2000)].iloc[-1]
    )
    forward_minimum_m = float(observation_length.min())
    peak_year = int(forward.loc[observation_length.idxmax(), "year"])
    lia_target_m = float(summary["targets"]["lia_length_m"])
    rgi_target_m = float(summary["targets"]["rgi_length_m"])
    separation_m = abs(lia_target_m - rgi_target_m)
    e_lia_total = abs(forward_peak_m - lia_target_m) / lia_target_m
    e_rgi_total = abs(forward_2000_m - rgi_target_m) / rgi_target_m
    e_lia_sep = abs(forward_peak_m - lia_target_m) / separation_m
    e_rgi_sep = abs(forward_2000_m - rgi_target_m) / separation_m
    amplitude_m = forward_peak_m - forward_2000_m
    source_fingerprint = str(row["fingerprint"])
    fingerprint = geometry_fingerprint(source_fingerprint)
    source_task = json.loads(
        (Path(str(row["output_dir"])) / "task.json").read_text(encoding="utf-8")
    )
    spinup = summary["spinup_control"]
    observation = summary["observation_domain"]
    technical_failure = summary.get("status") not in {"ok"}
    stable = bool(
        not technical_failure
        and _as_bool(spinup.get("converged"))
        and float(spinup.get("final_stability_fraction", np.inf)) <= 0.02
    )
    accepted = bool(
        stable
        and not _as_bool(observation.get("overflow"))
        and e_lia_total <= 0.10
        and e_rgi_total <= 0.10
    )
    score = {
        "task_id": (
            f"{int(row['lia_unique_id'])}-{source_task['block']}-"
            f"spatial-downstream-v2-{fingerprint[:16]}"
        ),
        "fingerprint": fingerprint,
        "source_fingerprint": source_fingerprint,
        "geometry_variant": GEOMETRY_VARIANT,
        "lia_unique_id": int(row["lia_unique_id"]),
        "rgi_id": summary["rgi_id"],
        "block": source_task["block"],
        "delta_t": float(summary["parameters"]["delta_t_c"]),
        "melt_f": float(summary["parameters"]["melt_f"]),
        "prcp_fac_multiplier": float(
            summary["parameters"]["prcp_fac_multiplier"]
        ),
        "glen_a": float(summary["parameters"]["glen_a_pa_3_s_1"]),
        "temp_melt": float(summary["parameters"]["temp_melt_c"]),
        "melt_f_stage2_multiplier": float(
            source_task.get("melt_f_stage2_multiplier", 1.0)
        ),
        "task_state": "completed",
        "output_dir": str(output_dir.resolve()),
        "workdir": "",
        "created_at_utc": "",
        "updated_at_utc": "",
        "returncode": 0,
        "error": "",
        "runtime_s_controller": float(summary["timing_seconds"]["total"]),
        "summary_status": summary["status"],
        "technical_failure": technical_failure,
        "spinup_converged": bool(spinup["converged"]),
        "spinup_stability_fraction": float(spinup["final_stability_fraction"]),
        "lia_target_m": lia_target_m,
        "rgi_target_m": rgi_target_m,
        "lia_rgi_separation_m": separation_m,
        "forward_peak_m": forward_peak_m,
        "forward_peak_year": peak_year,
        "forward_2000_m": forward_2000_m,
        "forward_minimum_m": forward_minimum_m,
        "e_lia_sep": e_lia_sep,
        "e_rgi_sep": e_rgi_sep,
        "objective_sep": e_lia_sep + e_rgi_sep,
        "peak_to_2000_amplitude_m": amplitude_m,
        "amplitude_ratio_to_target": amplitude_m / separation_m,
        "max_corridor_exceedance_m": _corridor_exceedance(
            observation_length.to_numpy(dtype=float), lia_target_m, rgi_target_m
        ),
        "acceptance_tolerance": 0.10,
        "accepted": accepted,
        "acceptance_state": (
            "accepted-spatial-downstream-v2"
            if accepted
            else "rejected-spatial-downstream-v2"
        ),
        "observation_domain_node_count": int(observation["node_count"]),
        "observation_domain_overflow": bool(observation["overflow"]),
    }
    return score, forward


def _plot_system(
    unique_id: int,
    entries: list[tuple[dict, pd.DataFrame]],
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(10.0, 5.8))
    for score, forward in entries:
        label = (
            f"deltaT={score['delta_t']:+.4g} K, "
            f"melt_f={score['melt_f']:.3g}"
        )
        axis.plot(
            forward["year"],
            forward["observation_length_m"] / 1000.0,
            linewidth=1.8,
            label=label,
        )
    lia = float(entries[0][0]["lia_target_m"]) / 1000.0
    rgi = float(entries[0][0]["rgi_target_m"]) / 1000.0
    axis.axhline(lia, color="0.2", linestyle="--", linewidth=1.2, label="LIA target")
    axis.axhline(rgi, color="0.2", linestyle=":", linewidth=1.2, label="RGI target")
    axis.set_xlabel("Year")
    axis.set_ylabel("Length in spatial observation domain (km)")
    axis.set_title(f"LIA system {unique_id}: spatial-downstream rerun")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run(
    *,
    invalidated_csv: Path,
    rerun_root: Path,
    preflight_2955_root: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    invalidated = pd.read_csv(invalidated_csv)
    scored_entries: list[tuple[dict, pd.DataFrame]] = []
    for _, row in invalidated.iterrows():
        output = _new_output_dir(
            row,
            rerun_root=rerun_root,
            preflight_2955_root=preflight_2955_root,
        )
        scored_entries.append(_score_one(row, output))
    scores = pd.DataFrame([entry[0] for entry in scored_entries]).sort_values(
        ["lia_unique_id", "objective_sep"]
    ).reset_index(drop=True)
    audits = scores[
        [
            "fingerprint",
            "source_fingerprint",
            "geometry_variant",
            "lia_unique_id",
            "observation_domain_node_count",
            "observation_domain_overflow",
            "output_dir",
        ]
    ].copy()
    audits["observation_domain_state"] = "observation-domain-valid"
    audits["length_observation_valid"] = True
    audits["observation_domain_unresolved"] = False

    output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_dir / "scores.csv", index=False)
    audits.to_csv(output_dir / "chain_observation_domain_audit.csv", index=False)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for unique_id in sorted(scores["lia_unique_id"].unique()):
        entries = [
            entry
            for entry in scored_entries
            if int(entry[0]["lia_unique_id"]) == int(unique_id)
        ]
        _plot_system(
            int(unique_id), entries, plot_dir / f"system_{int(unique_id)}_curve.png"
        )
    summary = {
        "status": "completed",
        "geometry_variant": GEOMETRY_VARIANT,
        "chain_count": int(len(scores)),
        "system_count": int(scores["lia_unique_id"].nunique()),
        "technical_failure_count": int(scores["technical_failure"].sum()),
        "observation_domain_overflow_count": int(
            scores["observation_domain_overflow"].sum()
        ),
        "accepted_10pct_chain_count": int(scores["accepted"].sum()),
        "accepted_10pct_system_count": int(
            scores.loc[scores["accepted"], "lia_unique_id"].nunique()
        ),
        "parameter_search": False,
        "output_directory": str(output_dir.resolve()),
    }
    (output_dir / "score_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return scores, audits, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--invalidated-csv", required=True, type=Path)
    parser.add_argument("--rerun-root", required=True, type=Path)
    parser.add_argument("--preflight-2955-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    _, _, summary = run(
        invalidated_csv=args.invalidated_csv,
        rerun_root=args.rerun_root,
        preflight_2955_root=args.preflight_2955_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
