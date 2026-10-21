"""Apply the frozen retained-32 essential curve standard v3 to all forward cells.

This is a read-only scorer with respect to model outputs.  It writes one new
versioned evidence package and never changes the accepted ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
MATRIX = BASE / "lia_191_all191_oggm_flowline_1d_forward_v10_243_30y_transaction_v1/forward_matrix.csv"
TARGETS = BASE / "lia_191_controller_v1/core191_length_references.csv"
STANDARD = BASE / "lia_191_retained_32_curve_analysis_v3_essential_standard/executable_standard_v3.json"
PRODUCTION_SUMMARY = ROOT / "w237f/r/production_summary_v10_243.json"
OUTPUT = BASE / "lia_191_all191_oggm_flowline_1d_forward_curve_screen_v10_246_v1"

RHO_ICE = 900.0
SMOOTH_YEARS = 21
ACTIVE_SD = 0.25
MAX_NEUTRAL_BRIDGE_YEARS = 5
MIN_PHASE_YEARS = 20
MIN_PHASES_EACH_DIRECTION = 2
MIN_SUSTAINED_VARIATION_SHARE = 0.60
LAG_MAX = 100
LAG_CORRELATION_MIN = 0.30
LAG_SIGN_CONSISTENCY_MIN = 0.55
TERMINAL_MAX_DOWNWARD_SHARE = 0.55
TERMINAL_MIN_ROBUST_Z = 3.0
HF_MIN_RMS_RANGE = 0.08
HF_MIN_TV_SHARE = 0.50
HF_MIN_ZERO_CROSSINGS = 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def reflect_boxcar(values: np.ndarray, window: int = SMOOTH_YEARS) -> np.ndarray:
    half = window // 2
    padded = np.pad(np.asarray(values, dtype=float), half, mode="reflect")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def total_variation(values: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(values))))


def robust_terminal_z(values: np.ndarray, horizon: int = 20) -> float:
    changes = values[horizon:] - values[:-horizon]
    center = float(np.median(changes))
    mad = float(np.median(np.abs(changes - center)))
    scale = max(1.4826 * mad, 1.0e-9)
    final = float(values[-1] - values[-1 - horizon])
    return (center - final) / scale


def lag_metrics(driver: np.ndarray, response: np.ndarray) -> tuple[int, float, float]:
    threshold = ACTIVE_SD * float(np.std(driver))
    rows: list[tuple[int, float, float]] = []
    for lag in range(LAG_MAX + 1):
        x = driver if lag == 0 else driver[:-lag]
        y = response if lag == 0 else response[lag:]
        correlation = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else np.nan
        active = np.abs(x) >= threshold
        consistency = float(np.mean(np.sign(x[active]) == np.sign(y[active]))) if active.any() else np.nan
        rows.append((lag, correlation, consistency))
    finite = [row for row in rows if np.isfinite(row[1])]
    if not finite:
        return -1, float("nan"), float("nan")
    return max(finite, key=lambda row: row[1])


def runs(states: np.ndarray) -> list[tuple[int, int, int]]:
    output: list[tuple[int, int, int]] = []
    start = 0
    for index in range(1, len(states) + 1):
        if index == len(states) or states[index] != states[start]:
            output.append((int(states[start]), start, index - 1))
            start = index
    return output


def phase_metrics(derivative: np.ndarray, length: np.ndarray) -> tuple[int, int, int, float, bool]:
    threshold = ACTIVE_SD * float(np.std(derivative))
    states = np.where(derivative > threshold, 1, np.where(derivative < -threshold, -1, 0)).astype(int)
    for sign in (-1, 1):
        indices = np.flatnonzero(states == sign)
        for left, right in zip(indices[:-1], indices[1:]):
            gap = right - left - 1
            if 0 < gap <= MAX_NEUTRAL_BRIDGE_YEARS and np.all(states[left + 1:right] == 0):
                states[left + 1:right] = sign
    phases = [item for item in runs(states) if item[0] != 0 and item[2] - item[1] + 1 >= MIN_PHASE_YEARS]
    positive = sum(sign > 0 for sign, _, _ in phases)
    negative = sum(sign < 0 for sign, _, _ in phases)
    total = total_variation(length)
    sustained = sum(float(np.sum(np.abs(np.diff(length[start:end + 1])))) for _, start, end in phases)
    share = sustained / max(total, 1.0e-12)
    passed = positive >= 2 and negative >= 2 and share >= MIN_SUSTAINED_VARIATION_SHARE
    return positive, negative, len(phases), share, bool(passed)


def load_diagnostics(path: Path) -> tuple[pd.DataFrame, str]:
    frame = pd.read_json(path, lines=True)
    required = ["year", "length_m", "area_m2", "volume_m3"]
    if any(column not in frame.columns for column in required):
        raise RuntimeError(f"missing-diagnostic-column:{path}")
    frame = frame[required].copy()
    frame["year"] = np.rint(frame["year"].astype(float)).astype(int)
    frame = frame.sort_values("year").drop_duplicates("year", keep="last")
    expected = np.arange(1470, 2001, dtype=int)
    if not np.array_equal(frame["year"].to_numpy(int), expected):
        raise RuntimeError(f"diagnostic-year-contract-failure:{path}")
    values = frame[["length_m", "area_m2", "volume_m3"]].to_numpy(float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise RuntimeError(f"invalid-diagnostic-values:{path}")
    return frame, sha256(path)


def score_one(row: pd.Series, target: pd.Series, diagnostics: pd.DataFrame, diagnostics_hash: str,
              result: dict, result_path: Path) -> dict[str, object]:
    system_id = int(row["lia_unique_id"])
    member_key = str(row["production_member_key"])
    target_lia = float(target["lia_registered_length_m"]) / 1000.0
    target_rgi = float(target["rgi_registered_length_m"]) / 1000.0
    separation = abs(target_lia - target_rgi)
    if separation <= 0:
        raise RuntimeError(f"zero-target-separation:{system_id}")

    peak_km = float(diagnostics["length_m"].max()) / 1000.0
    final_km = float(diagnostics.loc[diagnostics["year"] == 2000, "length_m"].iloc[-1]) / 1000.0
    e_lia_total = abs(peak_km - target_lia) / target_lia
    e_rgi_total = abs(final_km - target_rgi) / target_rgi
    e_lia_sep = abs(peak_km - target_lia) / separation
    e_rgi_sep = abs(final_km - target_rgi) / separation
    endpoint_pass = e_lia_total <= 0.10 and e_rgi_total <= 0.10 and e_lia_sep <= 0.20 and e_rgi_sep <= 0.20

    source = diagnostics[(diagnostics["year"] >= 1499) & (diagnostics["year"] <= 2000)].copy()
    length_raw = source.loc[source["year"] >= 1500, "length_m"].to_numpy(float) / 1000.0
    area_source = source["area_m2"].to_numpy(float)
    volume_source = source["volume_m3"].to_numpy(float)
    area_raw = area_source[1:] / 1.0e6
    volume_raw = volume_source[1:] / 1.0e9
    mb = RHO_ICE * np.diff(volume_source) / (0.5 * (area_source[1:] + area_source[:-1]))
    length = reflect_boxcar(length_raw)
    area = reflect_boxcar(area_raw)
    volume = reflect_boxcar(volume_raw)
    mb_smooth = reflect_boxcar(mb)
    dln_length = np.gradient(np.log(length))

    lag, correlation, consistency = lag_metrics(mb_smooth, dln_length)
    lag_pass = bool(np.isfinite(correlation) and np.isfinite(consistency) and
                    correlation >= LAG_CORRELATION_MIN and consistency >= LAG_SIGN_CONSISTENCY_MIN and lag < LAG_MAX)
    positive, negative, phase_count, sustained_share, phase_pass = phase_metrics(dln_length, length)

    difference = np.diff(length)
    downward_total = float(np.clip(-difference, 0, None).sum())
    terminal_decline = max(0.0, float(length[-21] - length[-1]))
    terminal_down_share = terminal_decline / max(downward_total, 1.0e-9)
    terminal_z = robust_terminal_z(length)
    terminal_drop = bool(terminal_down_share > TERMINAL_MAX_DOWNWARD_SHARE and terminal_z > TERMINAL_MIN_ROBUST_Z)
    residual = length_raw - length
    curve_range = float(np.ptp(length))
    residual_rms_range = float(np.sqrt(np.mean(residual ** 2)) / max(curve_range, 1.0e-9))
    residual_tv_share = total_variation(residual) / max(total_variation(length_raw), 1.0e-9)
    residual_zero_crossings = int(np.sum(np.signbit(residual[1:]) != np.signbit(residual[:-1])))
    high_frequency = bool(residual_rms_range > HF_MIN_RMS_RANGE and residual_tv_share > HF_MIN_TV_SHARE
                          and residual_zero_crossings >= HF_MIN_ZERO_CROSSINGS)
    artifact_pass = not (terminal_drop or high_frequency)
    all_gates = bool(endpoint_pass and lag_pass and phase_pass and artifact_pass)

    failed: list[str] = []
    if not endpoint_pass:
        failed.append("endpoint-bounds")
    if not lag_pass:
        failed.append("lag-response")
    if not phase_pass:
        failed.append("insufficient-bidirectional-persistent-phases")
    if terminal_drop:
        failed.append("terminal-endpoint-drop")
    if high_frequency:
        failed.append("numerical-high-frequency")
    return {
        "member_key": member_key,
        "system_id": system_id,
        "rgi_id": str(row["rgi_id"]),
        "delta_t_eq_degC": float(row["delta_t_eq_degC"]),
        "forward_fingerprint": str(result["forward_fingerprint"]),
        "result_path": str(result_path.relative_to(ROOT)).replace("\\", "/"),
        "result_sha256": sha256(result_path),
        "diagnostics_path": str((result_path.parent / "diagnostics.jsonl").relative_to(ROOT)).replace("\\", "/"),
        "diagnostics_sha256": diagnostics_hash,
        "inherited_upstream_gates_pass": True,
        "lia_target_km": target_lia,
        "rgi_target_km": target_rgi,
        "target_separation_km": separation,
        "source_forward_peak_km_1470_2000": peak_km,
        "source_length_2000_km": final_km,
        "computed_e_lia_total": e_lia_total,
        "computed_e_rgi_total": e_rgi_total,
        "computed_e_lia_separation": e_lia_sep,
        "computed_e_rgi_separation": e_rgi_sep,
        "original_joint_endpoint_bounds_pass": endpoint_pass,
        "length_best_lag_years": lag,
        "length_best_lag_correlation": correlation,
        "length_lag_sign_consistency": consistency,
        "lag_at_upper_boundary": lag == LAG_MAX,
        "lag_gate_pass": lag_pass,
        "persistent_growth_phase_count": positive,
        "persistent_retreat_phase_count": negative,
        "persistent_phase_count_total": phase_count,
        "sustained_variation_share": sustained_share,
        "bidirectional_multistage_gate_pass": phase_pass,
        "terminal_20yr_downward_variation_share": terminal_down_share,
        "terminal_20yr_robust_drop_z": terminal_z,
        "flag_terminal_endpoint_drop": terminal_drop,
        "high_frequency_residual_rms_range_ratio": residual_rms_range,
        "high_frequency_residual_tv_share": residual_tv_share,
        "high_frequency_residual_zero_crossings": residual_zero_crossings,
        "flag_high_frequency": high_frequency,
        "artifact_gate_pass": artifact_pass,
        "all_scientific_curve_gates_pass": all_gates,
        "failed_scientific_reasons": ";".join(failed) if failed else "none",
        "proposed_decision": "provisional-forward-curve-pass" if all_gates else "curve-screen-fail",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transaction", type=Path, required=True)
    args = parser.parse_args()
    transaction = json.loads(args.transaction.read_text(encoding="utf-8"))
    if transaction.get("launch_authorized") is not True:
        raise RuntimeError("curve-screen-not-authorized")
    if sha256(Path(__file__)) != transaction["inputs"]["scorer_sha256"]:
        raise RuntimeError("scorer-hash-mismatch")
    for key, path in [("matrix", MATRIX), ("targets", TARGETS), ("standard", STANDARD),
                      ("production_summary", PRODUCTION_SUMMARY)]:
        if sha256(path) != transaction["inputs"][f"{key}_sha256"]:
            raise RuntimeError(f"{key}-hash-mismatch")
    if OUTPUT.exists():
        raise RuntimeError(f"refusing-to-overwrite:{OUTPUT}")

    matrix = pd.read_csv(MATRIX)
    targets = pd.read_csv(TARGETS).set_index("lia_unique_id")
    if len(matrix) != 530 or matrix["production_member_key"].nunique() != 530:
        raise RuntimeError("matrix-cardinality-failure")
    if len(targets) != 191 or targets.index.nunique() != 191:
        raise RuntimeError("target-cardinality-failure")
    summary = json.loads(PRODUCTION_SUMMARY.read_text(encoding="utf-8"))
    if not (summary.get("batch_completed") and summary.get("completed_cells") == 530):
        raise RuntimeError("production-not-terminal")

    scored: list[dict[str, object]] = []
    technical: list[dict[str, object]] = []
    for _, row in matrix.iterrows():
        key = str(row["production_member_key"])
        result_path = ROOT / "w237f/m" / key / "c/result.json"
        if not result_path.is_file():
            raise RuntimeError(f"missing-result:{key}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        allowed = {str(row["forward_fingerprint"]), str(row["source_forward_fingerprint"])}
        if result.get("forward_fingerprint") not in allowed:
            raise RuntimeError(f"result-fingerprint-mismatch:{key}")
        if bool(result.get("technical_failure")):
            technical.append({"member_key": key, "system_id": int(row["lia_unique_id"]),
                              "rgi_id": str(row["rgi_id"]), "delta_t_eq_degC": float(row["delta_t_eq_degC"]),
                              "status": str(result.get("status")), "error": str(result.get("error", "")),
                              "result_path": str(result_path.relative_to(ROOT)).replace("\\", "/"),
                              "result_sha256": sha256(result_path)})
            continue
        if result.get("status") != "completed" or not result.get("terminal") or result.get("scientific_failure"):
            raise RuntimeError(f"unexpected-result-status:{key}")
        if (str(result.get("member_key")) != key or int(result.get("lia_unique_id")) != int(row["lia_unique_id"])
                or str(result.get("rgi_id")) != str(row["rgi_id"])
                or abs(float(result.get("delta_t_eq_degC")) - float(row["delta_t_eq_degC"])) > 1.0e-12):
            raise RuntimeError(f"result-identity-mismatch:{key}")
        target = targets.loc[int(row["lia_unique_id"])]
        if str(target["rgi_id"]) != str(row["rgi_id"]) or str(target["reference_status"]) != "ok":
            raise RuntimeError(f"target-reference-conflict:{key}")
        diagnostics_path = result_path.parent / "diagnostics.jsonl"
        diagnostics, diagnostics_hash = load_diagnostics(diagnostics_path)
        scored.append(score_one(row, target, diagnostics, diagnostics_hash, result, result_path))

    table = pd.DataFrame(scored).sort_values(["system_id", "delta_t_eq_degC", "member_key"])
    failures = pd.DataFrame(technical).sort_values("member_key")
    if len(table) != 528 or len(failures) != 2:
        raise RuntimeError(f"terminal-classification-cardinality:{len(table)}:{len(failures)}")
    passes = table[table["proposed_decision"] == "provisional-forward-curve-pass"].copy()
    system = table.groupby("system_id", as_index=False).agg(
        rgi_id=("rgi_id", "first"), evaluated_cells=("member_key", "size"),
        provisional_pass_cells=("all_scientific_curve_gates_pass", "sum"),
        endpoint_pass_cells=("original_joint_endpoint_bounds_pass", "sum"),
        lag_pass_cells=("lag_gate_pass", "sum"),
        phase_pass_cells=("bidirectional_multistage_gate_pass", "sum"),
        artifact_pass_cells=("artifact_gate_pass", "sum"),
    )
    system["system_has_provisional_pass"] = system["provisional_pass_cells"] > 0

    temp = OUTPUT.with_name(OUTPUT.name + f".tmp.{os.getpid()}")
    temp.mkdir(parents=True, exist_ok=False)
    table.to_csv(temp / "cell_scores.csv", index=False)
    passes.to_csv(temp / "provisional_pass_cells.csv", index=False)
    system.to_csv(temp / "system_summary.csv", index=False)
    failures.to_csv(temp / "technical_failures.csv", index=False)
    input_manifest = {
        "schema": "lia-all191-forward-curve-screen-input-manifest-v10.246",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": transaction["inputs"],
        "successful_result_count": len(table),
        "technical_failure_count": len(failures),
        "result_sha256_by_member": dict(zip(table["member_key"], table["result_sha256"])),
        "diagnostics_sha256_by_member": dict(zip(table["member_key"], table["diagnostics_sha256"])),
    }
    (temp / "input_manifest.json").write_text(json.dumps(input_manifest, indent=2), encoding="utf-8")
    reason_counts = (table.assign(reason=table["failed_scientific_reasons"].str.split(";"))
                     .explode("reason").query("reason != 'none'")["reason"].value_counts().to_dict())
    result_summary = {
        "schema": "lia-all191-forward-essential-curve-screen-result-v10.246",
        "status": "completed-provisional-screen-no-accepted-ledger-effect",
        "evaluated_cells": len(table),
        "technical_failure_cells_excluded": len(failures),
        "provisional_pass_cells": len(passes),
        "curve_screen_fail_cells": int(len(table) - len(passes)),
        "evaluated_systems": int(table["system_id"].nunique()),
        "systems_with_provisional_pass": int(system["system_has_provisional_pass"].sum()),
        "systems_without_provisional_pass": int((~system["system_has_provisional_pass"]).sum()),
        "failure_reason_counts_nonexclusive": {str(k): int(v) for k, v in reason_counts.items()},
        "formal_accepted_count_before": 58,
        "formal_accepted_count_after": 58,
        "independent_reproduction_completed": False,
        "simulation_launched": False,
        "output_files": ["cell_scores.csv", "provisional_pass_cells.csv", "system_summary.csv",
                         "technical_failures.csv", "input_manifest.json"],
    }
    (temp / "summary.json").write_text(json.dumps(result_summary, indent=2), encoding="utf-8")
    os.replace(temp, OUTPUT)
    print(json.dumps(result_summary, indent=2))


if __name__ == "__main__":
    main()
