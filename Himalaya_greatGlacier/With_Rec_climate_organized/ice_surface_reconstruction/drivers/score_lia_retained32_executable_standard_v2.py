"""Apply the proposed executable endpoint-plus-shape standard to retained-32.

This script is read-only with respect to model outputs and controller state. It
writes a new versioned analysis package only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
V1 = BASE / "lia_191_retained_32_curve_analysis_v1"
REGISTRY = BASE / "lia_191_accepted_58_registry_v1/accepted_58_curve_metrics.csv"
OUTPUT = BASE / "lia_191_retained_32_curve_analysis_v2_executable_standard"

LAG_MIN = 0
LAG_MAX = 100
LAG_CORRELATION_MIN = 0.30
LAG_SIGN_CONSISTENCY_MIN = 0.55
ACTIVE_DRIVER_SD = 0.25
SUSTAINED_RUN_MIN_YEARS = 20
SUSTAINED_VARIATION_SHARE_MIN = 0.60
NEUTRAL_BRIDGE_MAX_YEARS = 5
ISOLATED_PEAK_MAX_WIDTH_YEARS = 30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def lag_metrics(driver: np.ndarray, response: np.ndarray) -> tuple[int, float, float]:
    threshold = ACTIVE_DRIVER_SD * float(np.std(driver))
    rows: list[tuple[int, float, float]] = []
    for lag in range(LAG_MIN, LAG_MAX + 1):
        x = driver if lag == 0 else driver[:-lag]
        y = response if lag == 0 else response[lag:]
        correlation = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else np.nan
        active = np.abs(x) >= threshold
        consistency = float(np.mean(np.sign(x[active]) == np.sign(y[active]))) if active.any() else np.nan
        rows.append((lag, correlation, consistency))
    finite = [row for row in rows if np.isfinite(row[1])]
    return max(finite, key=lambda row: row[1])


def sign_runs(states: np.ndarray) -> list[tuple[int, int, int]]:
    output: list[tuple[int, int, int]] = []
    start = 0
    for index in range(1, len(states) + 1):
        if index == len(states) or states[index] != states[start]:
            if states[start] != 0:
                output.append((int(states[start]), start, index - 1))
            start = index
    return output


def sustained_variation(length: np.ndarray) -> tuple[float, int, int]:
    rate = np.gradient(length)
    threshold = ACTIVE_DRIVER_SD * float(np.std(rate))
    states = np.where(rate > threshold, 1, np.where(rate < -threshold, -1, 0))
    for sign in (-1, 1):
        indices = np.where(states == sign)[0]
        for left, right in zip(indices[:-1], indices[1:]):
            gap = right - left - 1
            if 0 < gap <= NEUTRAL_BRIDGE_MAX_YEARS and np.all(states[left + 1 : right] == 0):
                states[left + 1 : right] = sign
    accepted = [run for run in sign_runs(states) if run[2] - run[1] + 1 >= SUSTAINED_RUN_MIN_YEARS]
    total = float(np.sum(np.abs(np.diff(length))))
    sustained = 0.0
    for _, start, end in accepted:
        sustained += float(np.sum(np.abs(np.diff(length[start : end + 1]))))
    return sustained / max(total, 1.0e-12), len(accepted), max((end - start + 1 for _, start, end in accepted), default=0)


def source_endpoint_lengths(source: Path, length_column: str) -> tuple[float, float]:
    frame = pd.read_csv(source)
    if "phase" in frame.columns:
        frame = frame[frame["phase"].astype(str) == "forward"]
    years = np.rint(frame["year"].astype(float)).astype(int)
    frame = frame.assign(year_int=years)
    frame = frame[(frame["year_int"] >= 1470) & (frame["year_int"] <= 2000)]
    peak_km = float(frame[length_column].astype(float).max() / 1000.0)
    final_rows = frame[frame["year_int"] == 2000]
    if final_rows.empty:
        raise RuntimeError(f"missing 2000 endpoint: {source}")
    final_km = float(final_rows.iloc[-1][length_column]) / 1000.0
    return peak_km, final_km


def markdown_table(table: pd.DataFrame) -> str:
    lines = [
        "| ID | Original bounds | Lag years/r/sign | Sustained share | Shape flags | Evidence conflict | Decision |\n",
        "|---:|---|---|---:|---|---|---|\n",
    ]
    for _, row in table.sort_values("system_id").iterrows():
        flags = []
        for key, label in [
            ("flag_near_monotonic", "near-monotonic"),
            ("flag_isolated_short_peak", "isolated-short-peak"),
            ("flag_terminal_endpoint_drop", "terminal-drop"),
            ("flag_high_frequency", "high-frequency"),
            ("flag_net_growth", "net-growth"),
        ]:
            if bool(row[key]):
                flags.append(label)
        lines.append(
            f"| {int(row['system_id'])} | {'pass' if row['original_joint_endpoint_bounds_pass'] else 'fail'} | "
            f"{int(row['length_best_lag_years'])}/{row['length_best_lag_correlation']:.2f}/{row['length_lag_sign_consistency']:.2f} | "
            f"{row['sustained_variation_share']:.2f} | {', '.join(flags) if flags else 'none'} | "
            f"{'yes' if row['evidence_conflict'] else 'no'} | {row['proposed_decision']} |\n"
        )
    return "".join(lines)


def main() -> None:
    old = pd.read_csv(V1 / "classification_table.csv")
    annual = pd.read_csv(V1 / "annual_smoothed_multivariate_series.csv")
    registry = pd.read_csv(REGISTRY).set_index("system_id")
    if len(old) != 32 or old["system_id"].nunique() != 32:
        raise RuntimeError("v1 input is not exactly retained-32")

    rows: list[dict[str, object]] = []
    for _, prior in old.sort_values("system_id").iterrows():
        system_id = int(prior["system_id"])
        group = annual[annual["system_id"] == system_id].sort_values("year")
        source = ROOT / str(prior["source"])
        peak_km, final_km = source_endpoint_lengths(source, str(prior["length_source_column"]))
        target_lia = float(registry.loc[system_id, "lia_km"])
        target_rgi = float(registry.loc[system_id, "rgi_km"])
        separation = abs(target_lia - target_rgi)
        lia_tolerance = min(0.10 * target_lia, 0.20 * separation)
        rgi_tolerance = min(0.10 * target_rgi, 0.20 * separation)
        e_lia_total = abs(peak_km - target_lia) / target_lia
        e_rgi_total = abs(final_km - target_rgi) / target_rgi
        e_lia_sep = abs(peak_km - target_lia) / separation
        e_rgi_sep = abs(final_km - target_rgi) / separation
        endpoint_pass = bool(
            e_lia_total <= 0.10 and e_rgi_total <= 0.10
            and e_lia_sep <= 0.20 and e_rgi_sep <= 0.20
        )

        b = group["net_mb_21yr_kg_m2_yr"].to_numpy(float)
        dln_length = group["dln_length_21yr_per_year"].to_numpy(float)
        length = group["length_21yr_km"].to_numpy(float)
        lag, correlation, consistency = lag_metrics(b, dln_length)
        lag_pass = bool(correlation >= LAG_CORRELATION_MIN and consistency >= LAG_SIGN_CONSISTENCY_MIN and lag < LAG_MAX)
        sustained_share, sustained_runs, longest_run = sustained_variation(length)
        sustained_pass = bool(sustained_share >= SUSTAINED_VARIATION_SHARE_MIN)

        curve_range = float(np.ptp(length))
        peaks, _ = find_peaks(length, prominence=max(0.10 * curve_range, 1.0e-12), distance=40)
        widths = peak_widths(length, peaks, rel_height=0.5)[0] if len(peaks) else np.array([])
        widest_peak = float(np.max(widths)) if len(widths) else 0.0
        isolated = bool(not sustained_pass and len(peaks) <= 2 and widest_peak < ISOLATED_PEAK_MAX_WIDTH_YEARS)
        near_monotonic = bool(prior["flag_near_monotonic"])
        terminal = bool(prior["flag_terminal_endpoint_drop"])
        high_frequency = bool(prior["flag_numerical_high_frequency_oscillation"])
        net_growth = bool(prior["flag_net_growth_or_wrong_direction"])
        artifact_pass = bool(not (near_monotonic or isolated or terminal or high_frequency or net_growth))

        registered_lia_error = float(registry.loc[system_id, "e_lia"])
        registered_rgi_error = float(registry.loc[system_id, "e_rgi"])
        evidence_conflict = bool(
            abs(registered_lia_error - e_lia_sep) > 1.0e-6
            or abs(registered_rgi_error - e_rgi_sep) > 1.0e-6
        )
        all_scientific_gates = bool(endpoint_pass and lag_pass and sustained_pass and artifact_pass)
        if evidence_conflict:
            decision = "evidence-conflict-hold"
        elif all_scientific_gates:
            decision = "provisional-forward-shape-pass"
        elif lag == LAG_MAX and correlation >= LAG_CORRELATION_MIN and consistency >= LAG_SIGN_CONSISTENCY_MIN:
            decision = "uncertain-lag-at-upper-bound"
        else:
            decision = "shape-screen-fail"

        rows.append({
            "system_id": system_id,
            "rgi_id": prior["rgi_id"],
            "model_family_registry_label": prior["model_family_registry_label"],
            "route": prior["route"],
            "source": prior["source"],
            "source_sha256": prior["source_sha256"],
            "source_status": prior["source_status"],
            "analysis_status": "read-only-proposed-executable-standard-v2",
            "direct_surface_mb_available": bool(prior["direct_surface_mb_available"]),
            "mb_role": "derived-net-volume-change-equivalent; internal-coherence-only",
            "lia_target_km": target_lia,
            "rgi_target_km": target_rgi,
            "target_separation_km": separation,
            "lia_lower_bound_km": target_lia - lia_tolerance,
            "lia_upper_bound_km": target_lia + lia_tolerance,
            "rgi_lower_bound_km": target_rgi - rgi_tolerance,
            "rgi_upper_bound_km": target_rgi + rgi_tolerance,
            "source_forward_peak_km_1470_2000": peak_km,
            "source_length_2000_km": final_km,
            "computed_e_lia_total": e_lia_total,
            "computed_e_rgi_total": e_rgi_total,
            "computed_e_lia_separation": e_lia_sep,
            "computed_e_rgi_separation": e_rgi_sep,
            "registered_e_lia_separation": registered_lia_error,
            "registered_e_rgi_separation": registered_rgi_error,
            "original_joint_endpoint_bounds_pass": endpoint_pass,
            "length_best_lag_years": lag,
            "length_best_lag_correlation": correlation,
            "length_lag_sign_consistency": consistency,
            "lag_at_upper_boundary": lag == LAG_MAX,
            "lag_gate_pass": lag_pass,
            "sustained_variation_share": sustained_share,
            "sustained_run_count": sustained_runs,
            "longest_sustained_run_years": longest_run,
            "sustained_response_gate_pass": sustained_pass,
            "prominent_peak_count_auxiliary": int(len(peaks)),
            "widest_peak_half_prominence_years_auxiliary": widest_peak,
            "flag_near_monotonic": near_monotonic,
            "flag_isolated_short_peak": isolated,
            "flag_terminal_endpoint_drop": terminal,
            "flag_high_frequency": high_frequency,
            "flag_net_growth": net_growth,
            "artifact_direction_gate_pass": artifact_pass,
            "evidence_conflict": evidence_conflict,
            "all_proposed_scientific_gates_pass": all_scientific_gates,
            "proposed_decision": decision,
        })

    table = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / "rescored_32_executable_standard.csv", index=False)
    (OUTPUT / "rescored_32_executable_standard.md").write_text(markdown_table(table), encoding="utf-8")

    passes = table.loc[table["proposed_decision"] == "provisional-forward-shape-pass", "system_id"].astype(int).tolist()
    conflicts = table.loc[table["evidence_conflict"], "system_id"].astype(int).tolist()
    failures = table.loc[table["proposed_decision"] == "shape-screen-fail", "system_id"].astype(int).tolist()
    uncertain = table.loc[table["proposed_decision"].str.startswith("uncertain"), "system_id"].astype(int).tolist()

    standard = {
        "schema": "lia-retained32-executable-curve-standard-v2",
        "status": "proposed-read-only-standard-not-canonical",
        "order": [
            "inherited upstream numerical/geometry/stability/observation-domain evidence",
            "original joint endpoint bounds",
            "lagged length-response coherence",
            "sustained low-frequency length response",
            "artifact and net-direction exclusion",
            "independent reproduction before formal acceptance",
        ],
        "original_joint_endpoint_bounds": {
            "lia_peak_total_length_error_max": 0.10,
            "rgi_2000_total_length_error_max": 0.10,
            "lia_peak_separation_normalized_error_max": 0.20,
            "rgi_2000_separation_normalized_error_max": 0.20,
            "equivalent_tolerance": "min(0.10*target_length, 0.20*abs(LIA-RGI)) around each target",
            "whole_curve_corridor_is_hard_gate": False,
        },
        "common_processing": {
            "analysis_years": [1500, 2000],
            "smoothing": "21-year centered reflected boxcar",
            "direct_surface_mb_required": False,
            "derived_mb_role": "internal-coherence diagnostic only; not independent climate attribution",
        },
        "lagged_length_response": {
            "lag_years": [LAG_MIN, LAG_MAX],
            "minimum_correlation": LAG_CORRELATION_MIN,
            "minimum_sign_consistency": LAG_SIGN_CONSISTENCY_MIN,
            "active_driver_threshold_sd": ACTIVE_DRIVER_SD,
            "best_lag_at_100_years": "uncertain, not pass",
        },
        "sustained_response": {
            "minimum_directional_run_years": SUSTAINED_RUN_MIN_YEARS,
            "maximum_neutral_bridge_years": NEUTRAL_BRIDGE_MAX_YEARS,
            "minimum_share_of_smoothed_length_total_variation": SUSTAINED_VARIATION_SHARE_MIN,
            "minimum_number_of_stages": None,
        },
        "artifact_direction_exclusions": [
            "nonnegative 1500-2000 smoothed net length change",
            "near-monotonic decline under the registered up-variation/rebound definition",
            "isolated short peak defined jointly by low sustained-variation share and <30-year half-prominence width",
            "terminal endpoint drop",
            "numerical high-frequency oscillation",
        ],
        "diagnostic_only_not_hard": [
            "area amplitude", "volume amplitude", "peak count alone", "B-to-volume correlation",
            "whole-curve corridor", "minimum-at-2000", "modern MB continuation",
        ],
        "decision_rule": "all non-diagnostic gates must pass; evidence conflict or lag at 100-year boundary is a hold",
    }
    (OUTPUT / "executable_standard_v2.json").write_text(json.dumps(standard, indent=2), encoding="utf-8")

    review = f"""# Previous standards and proposed executable curve standard v2

## What previously counted as a pass

1. The older provisional rule required a numerically clean, stable, geometry-valid chain whose forward peak and year-2000 length were each within 10% of the corresponding total-length target.
2. The later formal Policy-V6 rule retained those two 10% gates and added separation-normalized errors no larger than 20% at both endpoints. The executable interval around each target is therefore `target +/- min(0.10*target, 0.20*|LIA-RGI|)`.
3. Numerical cleanliness, geometry/length-definition validity, observation-domain validity, equilibrium/spin-up stability, and exact independent reproduction were upstream acceptance gates. They cannot be reconstructed from the length curve alone and remain inherited prerequisites.
4. The V3 whole-curve corridor, minimum-at-2000, amplitude ratio, area, volume, and modern-MB continuation were diagnostics rather than the current formal acceptance rule.
5. The accepted-58 temporary shape screen used an 11-year smooth and temporarily excluded a net-declining curve only when upward-variation share was no more than 0.15 and rebound/range was no more than 0.20. It did not alter the accepted ledger.
6. The v1 retained-32 research analysis added a three-stage count and area/volume amplitude gates. These are now removed: stage count is smoothing-sensitive, area/volume are not observational endpoint variables, and a 50% length-amplitude gate is mathematically redundant once both 20%-separation endpoint gates pass (the endpoints already imply at least 60% target separation).

## Proposed executable order

1. Verify inherited upstream evidence and source hashes. Any conflict stops only the affected member.
2. Apply the original joint endpoint bounds: both 10% total-length and 20% separation-normalized errors at the 1470-2000 forward peak and year 2000.
3. On 1500-2000 data, apply one 21-year centered reflected smooth. Search only B(t) to later dlnL/dt lags from 0 to 100 years. Require r >= 0.30 and sign consistency >= 0.55. A best lag at exactly 100 years is uncertain.
4. Require sustained response without counting stages: at least 60% of smoothed length total variation must occur in directional runs lasting at least 20 years; bridge at most five neutral years.
5. Require net retreat and exclude near-monotonic, isolated-short-peak, terminal-drop, and high-frequency artifacts. Peak number is never sufficient by itself.
6. Treat area, volume, and derived B-to-volume agreement as diagnostics. Direct climatic response remains unconfirmed when direct surface MB is absent.
7. A curve passing these checks is only a provisional forward-shape pass. Formal acceptance still requires the applicable independent reproduction and controller transaction.

## Read-only retained-32 result

- Provisional forward-shape passes: **{len(passes)}** — {', '.join(map(str, passes)) if passes else 'none'}.
- Shape failures: **{len(failures)}** — {', '.join(map(str, failures)) if failures else 'none'}.
- Lag-boundary uncertain: **{len(uncertain)}** — {', '.join(map(str, uncertain)) if uncertain else 'none'}.
- Evidence-conflict holds: **{len(conflicts)}** — {', '.join(map(str, conflicts)) if conflicts else 'none'}.

System 1326 is held because the accepted-58 registry records e_lia=0.258177 while its current source curve, canonical facts, formal adjudication, and independent reproduction record e_lia=0.190823. The current source historical-summary hash exactly matches the canonical adjudication hash, so the registry value must not be used silently.

{markdown_table(table)}
"""
    (OUTPUT / "previous_standard_review_and_v2_proposal.md").write_text(review, encoding="utf-8")

    summary = {
        "schema": "lia-retained32-executable-standard-rescore-v2",
        "status": "read-only-rescore-complete-proposed-not-canonical",
        "system_count": 32,
        "provisional_forward_shape_pass_count": len(passes),
        "provisional_forward_shape_pass_system_ids": passes,
        "shape_fail_count": len(failures),
        "shape_fail_system_ids": failures,
        "lag_boundary_uncertain_count": len(uncertain),
        "lag_boundary_uncertain_system_ids": uncertain,
        "evidence_conflict_hold_count": len(conflicts),
        "evidence_conflict_hold_system_ids": conflicts,
        "formal_accepted_count_unchanged": 58,
        "canonical_state_modified": False,
        "simulation_launched_stopped_resumed_or_modified": False,
        "input_hashes": {
            "v1_classification": sha256(V1 / "classification_table.csv"),
            "v1_annual_series": sha256(V1 / "annual_smoothed_multivariate_series.csv"),
            "accepted_58_registry": sha256(REGISTRY),
        },
        "outputs": {
            "review": "previous_standard_review_and_v2_proposal.md",
            "standard": "executable_standard_v2.json",
            "rescored_csv": "rescored_32_executable_standard.csv",
            "rescored_markdown": "rescored_32_executable_standard.md",
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
