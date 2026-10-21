"""Derive trust-region parameter updates from saved OFAT chains."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear


PARAMS = ["delta_t", "melt_f", "prcp_fac_multiplier", "glen_a_multiplier", "temp_melt"]
BOUNDS = {
    "delta_t": (-3.0, 1.5),
    "melt_f": (2.5, 7.5),
    "prcp_fac_multiplier": (0.8, 1.2),
    "glen_a_multiplier": (0.5, 3.0),
    "temp_melt": (-2.0, 0.0),
}


def _signed_residual(
    chain_dir: Path,
    lia: float,
    rgi: float,
) -> tuple[np.ndarray, list[str]]:
    series = pd.read_csv(chain_dir / "timeseries.csv")
    spin = series[series["phase"].astype(str).eq("spinup")].sort_values("year")
    forward = series[series["phase"].astype(str).eq("forward")].sort_values("year")
    if spin.empty or forward.empty:
        raise ValueError(f"missing phase in {chain_dir}")
    end = spin.iloc[-1]
    start_year = float(end.year) - 200.0
    start = spin.iloc[(pd.to_numeric(spin["year"]) - start_year).abs().argmin()]
    r_lia = (float(end.length_m) - lia) / lia / 0.10
    r_rgi = (float(forward.iloc[-1].length_m) - rgi) / rgi / 0.10
    r_spin = (float(end.volume_m3) - float(start.volume_m3)) / float(start.volume_m3) / 0.02

    gap = abs(lia - rgi)
    buffer = 0.20 * gap
    scale = max(buffer, 1.0)
    low = min(lia, rgi) - buffer
    high = max(lia, rgi) + buffer
    lengths = pd.to_numeric(forward["length_m"]).to_numpy(float)
    signed = np.where(lengths < low, lengths - low, np.where(lengths > high, lengths - high, 0.0))
    curve = signed / scale / np.sqrt(len(signed))
    years = pd.to_numeric(forward["year"]).to_numpy(int)
    labels = ["lia_endpoint", "rgi_endpoint", "spinup_stability"] + [f"corridor_{year}" for year in years]
    return np.concatenate(([r_lia, r_rgi, r_spin], curve)), labels


def _trial(group: pd.DataFrame, name: str) -> pd.Series:
    rows = group[group["ofat_trial"].astype(str).eq(name)]
    if len(rows) != 1:
        raise ValueError(f"expected one {name}, found {len(rows)}")
    return rows.iloc[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ofat-results", type=Path, required=True)
    parser.add_argument("--targets-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=0.5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.ofat_results)
    targets = pd.read_csv(args.targets_csv).set_index("lia_unique_id")

    jacobian_rows = []
    proposal_rows = []
    diagnostic_rows = []
    for system_id, group in table.groupby(table["system_id"].astype(int)):
        target = targets.loc[int(system_id)]
        lia = float(target.lia_registered_length_m)
        rgi = float(target.rgi_registered_length_m)
        base = _trial(group, "base")
        z0, labels = _signed_residual(Path(str(base.output_dir)), lia, rgi)
        columns = []
        nominal_steps = []
        for parameter in PARAMS:
            minus = _trial(group, f"{parameter}_minus")
            plus = _trial(group, f"{parameter}_plus")
            z_minus, minus_labels = _signed_residual(Path(str(minus.output_dir)), lia, rgi)
            z_plus, plus_labels = _signed_residual(Path(str(plus.output_dir)), lia, rgi)
            if minus_labels != labels or plus_labels != labels:
                raise ValueError(f"time-grid mismatch for {system_id}/{parameter}")
            p0 = float(base[parameter])
            p_minus = float(minus[parameter])
            p_plus = float(plus[parameter])
            nominal = max(abs(p_plus - p0), abs(p0 - p_minus))
            if nominal <= 0 or p_plus == p_minus:
                raise ValueError(f"zero perturbation for {system_id}/{parameter}")
            column = (z_plus - z_minus) / ((p_plus - p_minus) / nominal)
            columns.append(column)
            nominal_steps.append(nominal)
            for label, value in zip(labels, column):
                jacobian_rows.append({
                    "system_id": int(system_id),
                    "residual": label,
                    "parameter": parameter,
                    "derivative_per_nominal_step": float(value),
                    "base_residual": float(z0[labels.index(label)]),
                    "nominal_parameter_step": nominal,
                    "parameter_base": p0,
                    "parameter_minus": p_minus,
                    "parameter_plus": p_plus,
                })

        matrix = np.column_stack(columns)
        singular = np.linalg.svd(matrix, compute_uv=False)
        rank = int(np.linalg.matrix_rank(matrix))
        condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else np.inf
        lower = []
        upper = []
        for index, parameter in enumerate(PARAMS):
            p0 = float(base[parameter])
            low, high = BOUNDS[parameter]
            lower.append(max(-1.0, (low - p0) / nominal_steps[index]))
            upper.append(min(1.0, (high - p0) / nominal_steps[index]))
        augmented_matrix = np.vstack([matrix, np.sqrt(args.ridge) * np.eye(len(PARAMS))])
        augmented_target = np.concatenate([-z0, np.zeros(len(PARAMS))])
        solution = lsq_linear(
            augmented_matrix,
            augmented_target,
            bounds=(np.asarray(lower), np.asarray(upper)),
            lsmr_tol="auto",
        )
        dx = solution.x
        base_objective = float(z0 @ z0)

        for alpha in (0.25, 0.50, 1.00):
            proposed = {}
            actual_dx = []
            for index, parameter in enumerate(PARAMS):
                value = float(base[parameter]) + alpha * dx[index] * nominal_steps[index]
                value = float(np.clip(value, *BOUNDS[parameter]))
                proposed[parameter] = value
                actual_dx.append((value - float(base[parameter])) / nominal_steps[index])
            predicted = z0 + matrix @ np.asarray(actual_dx)
            proposal_rows.append({
                "system_id": int(system_id),
                "alpha": alpha,
                "base_objective_linearized": base_objective,
                "predicted_objective_linearized": float(predicted @ predicted),
                "predicted_improvement_fraction": float(1.0 - (predicted @ predicted) / base_objective),
                **proposed,
            })

        diagnostic_rows.append({
            "system_id": int(system_id),
            "residual_count": len(z0),
            "parameter_count": len(PARAMS),
            "matrix_rank": rank,
            "condition_number": condition,
            "ridge": args.ridge,
            "bounded_solver_success": bool(solution.success),
            "bounded_solver_status": int(solution.status),
            "base_objective_linearized": base_objective,
            "singular_values": "|".join(f"{value:.8g}" for value in singular),
            **{f"step_{parameter}_nominal_units": float(dx[index]) for index, parameter in enumerate(PARAMS)},
        })

    pd.DataFrame(jacobian_rows).to_csv(args.output_dir / "signed_jacobian.csv", index=False)
    pd.DataFrame(proposal_rows).to_csv(args.output_dir / "proposed_parameters.csv", index=False)
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(args.output_dir / "diagnostics.csv", index=False)
    print(diagnostics.to_string(index=False))
    print(pd.DataFrame(proposal_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
