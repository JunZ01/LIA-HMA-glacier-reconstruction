"""Locate the first-step CFL limiter in system 14's native 24 m 2-D state."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[4]
STATE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline_planform2d_source_native_padded_static_v1/system_14/source_native_padded_initial_state.npz"
ZERO = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline_planform2d_source_native_zero_mb_v1/system_14/summary.json"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline_planform2d_source_native_cfl_diagnostic_v1/system_14"
GLEN_A = 6.642407336686561e-23


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    state = np.load(STATE)
    bed = np.asarray(state["bed_elevation_m"], dtype=float)
    h = np.asarray(state["thickness_m"], dtype=float)
    x = np.asarray(state["x_m"], dtype=float)
    y = np.asarray(state["y_m"], dtype=float)
    dx = float(state["dx_m"])
    row0 = int(state["source_row_offset"])
    col0 = int(state["source_col_offset"])

    from oggm import cfg
    from oggm.core.massbalance import ScalarMassBalance
    from oggm.core.sia2d import Upstream2D

    cfg.initialize(logging_level="WARNING")
    model = Upstream2D(
        bed,
        init_ice_thick=h,
        dx=dx,
        mb_model=ScalarMassBalance(0.0),
        glen_a=GLEN_A,
        ice_thick_filter=None,
    )
    S = model.surface_h
    H = model.ice_thick
    N = model.glen_n
    kp = np.r_[np.arange(1, model.ny), model.ny - 1]
    km = np.r_[0, np.arange(model.ny - 1)]
    lp = np.r_[np.arange(1, model.nx), model.nx - 1]
    lm = np.r_[0, np.arange(model.nx - 1)]
    K, L = np.meshgrid(np.arange(model.ny), np.arange(model.nx), indexing="ij")

    def take(rows, cols):
        return S[rows, cols]

    s0 = S[K, L]
    slp, slm = S[K, lp[L]], S[K, lm[L]]
    skp, skm = S[kp[K], L], S[km[K], L]
    h0 = H[K, L]
    hlp, hlm = H[K, lp[L]], H[K, lm[L]]
    hkp, hkm = H[kp[K], L], H[km[K], L]

    h_l_up = 0.5 * (hlp + h0)
    h_l_dn = 0.5 * (h0 + hlm)
    h_l_us_up = np.where(slp > s0, hlp, h0)
    h_l_us_dn = np.where(s0 > slm, h0, hlm)
    sdiff = skp - skm
    slpdiff = slp - s0
    slmdiff = s0 - slm
    sl_grad_up = (((sdiff + S[kp[K], lp[L]] - S[km[K], lp[L]]) ** 2 / (4 * dx) ** 2) + slpdiff ** 2 / dx ** 2) ** ((N - 1) / 2)
    sl_grad_dn = (((sdiff + S[kp[K], lm[L]] - S[km[K], lm[L]]) ** 2 / (4 * dx) ** 2) + slmdiff ** 2 / dx ** 2) ** ((N - 1) / 2)
    d_l_up = model.gamma * h_l_up ** (N + 1) * h_l_us_up * sl_grad_up
    d_l_dn = model.gamma * h_l_dn ** (N + 1) * h_l_us_dn * sl_grad_dn

    h_k_up = 0.5 * (hkp + h0)
    h_k_dn = 0.5 * (h0 + hkm)
    h_k_us_up = np.where(skp > s0, hkp, h0)
    h_k_us_dn = np.where(s0 > skm, h0, hkm)
    sdiff = slp - slm
    skpdiff = skp - s0
    skmdiff = s0 - skm
    sk_grad_up = (((sdiff + S[kp[K], lp[L]] - S[kp[K], lm[L]]) ** 2 / (4 * dx) ** 2) + skpdiff ** 2 / dx ** 2) ** ((N - 1) / 2)
    sk_grad_dn = (((sdiff + S[km[K], lp[L]] - S[km[K], lm[L]]) ** 2 / (4 * dx) ** 2) + skmdiff ** 2 / dx ** 2) ** ((N - 1) / 2)
    d_k_up = model.gamma * h_k_up ** (N + 1) * h_k_us_up * sk_grad_up
    d_k_dn = model.gamma * h_k_dn ** (N + 1) * h_k_us_dn * sk_grad_dn

    arrays = {"east": d_l_up, "west": d_l_dn, "south": d_k_up, "north": d_k_dn}
    direction = max(arrays, key=lambda key: float(np.max(np.abs(arrays[key]))))
    diffusivity = arrays[direction]
    row, col = np.unravel_index(np.argmax(np.abs(diffusivity)), diffusivity.shape)
    neighbour = {
        "east": (row, min(col + 1, model.nx - 1)),
        "west": (row, max(col - 1, 0)),
        "south": (min(row + 1, model.ny - 1), col),
        "north": (max(row - 1, 0), col),
    }[direction]
    nr, nc = neighbour
    max_d = float(abs(diffusivity[row, col]))
    dt_cfl = float(model.cfl * dx * dx / max_d)
    source_shape = (417, 649)

    def inside_source(r, c):
        return row0 <= r < row0 + source_shape[0] and col0 <= c < col0 + source_shape[1]

    rows = []
    for rr in range(max(0, row - 3), min(model.ny, row + 4)):
        for cc in range(max(0, col - 3), min(model.nx, col + 4)):
            rows.append(
                {
                    "row": rr,
                    "col": cc,
                    "x_m": x[cc],
                    "y_m": y[rr],
                    "bed_m": bed[rr, cc],
                    "thickness_m": h[rr, cc],
                    "surface_m": S[rr, cc],
                    "positive_ice": h[rr, cc] > 1e-6,
                    "inside_source_grid": inside_source(rr, cc),
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUT / "limiter_window.csv", index=False)
    summary = {
        "schema": "lia-191-system14-source-native-planform2d-cfl-diagnostic-v1",
        "status": "diagnostic-complete",
        "system_id": 14,
        "limiting_direction": direction,
        "limiting_cell": {"row": int(row), "col": int(col), "x_m": float(x[col]), "y_m": float(y[row])},
        "neighbour_cell": {"row": int(nr), "col": int(nc), "x_m": float(x[nc]), "y_m": float(y[nr])},
        "maximum_diffusivity_m2_s": max_d,
        "reconstructed_cfl_timestep_seconds": dt_cfl,
        "reported_zero_mb_timestep_seconds": json.loads(ZERO.read_text(encoding="utf-8"))["checkpoint_0p1y"]["runtime_error"],
        "limiting_cell_inside_source_grid": inside_source(row, col),
        "neighbour_inside_source_grid": inside_source(nr, nc),
        "limiting_cell_positive_ice": bool(h[row, col] > 1e-6),
        "neighbour_positive_ice": bool(h[nr, nc] > 1e-6),
        "limiting_bed_m": float(bed[row, col]),
        "neighbour_bed_m": float(bed[nr, nc]),
        "bed_step_m": float(bed[nr, nc] - bed[row, col]),
        "limiting_thickness_m": float(h[row, col]),
        "neighbour_thickness_m": float(h[nr, nc]),
        "thickness_step_m": float(h[nr, nc] - h[row, col]),
        "limiting_surface_m": float(S[row, col]),
        "neighbour_surface_m": float(S[nr, nc]),
        "surface_step_m": float(S[nr, nc] - S[row, col]),
        "source_padding_seam_is_limiter": bool(not (inside_source(row, col) and inside_source(nr, nc))),
        "oggm_run": False,
        "zero_mb_rerun": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
