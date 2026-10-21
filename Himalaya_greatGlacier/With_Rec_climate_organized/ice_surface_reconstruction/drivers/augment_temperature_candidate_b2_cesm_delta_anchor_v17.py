from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d

import build_temperature_candidate_b2_allowed_input_v1 as source


LOOP = source.LOOP
INPUT = LOOP / (
    "temperature_balanced_pooled_candidate_b2_allowed_input_v15/"
    "candidate_b2_allowed_raw_input_v1.npz"
)
STAGE_B = LOOP / (
    "temperature_multimodel_evolving_prior_stage_b_pseudoproxy_preregistration_v2/"
    "stage_b_pseudoproxy_input_v1.npz"
)
CONTRACT = LOOP / (
    "temperature_balanced_pooled_candidate_b2_cesm_operator_extension_preregistration_v1/"
    "CONTRACT.md"
)
OUT = LOOP / "temperature_balanced_pooled_candidate_b2_allowed_input_v17"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    paths = {
        "v15_raw_input": INPUT,
        "stage_b": STAGE_B,
        "extension_contract": CONTRACT,
        "proxy6": source.PROXY[6],
        "proxy9": source.PROXY[9],
    }
    before = {name: sha256(path) for name, path in paths.items()}
    with np.load(INPUT, allow_pickle=False) as raw:
        arrays = {name: raw[name] for name in raw.files}
    stage = np.load(STAGE_B, allow_pickle=False)
    years = np.asarray(arrays["years"], dtype=int)
    pool_mask = (years >= 1500) & (years <= 1848)
    pool_years = years[pool_mask]
    if not np.array_equal(pool_years, np.arange(1500, 1849)):
        raise RuntimeError("pool-year-axis")
    anchor_index = int(np.flatnonzero(years == 1848)[0])
    next_index = int(np.flatnonzero(years == 1849)[0])
    state_count = int(np.asarray(arrays["state_count"]))
    closure_rows = []
    for network in (6, 9):
        wanted = np.asarray(arrays[f"proxy{network}_record_indices_0based"], dtype=int)
        b = np.asarray(arrays[f"proxy{network}_psm_b0_b1_b2"], dtype=float)
        point_index = np.asarray(arrays[f"proxy{network}_unique_point_indices"], dtype=int) + state_count
        tas = np.asarray(arrays["cesm_tas_anomaly_1961_1990_degC"], dtype=float)[:, point_index]
        pr = np.asarray(arrays["cesm_pr_mm_day"], dtype=float)[:, point_index]
        annual = b[:, 0][None, :] + b[:, 1][None, :] * tas + b[:, 2][None, :] * pr
        archived_pool = []
        maximum_formula_error = 0.0
        with h5py.File(source.PROXY[network], "r") as handle:
            refs = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
            for record_index in wanted:
                proxy = handle[refs[record_index]]
                proxy_b = np.asarray(proxy["b"], dtype=float).reshape(-1, order="F")
                mt = np.asarray(proxy["M_t"], dtype=float).reshape(-1, order="F")[499:848]
                mp = np.asarray(proxy["M_p"], dtype=float).reshape(-1, order="F")[499:848]
                hxb = np.asarray(proxy["HXb"], dtype=float).reshape(-1, order="F")[499:848]
                maximum_formula_error = max(
                    maximum_formula_error,
                    float(np.max(np.abs(proxy_b[0] + proxy_b[1] * mt + proxy_b[2] * mp - hxb))),
                )
                archived_pool.append(hxb)
        archived_pool = np.asarray(archived_pool, dtype=float).T
        anchored = annual + (archived_pool[-1] - annual[anchor_index])[None, :]
        anchored[pool_mask] = archived_pool
        anchor_error = float(np.max(np.abs(anchored[anchor_index] - archived_pool[-1])))
        change_error = float(
            np.max(
                np.abs(
                    (anchored[next_index] - anchored[anchor_index])
                    - (annual[next_index] - annual[anchor_index])
                )
            )
        )
        anomaly = archived_pool - archived_pool.mean(axis=0, keepdims=True)
        rebuilt_stage = uniform_filter1d(anomaly, size=31, axis=0, mode="reflect")[15:334]
        archived_stage = np.asarray(
            stage[f"proxy{network}_prior_operator_forced_mean"][0], dtype=float
        )
        stage_closure = float(np.max(np.abs(rebuilt_stage - archived_stage)))
        if maximum_formula_error > 1e-12 or anchor_error > 1e-12 or change_error > 1e-12 or stage_closure > 2e-5:
            raise RuntimeError(
                f"delta-anchor-gate:{network}:{maximum_formula_error}:{anchor_error}:{change_error}:{stage_closure}"
            )
        arrays[f"proxy{network}_cesm_delta_anchored_operator"] = anchored.astype(np.float32)
        closure_rows.append(
            {
                "network": network,
                "record_count": len(wanted),
                "maximum_archived_hxb_formula_error": maximum_formula_error,
                "maximum_stage_b_cesm_operator_closure": stage_closure,
                "maximum_1848_anchor_error": anchor_error,
                "maximum_1848_1849_change_error": change_error,
            }
        )
    np.savez_compressed(OUT / "candidate_b2_allowed_raw_input_v3.npz", **arrays)
    after = {name: sha256(path) for name, path in paths.items()}
    if before != after:
        raise RuntimeError("source-hash-changed")
    summary = {
        "experiment": OUT.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "parent_raw_input": str(INPUT),
        "method": "archived CESM HXb through 1848; annual-operator delta anchored at 1848 afterward",
        "closure": closure_rows,
        "source_hashes_unchanged": True,
        "sealed_1921_1939_values_used": False,
        "real_proxy_observations_read": False,
        "stage_c_enkf_run": False,
        "production_modified": False,
        "oggm_run": False,
        "next_action": "run_candidate_b2_allowed_input_preflight_v3",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUT / "source_manifest.json").write_text(json.dumps(before, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
