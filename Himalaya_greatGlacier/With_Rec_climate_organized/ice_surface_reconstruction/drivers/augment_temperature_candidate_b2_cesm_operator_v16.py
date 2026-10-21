from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d

import build_temperature_candidate_b2_allowed_input_v1 as source


ROOT = source.ROOT
LOOP = source.LOOP
INPUT = LOOP / (
    "temperature_balanced_pooled_candidate_b2_allowed_input_v15/"
    "candidate_b2_allowed_raw_input_v1.npz"
)
STAGE_B = LOOP / (
    "temperature_multimodel_evolving_prior_stage_b_pseudoproxy_preregistration_v2/"
    "stage_b_pseudoproxy_input_v1.npz"
)
OUT = LOOP / "temperature_balanced_pooled_candidate_b2_allowed_input_v16"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    paths = {"v15_raw_input": INPUT, "stage_b": STAGE_B, "proxy6": source.PROXY[6], "proxy9": source.PROXY[9]}
    before = {name: sha256(path) for name, path in paths.items()}
    with np.load(INPUT, allow_pickle=False) as raw:
        arrays = {name: raw[name] for name in raw.files}
    with np.load(STAGE_B, allow_pickle=False) as stage:
        stage_arrays = {name: stage[name] for name in stage.files}
    years = np.asarray(arrays["years"], dtype=int)
    if not np.array_equal(years, np.r_[np.arange(1500, 1921), np.arange(1940, 2001)]):
        raise RuntimeError("allowed-year-axis")
    allowed_indices = years - 1001
    pool_indices = np.arange(499, 848, dtype=int)
    evaluation = np.arange(15, 334, dtype=int)
    closure_rows = []
    for network in (6, 9):
        wanted = np.asarray(arrays[f"proxy{network}_record_indices_0based"], dtype=int)
        operators = []
        maximum_formula_error = 0.0
        with h5py.File(source.PROXY[network], "r") as handle:
            refs = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
            for record_index in wanted:
                proxy = handle[refs[record_index]]
                b = np.asarray(proxy["b"], dtype=float).reshape(-1, order="F")
                mt = np.asarray(proxy["M_t"], dtype=float).reshape(-1, order="F")
                mp = np.asarray(proxy["M_p"], dtype=float).reshape(-1, order="F")
                hxb = np.asarray(proxy["HXb"], dtype=float).reshape(-1, order="F")
                if min(len(mt), len(mp), len(hxb)) <= int(allowed_indices.max()):
                    raise RuntimeError(f"short-operator:{network}:{record_index}")
                rebuilt = b[0] + b[1] * mt[allowed_indices] + b[2] * mp[allowed_indices]
                maximum_formula_error = max(
                    maximum_formula_error,
                    float(np.max(np.abs(rebuilt - hxb[allowed_indices]))),
                )
                operators.append(hxb[allowed_indices])
        direct = np.asarray(operators, dtype=float).T
        arrays[f"proxy{network}_cesm_archived_operator"] = direct.astype(np.float32)
        pool = direct[: len(pool_indices)]
        anomaly = pool - pool.mean(axis=0, keepdims=True)
        rebuilt_stage = uniform_filter1d(anomaly, size=31, axis=0, mode="reflect")[evaluation]
        archived_stage = np.asarray(
            stage_arrays[f"proxy{network}_prior_operator_forced_mean"][0], dtype=float
        )
        stage_closure = float(np.max(np.abs(rebuilt_stage - archived_stage)))
        closure_rows.append(
            {
                "network": network,
                "record_count": len(wanted),
                "maximum_hxb_formula_error": maximum_formula_error,
                "maximum_stage_b_cesm_operator_closure": stage_closure,
            }
        )
        if maximum_formula_error > 1e-12 or stage_closure > 1e-12:
            raise RuntimeError(f"cesm-archived-operator-closure:{network}:{maximum_formula_error}:{stage_closure}")
    np.savez_compressed(OUT / "candidate_b2_allowed_raw_input_v2.npz", **arrays)
    after = {name: sha256(path) for name, path in paths.items()}
    if before != after:
        raise RuntimeError("source-hash-changed")
    summary = {
        "experiment": OUT.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "parent_raw_input": str(INPUT),
        "year_count": len(years),
        "cesm_operator_source": "archived proxy MAT HXb, matching Stage B",
        "closure": closure_rows,
        "source_hashes_unchanged": True,
        "sealed_1921_1939_values_used": False,
        "real_proxy_observations_read": False,
        "stage_c_enkf_run": False,
        "production_modified": False,
        "oggm_run": False,
        "next_action": "run_candidate_b2_allowed_input_preflight_v2",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUT / "source_manifest.json").write_text(json.dumps(before, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
