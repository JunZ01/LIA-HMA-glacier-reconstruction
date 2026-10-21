from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import scipy.io


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "enkf_replay_v1", HERE / "replay_enkf_fixed_experiment_v1.py"
)
v1 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(v1)

OUTPUT_DIR = (
    v1.PROJECT_ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "enkf_reconstruction_source_recovery_audit_v2"
    / "replay_proxy6_enkf_exp1_matlab_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    indices_path = (
        v1.ARCHIVE_ROOT
        / "fixed_proxy_experiments_69"
        / "tas"
        / "indices"
        / "all_Xb_indices.mat"
    )
    proxy_path = v1.ARCHIVE_ROOT / "python_data" / "proxy_data_6.mat"
    result_path = (
        v1.ARCHIVE_ROOT
        / "fixed_proxy_experiments_69"
        / "tas"
        / "results"
        / "proxy6"
        / "EnKF_results.mat"
    )
    regem_path = (
        v1.PROJECT_ROOT
        / "validation_data"
        / "results"
        / "paleo_surface_validation_v0"
        / "enkf_reconstruction_source_recovery_audit_v2"
        / "regem_proxy6_exact_v1"
        / "regem_output.mat"
    )

    sample_indices = v1._load_indices(indices_path)
    hxb, residuals, observations = v1._load_proxy6(proxy_path)
    regem = scipy.io.loadmat(regem_path)["R0v_filled"]
    if regem.shape != residuals.shape or np.isnan(regem).any():
        raise ValueError("Invalid exact RegEM matrix")

    with h5py.File(result_path, "r") as archived:
        target_mean = np.asarray(archived["Xa_m_all"][0, :, :], dtype=np.float64).T
        target_sigma = np.asarray(archived["Xa_sigma_all"][0, :, :], dtype=np.float64).T
        target_prct = np.transpose(
            np.asarray(archived["Xa_prct_all"][0, :, :, :], dtype=np.float64),
            (2, 1, 0),
        )

    preflight_years = np.array(
        [1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1449],
        dtype=np.float64,
    )
    output_path = OUTPUT_DIR / "replay_input.mat"
    scipy.io.savemat(
        output_path,
        {
            "sample_indices": (sample_indices + 1).astype(np.float64),
            "HXb0": hxb[:, sample_indices],
            "y0": observations,
            "R0v_filled": regem,
            "target_mean": target_mean,
            "target_sigma": target_sigma,
            "target_prct": target_prct,
            "preflight_columns": preflight_years - 999.0,
        },
        do_compression=False,
    )
    summary = {
        "input_path": str(output_path),
        "input_sha256": _sha256(output_path),
        "sample_count": int(sample_indices.size),
        "HXb0_shape": list(hxb[:, sample_indices].shape),
        "y0_shape": list(observations.shape),
        "R0v_filled_shape": list(regem.shape),
        "target_mean_shape": list(target_mean.shape),
        "target_sigma_shape": list(target_sigma.shape),
        "target_prct_shape": list(target_prct.shape),
        "preflight_years": preflight_years.astype(int).tolist(),
    }
    (OUTPUT_DIR / "input_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
