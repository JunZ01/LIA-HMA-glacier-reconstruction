"""Prepare frozen inputs for proxy6 precipitation EnKF experiment-1 replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import scipy.io


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def archive_root() -> Path:
    archive_folder = "".join(chr(value) for value in (25991, 20214, 25972, 29702))
    paper_folder = "1 " + "".join(chr(value) for value in (35770, 25991))
    return Path(r"D:\Users\JUNZ\Desktop") / archive_folder / paper_folder / "work1" / "compared_result_all"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_proxy(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    years = np.arange(1000, 2001)
    with h5py.File(path, "r") as handle:
        group = handle["proxy_data"]
        references = np.asarray(group["full_proxy"]).reshape(-1, order="F")
        hxb = np.empty((len(references), 848), dtype=np.float64)
        residuals = np.empty((len(references), 61), dtype=np.float64)
        observations = np.full((len(references), years.size), np.nan)
        for index, reference in enumerate(references):
            proxy = handle[reference]
            hxb[index] = np.asarray(proxy["HXb"]).reshape(-1, order="F")
            residuals[index] = np.asarray(proxy["Rvec"]).reshape(-1, order="F")
            ages = np.asarray(proxy["age_yng"]).reshape(-1, order="F")
            data = np.asarray(proxy["data"]).reshape(-1, order="F")
            by_year = {
                int(age): float(data[item])
                for item, age in enumerate(ages)
                if np.isfinite(age) and np.isfinite(data[item])
            }
            observations[index] = [by_year.get(int(year), np.nan) for year in years]
    return hxb, residuals, observations


def main() -> int:
    root = project_root()
    archive = archive_root()
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_enkf_exp1_replay_v1"
    )
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "prior": root / (
            "validation_data/results/paleo_surface_validation_v0/"
            "climate_reconstruction_precipitation_prior_recovery_preflight_v1/"
            "precipitation_prior_336x848_mm_day.mat"
        ),
        "indices": archive / "fixed_proxy_experiments_69" / "pr" / "indices" / "all_Xb_indices.mat",
        "proxy6": archive / "python_data" / "proxy_data_6.mat",
        "archived_result": archive / "fixed_proxy_experiments_69" / "pr" / "results" / "proxy6" / "EnKF_results.mat",
        "regem": root / (
            "validation_data/results/paleo_surface_validation_v0/"
            "enkf_reconstruction_source_recovery_audit_v2/regem_proxy6_exact_v1/regem_output.mat"
        ),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))

    with h5py.File(paths["indices"], "r") as handle:
        sample_indices = np.asarray(handle["all_Xb_indices"], dtype=int)[0]
    if sample_indices.shape != (500,) or np.unique(sample_indices).size != 500:
        raise RuntimeError("invalid-experiment1-member-indices")

    hxb, residuals, observations = load_proxy(paths["proxy6"])
    regem = np.asarray(scipy.io.loadmat(paths["regem"])["R0v_filled"], dtype=float)
    if regem.shape != residuals.shape or not np.isfinite(regem).all():
        raise RuntimeError("invalid-regem-matrix")

    with h5py.File(paths["archived_result"], "r") as handle:
        target_mean = np.asarray(handle["Xa_m_all"][0], dtype=float).T
        target_sigma = np.asarray(handle["Xa_sigma_all"][0], dtype=float).T
        target_prct = np.transpose(np.asarray(handle["Xa_prct_all"][0], dtype=float), (2, 1, 0))
    if target_mean.shape != (336, 1001):
        raise RuntimeError(f"invalid-target-mean-shape:{target_mean.shape}")
    if target_sigma.shape != (336, 1001):
        raise RuntimeError(f"invalid-target-sigma-shape:{target_sigma.shape}")
    if target_prct.shape != (336, 3, 1001):
        raise RuntimeError(f"invalid-target-prct-shape:{target_prct.shape}")

    input_path = output / "replay_input.mat"
    scipy.io.savemat(
        input_path,
        {
            "sample_indices": sample_indices.astype(float),
            "HXb0": hxb[:, sample_indices - 1],
            "y0": observations,
            "R0v_filled": regem,
            "target_mean": target_mean,
            "target_sigma": target_sigma,
            "target_prct": target_prct,
        },
        do_compression=False,
    )
    summary = {
        "input_path": str(input_path),
        "input_sha256": sha256(input_path),
        "source_hashes": {name: sha256(path) for name, path in paths.items()},
        "sample_indices_shape": list(sample_indices.shape),
        "sample_indices_range": [int(sample_indices.min()), int(sample_indices.max())],
        "HXb0_shape": [int(hxb.shape[0]), 500],
        "observation_shape": list(observations.shape),
        "regem_shape": list(regem.shape),
        "target_mean_shape": list(target_mean.shape),
        "target_sigma_shape": list(target_sigma.shape),
        "target_prct_shape": list(target_prct.shape),
    }
    (output / "prepared_input_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

