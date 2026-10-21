"""Prepare an independent MATLAB replay input for proxy9 / EnKF / experiment 1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import scipy.io


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_proxy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    years = np.arange(1000, 2001)
    with h5py.File(path, "r") as archive:
        references = np.asarray(archive["proxy_data/full_proxy"]).reshape(-1, order="F")
        hxb = np.empty((len(references), 848), dtype=float)
        observations = np.full((len(references), len(years)), np.nan)
        for index, reference in enumerate(references):
            proxy = archive[reference]
            hxb[index] = np.asarray(proxy["HXb"], dtype=float).reshape(-1, order="F")
            ages = np.asarray(proxy["age_yng"], dtype=float).reshape(-1, order="F")
            values = np.asarray(proxy["data"], dtype=float).reshape(-1, order="F")
            by_year = {int(age): values[row] for row, age in enumerate(ages) if np.isfinite(age)}
            observations[index] = [by_year.get(int(year), np.nan) for year in years]
    return hxb, observations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment_config.json"))
    args = parser.parse_args()
    root = _project_root()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    archive_root = Path(config["archive_root"])
    output = root / config["result_root"]
    output.mkdir(parents=True, exist_ok=True)
    indices_path = archive_root / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat"
    proxy_path = archive_root / "python_data/proxy_data_9.mat"
    posterior_path = archive_root / "fixed_proxy_experiments_69/tas/results/proxy9/EnKF_results.mat"
    prior_path = root / config["prior_path"]
    regem_path = root / config["regem_output"]

    with h5py.File(indices_path, "r") as archive:
        sample_indices = np.asarray(archive["all_Xb_indices"], dtype=int)[0] - 1
    hxb, observations = _load_proxy(proxy_path)
    residuals = np.asarray(scipy.io.loadmat(regem_path)["R0v_filled"], dtype=float)
    prior = np.load(prior_path)
    if prior.shape != (336, 848) or hxb.shape != (297, 848) or observations.shape != (297, 1001) or residuals.shape != (297, 61):
        raise ValueError("unexpected-proxy9-replay-input-shape")
    with h5py.File(posterior_path, "r") as archive:
        target_mean = np.asarray(archive["Xa_m_all"][0], dtype=float).T
        target_sigma = np.asarray(archive["Xa_sigma_all"][0], dtype=float).T
        target_prct = np.transpose(np.asarray(archive["Xa_prct_all"][0], dtype=float), (2, 1, 0))
    input_path = output / "replay_input.mat"
    scipy.io.savemat(input_path, {
        "sample_indices": (sample_indices + 1).astype(float),
        "HXb0": hxb[:, sample_indices],
        "y0": observations,
        "R0v_filled": residuals,
        "target_mean": target_mean,
        "target_sigma": target_sigma,
        "target_prct": target_prct,
        "preflight_columns": np.array([1, 51, 101, 151, 201, 251, 301, 351, 401, 450], dtype=float),
    }, do_compression=False)
    manifest = {
        "experiment_id": config["experiment_id"],
        "source_mode": "read-only",
        "sources": [{"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size} for path in (indices_path, proxy_path, posterior_path, prior_path, regem_path)],
        "shapes": {"sample_indices": list(sample_indices.shape), "HXb0": list(hxb[:, sample_indices].shape), "y0": list(observations.shape), "R0v_filled": list(residuals.shape), "target_mean": list(target_mean.shape)},
        "input_path": str(input_path),
        "input_sha256": _sha256(input_path),
    }
    (output / "input_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(manifest["shapes"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
