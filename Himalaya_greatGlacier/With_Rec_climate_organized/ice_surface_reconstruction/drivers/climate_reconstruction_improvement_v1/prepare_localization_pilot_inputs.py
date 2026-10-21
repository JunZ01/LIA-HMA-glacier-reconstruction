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


def _decode_char(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset).ravel(order="F")
    return "".join(chr(int(value)) for value in values if int(value) != 0)


def _state_coordinates() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_lat = np.linspace(-89.0, 89.0, 90)
    all_lon = np.linspace(-179.0, 179.0, 180)
    lat = all_lat[(all_lat >= 24.0) & (all_lat <= 51.0)]
    lon = all_lon[(all_lon >= 62.0) & (all_lon <= 110.0)]
    state_lat = np.tile(lat, len(lon))
    state_lon = np.repeat(lon, len(lat))
    return lat, lon, state_lat, state_lon


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = _project_root()
    config_default = Path(__file__).with_name("experiment_config.json")
    output_default = (
        root
        / "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_localization_pilot_v1/inputs"
    )
    parser.add_argument("--config", type=Path, default=config_default)
    parser.add_argument("--output-dir", type=Path, default=output_default)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    archive = Path(config["source_paths"]["archive_root"])
    phyda = Path(config["source_paths"]["phyda_root"])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    replay_dir = (
        root
        / "validation_data/results/paleo_surface_validation_v0/"
        "enkf_reconstruction_source_recovery_audit_v2/"
        "replay_proxy6_enkf_exp1_matlab_v1"
    )
    replay_path = replay_dir / "replay_input.mat"
    prior_path = replay_dir / "matlab_prior_336x848.npy"
    proxy_path = archive / "python_data" / "proxy_data_6.mat"
    precipitation_path = (
        archive
        / "save_result/pr/prxy6/HMA/EnKF/psm7_clb1940-2000/data/Xa_m_EnKF.mat"
    )

    replay = scipy.io.loadmat(replay_path)
    sample_indices = np.asarray(replay["sample_indices"], dtype=int).ravel() - 1
    prior = np.load(prior_path)
    if prior.shape != (336, 848):
        raise ValueError(f"unexpected-prior-shape:{prior.shape}")
    background = prior[:, sample_indices]

    hxb = np.asarray(replay["HXb0"], dtype=float)
    observations = np.asarray(replay["y0"], dtype=float)
    residuals = np.asarray(replay["R0v_filled"], dtype=float)
    target_mean = np.asarray(replay["target_mean"], dtype=float)
    if hxb.shape != (202, 500) or observations.shape != (202, 1001):
        raise ValueError("unexpected-replay-input-shape")

    locations = np.full((202, 2), np.nan)
    coefficients = np.full((202, 3), np.nan)
    bivariate = np.zeros(202, dtype=bool)
    with h5py.File(proxy_path, "r") as handle:
        group = handle["proxy_data"]
        references = np.asarray(group["full_proxy"]).reshape(-1, order="F")
        train_table = np.asarray(group["temp_proxy_indices"], dtype=int)
        if train_table.shape[0] != 10 and train_table.shape[1] == 10:
            train_table = train_table.T
        repeat = int(config["scope"]["holdout_repeat"]) - 1
        train_indices = train_table[repeat]
        train_indices = train_indices[train_indices > 0] - 1
        for index, reference in enumerate(references):
            proxy = handle[reference]
            locations[index] = np.asarray(proxy["location"]).reshape(-1, order="F")[:2]
            values = np.asarray(proxy["b"]).reshape(-1, order="F")
            coefficients[index, : min(3, values.size)] = values[:3]
            bivariate[index] = _decode_char(proxy["psm_mthd"]).lower() == "bivariate"

    all_indices = np.arange(202, dtype=int)
    holdout_indices = np.setdiff1d(all_indices, train_indices, assume_unique=False)
    if train_indices.size != 152 or holdout_indices.size != 50:
        raise ValueError(
            f"unexpected-split:{train_indices.size}-train:{holdout_indices.size}-holdout"
        )

    precipitation = np.asarray(
        scipy.io.loadmat(precipitation_path, variable_names=["Xa_m"])["Xa_m"],
        dtype=float,
    )
    if precipitation.shape != (336, 1001):
        raise ValueError(f"unexpected-precipitation-shape:{precipitation.shape}")

    grid_lat, grid_lon, state_lat, state_lon = _state_coordinates()
    proxy_state_index = np.empty(202, dtype=int)
    for index, (lat, lon) in enumerate(locations):
        lat_index = int(np.argmin(np.abs(grid_lat - lat)))
        lon_index = int(np.argmin(np.abs(grid_lon - lon)))
        proxy_state_index[index] = lon_index * len(grid_lat) + lat_index

    output_path = output / "localization_pilot_inputs.npz"
    np.savez_compressed(
        output_path,
        background=background,
        hxb=hxb,
        observations=observations,
        residuals=residuals,
        target_mean=target_mean,
        train_indices=train_indices,
        holdout_indices=holdout_indices,
        proxy_locations=locations,
        proxy_coefficients=coefficients,
        proxy_bivariate=bivariate,
        proxy_state_index=proxy_state_index,
        precipitation=precipitation,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        state_lat=state_lat,
        state_lon=state_lon,
        years=np.arange(1000, 2001, dtype=int),
    )

    source_paths = [
        replay_path,
        prior_path,
        proxy_path,
        precipitation_path,
        phyda / "M_update.m",
        phyda / "M_update_localized.m",
        phyda / "local_metric_maps/localization_radius_results.csv",
    ]
    manifest = {
        "experiment_id": config["experiment_id"],
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "source_mode": "read-only",
        "source_files": [
            {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in source_paths
        ],
        "shapes": {
            "background": list(background.shape),
            "hxb": list(hxb.shape),
            "observations": list(observations.shape),
            "precipitation": list(precipitation.shape),
        },
        "split": {
            "repeat_1based": int(config["scope"]["holdout_repeat"]),
            "train_count": int(train_indices.size),
            "holdout_count": int(holdout_indices.size),
        },
    }
    (output / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
