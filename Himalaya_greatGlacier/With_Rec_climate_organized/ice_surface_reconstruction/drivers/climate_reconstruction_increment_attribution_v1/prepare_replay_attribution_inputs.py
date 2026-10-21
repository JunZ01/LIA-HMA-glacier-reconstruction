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
    return "".join(chr(int(value)) for value in values if int(value) != 0).strip()


def _state_coordinates() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_lat = np.linspace(-89.0, 89.0, 90)
    all_lon = np.linspace(-179.0, 179.0, 180)
    grid_lat = all_lat[(all_lat >= 24.0) & (all_lat <= 51.0)]
    grid_lon = all_lon[(all_lon >= 62.0) & (all_lon <= 110.0)]
    state_lat = np.tile(grid_lat, len(grid_lon))
    state_lon = np.repeat(grid_lon, len(grid_lat))
    return grid_lat, grid_lon, state_lat, state_lon


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = _project_root()
    config_default = Path(__file__).with_name("experiment_config.json")
    output_default = (
        root
        / "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_increment_attribution_v1/inputs"
    )
    parser.add_argument("--config", type=Path, default=config_default)
    parser.add_argument("--output-dir", type=Path, default=output_default)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    replay_path = root / config["source_paths"]["replay_input"]
    prior_path = root / config["source_paths"]["replay_prior"]
    proxy_path = (
        Path(config["source_paths"]["archive_root"]) / "python_data" / "proxy_data_6.mat"
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    replay = scipy.io.loadmat(replay_path)
    sample_indices = np.asarray(replay["sample_indices"], dtype=int).ravel() - 1
    prior = np.load(prior_path)
    background = prior[:, sample_indices]
    hxb = np.asarray(replay["HXb0"], dtype=float)
    observations = np.asarray(replay["y0"], dtype=float)
    residuals = np.asarray(replay["R0v_filled"], dtype=float)
    target_mean = np.asarray(replay["target_mean"], dtype=float)
    if background.shape != (336, 500):
        raise ValueError(f"unexpected-background-shape:{background.shape}")
    if hxb.shape != (202, 500) or observations.shape != (202, 1001):
        raise ValueError("unexpected-replay-input-shape")

    locations = np.full((202, 2), np.nan)
    coefficients = np.full((202, 3), np.nan)
    proxy_types = np.empty(202, dtype="U32")
    psm_methods = np.empty(202, dtype="U32")
    with h5py.File(proxy_path, "r") as handle:
        references = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
        if references.size != 202:
            raise ValueError(f"unexpected-proxy-count:{references.size}")
        for index, reference in enumerate(references):
            proxy = handle[reference]
            locations[index] = np.asarray(proxy["location"]).reshape(-1, order="F")[:2]
            values = np.asarray(proxy["b"]).reshape(-1, order="F")
            coefficients[index, : min(values.size, 3)] = values[:3]
            proxy_types[index] = _decode_char(proxy["ptype"])
            psm_methods[index] = _decode_char(proxy["psm_mthd"])

    grid_lat, grid_lon, state_lat, state_lon = _state_coordinates()
    output_path = output / "replay_attribution_inputs.npz"
    np.savez_compressed(
        output_path,
        years=np.arange(1000, 2001, dtype=int),
        background=background,
        hxb=hxb,
        observations=observations,
        residuals=residuals,
        target_mean=target_mean,
        proxy_locations=locations,
        proxy_coefficients=coefficients,
        proxy_types=proxy_types,
        psm_methods=psm_methods,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        state_lat=state_lat,
        state_lon=state_lon,
    )
    source_paths = [replay_path, prior_path, proxy_path]
    manifest = {
        "experiment_id": config["experiment_id"],
        "source_mode": "read-only",
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "source_files": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in source_paths
        ],
        "shapes": {
            "background": list(background.shape),
            "hxb": list(hxb.shape),
            "observations": list(observations.shape),
            "target_mean": list(target_mean.shape),
        },
    }
    (output / "replay_input_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
