from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr


OBSERVATION_VARIABLES = {
    "Berkeley Earth": "tas_obs_bea_05deg",
    "CRU TS": "tas_obs_cru_tmp_05deg",
    "GISTEMP": "tas_obs_GIST_05deg",
    "HadCRUT5": "tas_obs_had5_05deg",
    "NOAA": "tas_obs_NOAA_05deg",
}


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
    input_path = Path(config["source_paths"]["instrumental_netcdf"])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    prepared = np.load(output / "localization_pilot_inputs.npz")
    state_lat = prepared["state_lat"]
    state_lon = prepared["state_lon"]
    point_lat = xr.DataArray(state_lat, dims="point")
    point_lon = xr.DataArray(state_lon, dims="point")

    arrays = []
    target_arrays = []
    with xr.open_dataset(input_path) as dataset:
        years = np.asarray(dataset.time.values, dtype=int)
        for variable in OBSERVATION_VARIABLES.values():
            paired = dataset[variable].interp(lat=point_lat, lon=point_lon)
            arrays.append(np.asarray(paired.transpose("point", "time").values, dtype=float))
            target = dataset[variable].interp(lat=33.75, lon=75.75)
            target_arrays.append(np.asarray(target.values, dtype=float))

    observations = np.stack(arrays, axis=0)
    target_3584 = np.stack(target_arrays, axis=0)
    output_path = output / "instrumental_targets.npz"
    np.savez_compressed(
        output_path,
        observations=observations,
        target_3584=target_3584,
        product_names=np.asarray(list(OBSERVATION_VARIABLES), dtype="U32"),
        variable_names=np.asarray(list(OBSERVATION_VARIABLES.values()), dtype="U32"),
        years=years,
        state_lat=state_lat,
        state_lon=state_lon,
    )
    manifest = {
        "source_mode": "read-only",
        "input_netcdf": str(input_path),
        "input_sha256": _sha256(input_path),
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "observation_shape": list(observations.shape),
        "target_3584_shape": list(target_3584.shape),
        "products": list(OBSERVATION_VARIABLES),
    }
    (output / "instrumental_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
