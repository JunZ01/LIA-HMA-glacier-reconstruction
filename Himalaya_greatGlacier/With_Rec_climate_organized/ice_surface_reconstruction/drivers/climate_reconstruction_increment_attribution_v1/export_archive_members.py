from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr


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


def _state_coordinates() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_lat = np.linspace(-89.0, 89.0, 90)
    all_lon = np.linspace(-179.0, 179.0, 180)
    grid_lat = all_lat[(all_lat >= 24.0) & (all_lat <= 51.0)]
    grid_lon = all_lon[(all_lon >= 62.0) & (all_lon <= 110.0)]
    state_lat = np.tile(grid_lat, len(grid_lon))
    state_lon = np.repeat(grid_lon, len(grid_lat))
    return grid_lat, grid_lon, state_lat, state_lon


def _paired_points(
    field: xr.DataArray,
    state_lat: np.ndarray,
    state_lon: np.ndarray,
) -> np.ndarray:
    point_lat = xr.DataArray(state_lat, dims="point")
    point_lon = xr.DataArray(state_lon, dims="point")
    paired = field.interp(lat=point_lat, lon=point_lon)
    return np.asarray(paired.transpose("point", "time").values, dtype=float)


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
    source = Path(config["source_paths"]["instrumental_netcdf"])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    grid_lat, grid_lon, state_lat, state_lon = _state_coordinates()
    target_lat = float(config["scope"]["target_lat"])
    target_lon = float(config["scope"]["target_lon"])
    member_names = list(config["member_variables"])
    member_variables = list(config["member_variables"].values())
    product_names = list(config["observation_variables"])
    product_variables = list(config["observation_variables"].values())

    member_fields = []
    member_targets = []
    observation_fields = []
    observation_targets = []
    with xr.open_dataset(source) as dataset:
        years = np.asarray(dataset.time.values, dtype=int)
        native_stack = xr.concat([dataset[name] for name in member_variables], dim="member")
        ensemble_native = dataset[config["ensemble_variable"]]
        native_difference = native_stack.mean("member") - ensemble_native
        native_closure_max = float(np.nanmax(np.abs(native_difference.values)))
        native_closure_rmse = float(
            np.sqrt(np.nanmean(np.asarray(native_difference.values, dtype=float) ** 2))
        )
        for variable in member_variables:
            member_fields.append(_paired_points(dataset[variable], state_lat, state_lon))
            member_targets.append(
                np.asarray(
                    dataset[variable].interp(lat=target_lat, lon=target_lon).values,
                    dtype=float,
                )
            )
        ensemble_field = _paired_points(ensemble_native, state_lat, state_lon)
        ensemble_target = np.asarray(
            ensemble_native.interp(lat=target_lat, lon=target_lon).values,
            dtype=float,
        )
        reference = dataset[config["reference_variable"]]
        reference_field = _paired_points(reference, state_lat, state_lon)
        reference_target = np.asarray(
            reference.interp(lat=target_lat, lon=target_lon).values,
            dtype=float,
        )
        for variable in product_variables:
            observation_fields.append(
                _paired_points(dataset[variable], state_lat, state_lon)
            )
            observation_targets.append(
                np.asarray(
                    dataset[variable].interp(lat=target_lat, lon=target_lon).values,
                    dtype=float,
                )
            )

    member_fields_array = np.stack(member_fields)
    member_targets_array = np.stack(member_targets)
    observation_fields_array = np.stack(observation_fields)
    observation_targets_array = np.stack(observation_targets)
    state_difference = np.mean(member_fields_array, axis=0) - ensemble_field
    target_difference = np.mean(member_targets_array, axis=0) - ensemble_target
    output_path = output / "archive_members_and_instruments.npz"
    np.savez_compressed(
        output_path,
        years=years,
        member_names=np.asarray(member_names, dtype="U32"),
        member_variables=np.asarray(member_variables, dtype="U64"),
        member_fields=member_fields_array,
        member_targets=member_targets_array,
        ensemble_field=ensemble_field,
        ensemble_target=ensemble_target,
        reference_field=reference_field,
        reference_target=reference_target,
        product_names=np.asarray(product_names, dtype="U32"),
        product_variables=np.asarray(product_variables, dtype="U64"),
        observation_fields=observation_fields_array,
        observation_targets=observation_targets_array,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        state_lat=state_lat,
        state_lon=state_lon,
    )
    manifest = {
        "experiment_id": config["experiment_id"],
        "source_mode": "read-only",
        "source": str(source),
        "source_sha256": _sha256(source),
        "source_size_bytes": source.stat().st_size,
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "shapes": {
            "member_fields": list(member_fields_array.shape),
            "observation_fields": list(observation_fields_array.shape),
        },
        "closure": {
            "native_max_abs_c": native_closure_max,
            "native_rmse_c": native_closure_rmse,
            "state_grid_max_abs_c": float(np.nanmax(np.abs(state_difference))),
            "target_max_abs_c": float(np.nanmax(np.abs(target_difference))),
        },
    }
    (output / "archive_export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
