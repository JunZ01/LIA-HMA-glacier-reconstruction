"""Prepare allowed-period proxy inputs for Round 03 operator-level OOF."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import scipy.io


EXPERIMENT_ID = "climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection"
YEARS = np.arange(1940, 2001, dtype=int)


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def state(path: Path) -> dict[str, object]:
    item = path.stat()
    return {"path": str(path), "size_bytes": int(item.st_size), "mtime_ns": int(item.st_mtime_ns)}


def decode_char(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset).reshape(-1, order="F")
    return "".join(chr(int(value)) for value in values if int(value) > 0)


def selected_values(dataset: h5py.Dataset, indices: np.ndarray, axis_length: int) -> np.ndarray:
    if dataset.shape[0] == axis_length:
        return np.asarray(dataset[indices, ...], dtype=float).reshape(-1, order="F")
    if dataset.shape[-1] == axis_length:
        return np.asarray(dataset[..., indices], dtype=float).reshape(-1, order="F")
    raise RuntimeError(f"proxy-data-axis-not-found:{dataset.shape}:{axis_length}")


def extract(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as archive:
        references = np.asarray(archive["proxy_data/full_proxy"]).reshape(-1, order="F")
        n = len(references)
        observations = np.full((len(YEARS), n), np.nan)
        locations = np.full((n, 2), np.nan)
        archived_hxb = np.full((n, 848), np.nan)
        archived_b = np.full((n, 3), np.nan)
        model_type = np.full(n, 2, dtype=np.int16)
        for column, reference in enumerate(references):
            proxy = archive[reference]
            ages = np.asarray(proxy["age_yng"], dtype=float).reshape(-1, order="F")
            rounded = np.rint(ages).astype(int)
            positions = np.asarray([int(np.flatnonzero(rounded == year)[0]) for year in YEARS], dtype=int)
            observations[:, column] = selected_values(proxy["data"], positions, len(ages))
            locations[column] = np.asarray(proxy["location"], dtype=float).reshape(-1, order="F")[:2]
            archived_hxb[column] = np.asarray(proxy["HXb"], dtype=float).reshape(-1, order="F")
            coefficients = np.asarray(proxy["b"], dtype=float).reshape(-1, order="F")
            archived_b[column, : min(3, len(coefficients))] = coefficients[:3]
            method = decode_char(proxy["psm_mthd"]).strip().lower()
            if method in {"uni", "univariate"} or len(coefficients) == 2:
                model_type[column] = 1
            elif method not in {"bi", "bivariate"} and len(coefficients) != 3:
                raise RuntimeError(f"unknown-psm-method:{column + 1}:{method}:{len(coefficients)}")
    return {
        "years": YEARS[:, None],
        "observations": observations,
        "locations": locations,
        "archived_hxb": archived_hxb,
        "archived_b": archived_b,
        "model_type": model_type[:, None],
    }


def main() -> int:
    project = root()
    output = project / "validation_data/results/paleo_surface_validation_v0" / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    config_path = project / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/climate_reconstruction_controlled_enkf_early20_v1/experiment_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    archive = Path(config["archive_root"])
    paths = {
        "proxy6": archive / config["sources"]["proxy6"],
        "proxy9": archive / config["sources"]["proxy9"],
        "lock": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_instrumental_baseline_v1/LOCKED_FINAL_PERIOD.json",
        "facts": project / "LIA_PROJECT_FACTS.md",
        "round02": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1/round_02_network_operator_repair/summary.json",
        "config": config_path,
    }
    if missing := [str(path) for path in paths.values() if not path.exists()]:
        raise FileNotFoundError("missing-round03-inputs:" + ";".join(missing))
    locked = json.loads(paths["lock"].read_text(encoding="utf-8"))
    if locked.get("status") != "locked_not_evaluated":
        raise RuntimeError("locked-period-contract-failed")
    if np.any((YEARS >= 1921) & (YEARS <= 1939)):
        raise RuntimeError("locked-period-requested")

    before = {name: state(path) for name, path in paths.items()}
    hashes = {name: digest(path) for name, path in paths.items()}
    prepared = {network: extract(paths[f"proxy{network}"]) for network in (6, 9)}
    if prepared[6]["observations"].shape != (61, 202):
        raise RuntimeError(f"proxy6-shape-failed:{prepared[6]['observations'].shape}")
    if prepared[9]["observations"].shape != (61, 297):
        raise RuntimeError(f"proxy9-shape-failed:{prepared[9]['observations'].shape}")

    output.mkdir(parents=True)
    for network in (6, 9):
        scipy.io.savemat(output / f"proxy{network}_allowed_1940_2000.mat", prepared[network], do_compression=True)
    after = {name: state(path) for name, path in paths.items()}
    if before != after:
        raise RuntimeError("source-changed-during-round03-preparation")
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "allowed_proxy_inputs_prepared",
        "years": [1940, 2000],
        "locked_1921_1939_accessed": False,
        "proxy_counts": {"6": 202, "9": 297},
        "model_type_counts": {
            str(network): {
                "univariate": int(np.sum(prepared[network]["model_type"] == 1)),
                "bivariate": int(np.sum(prepared[network]["model_type"] == 2)),
            }
            for network in (6, 9)
        },
        "source_hashes_unchanged": before == after,
    }
    (output / "preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest_preparation.json").write_text(json.dumps({"before": before, "after": after, "sha256": hashes, "driver_sha256": digest(Path(__file__).resolve())}, indent=2), encoding="utf-8")
    (output / "ROUND_CONTRACT.md").write_text(
        "# Round 03 Contract\n\nDecision: confirm whether the AO-dominant precipitation candidate retains skill when PSM coefficients and residual errors are fitted without each held-out 1940-2000 block.\n\nThe preparation step reads proxy values only for 1940-2000.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
