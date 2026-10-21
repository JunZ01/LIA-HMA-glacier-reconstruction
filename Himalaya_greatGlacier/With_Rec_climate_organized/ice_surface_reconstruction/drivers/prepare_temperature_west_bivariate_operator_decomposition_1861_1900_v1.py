"""Prepare exact temperature/precipitation PSM operator components."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
OUTPUT = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_temperature_west_bivariate_operator_decomposition_1861_1900_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset).reshape(-1, order="F")
    return "".join(chr(int(value)) for value in values if int(value)).strip()


def recover(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    selected = []
    temperature = []
    precipitation = []
    full = []
    b1 = []
    b2 = []
    latitude = []
    longitude = []
    errors = []
    with h5py.File(path, "r") as archive:
        references = np.asarray(archive["proxy_data/full_proxy"]).reshape(-1, order="F")
        ensemble_size = None
        for index, reference in enumerate(references):
            proxy = archive[reference]
            coefficients = np.asarray(proxy["b"], dtype=float).reshape(-1, order="F")
            if len(coefficients) != 3:
                continue
            if decode(proxy["psm_mthd"]) != "bivariate":
                raise RuntimeError(f"Unexpected three-coefficient method at record {index + 1}")
            mt = np.asarray(proxy["M_t"], dtype=float).reshape(-1, order="F")
            mp = np.asarray(proxy["M_p"], dtype=float).reshape(-1, order="F")
            hxb = np.asarray(proxy["HXb"], dtype=float).reshape(-1, order="F")
            location = np.asarray(proxy["location"], dtype=float).reshape(-1, order="F")
            if not (len(mt) == len(mp) == len(hxb)):
                raise RuntimeError(f"Operator length mismatch at record {index + 1}")
            ensemble_size = len(hxb) if ensemble_size is None else ensemble_size
            if len(hxb) != ensemble_size:
                raise RuntimeError("Inconsistent ensemble length")
            t_component = coefficients[1] * (mt - np.mean(mt))
            p_component = coefficients[2] * (mp - np.mean(mp))
            combined = t_component + p_component
            error = float(np.max(np.abs(hxb - (np.mean(hxb) + combined))))
            selected.append(index)
            temperature.append(t_component)
            precipitation.append(p_component)
            full.append(combined)
            b1.append(coefficients[1])
            b2.append(coefficients[2])
            latitude.append(location[0])
            longitude.append(location[1])
            errors.append(error)
    arrays = {
        "selected_indices_0based": np.asarray(selected, dtype=np.int64),
        "temperature_anomaly": np.asarray(temperature, dtype=float),
        "precipitation_anomaly": np.asarray(precipitation, dtype=float),
        "full_anomaly": np.asarray(full, dtype=float),
        "b1_temperature": np.asarray(b1, dtype=float),
        "b2_precipitation": np.asarray(b2, dtype=float),
        "latitude": np.asarray(latitude, dtype=float),
        "longitude": np.asarray(longitude, dtype=float),
    }
    summary = {
        "selected_bivariate_count": len(selected),
        "ensemble_size": int(ensemble_size or 0),
        "operator_reconstruction_max_abs_error": max(errors),
        "temperature_component_mean_max_abs": float(np.max(np.abs(np.mean(arrays["temperature_anomaly"], axis=1)))),
        "precipitation_component_mean_max_abs": float(np.max(np.abs(np.mean(arrays["precipitation_anomaly"], axis=1)))),
    }
    return arrays, summary


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest27 = LOOP / "round_27_temperature_innovation_gain_sign_decomposition_v1/source_manifest.json"
    recovered = json.loads(manifest27.read_text(encoding="utf-8"))["sources_before"]
    paths = {
        "proxy6": Path(recovered["proxy6"]["path"]),
        "proxy9": Path(recovered["proxy9"]["path"]),
        "round27_manifest": manifest27,
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: sha256(path) for name, path in paths.items()}
    outputs = {}
    summaries = {}
    for network in (6, 9):
        arrays, summaries[str(network)] = recover(paths[f"proxy{network}"])
        expected = 201 if network == 6 else 290
        if len(arrays["selected_indices_0based"]) != expected:
            raise RuntimeError(f"Unexpected proxy{network} bivariate count")
        if summaries[str(network)]["operator_reconstruction_max_abs_error"] > 1e-12:
            raise RuntimeError(f"proxy{network} operator closure failed")
        outputs[network] = arrays

    OUTPUT.mkdir(parents=True)
    (OUTPUT / "ROUND_CONTRACT.md").write_text(
        "# Round contract\n\n"
        "- Period for the subsequent diagnosis is 1861-1900 only.\n"
        "- Select every archived three-coefficient `bivariate` PSM and no outcome-based subset.\n"
        "- Rebuild each archived HXb anomaly exactly as b1*(M_t-mean) + b2*(M_p-mean).\n"
        "- Preserve operator means so innovations remain unchanged in every branch.\n"
        "- Original MAT files are read-only; no climate or OGGM run occurs in preparation.\n",
        encoding="utf-8",
    )
    for network, arrays in outputs.items():
        np.savez_compressed(OUTPUT / f"proxy{network}_bivariate_operator_components.npz", **arrays)
    after = {name: sha256(path) for name, path in paths.items()}
    summary = {
        "status": "complete",
        "period": [1861, 1900],
        "networks": summaries,
        "source_hashes_unchanged": before == after,
        "phase_1901_1920_accessed_or_used": False,
        "locked_1921_1939_accessed": False,
        "original_sources_modified": False,
    }
    (OUTPUT / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "prepare_source_manifest.json").write_text(json.dumps({name: {"path": str(paths[name]), "sha256": before[name]} for name in paths}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
