"""Corrected runner for the fixed 50:50 diagnostic spatial-field preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import preflight_temperature_pmip_equal_blend_spatial_field_v1 as base


OUTPUT_NAME = "climate_reconstruction_temperature_pmip_equal_blend_spatial_preflight_v2"
PRIOR = base.regional_base.SOURCE_RECOVERY / "replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy"
INDICES = base.regional_base.ARCHIVE / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat"
PROXY6 = base.regional_base.ARCHIVE / "python_data/proxy_data_6.mat"
PROXY9 = base.regional_base.ARCHIVE / "python_data/proxy_data_9.mat"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_mean(
    values: np.ndarray,
    domain_mask: np.ndarray,
    lat: np.ndarray,
) -> np.ndarray:
    point_mask = domain_mask[base._hma_mask_for_weighting]
    weights2 = np.cos(np.deg2rad(lat))[:, None] * domain_mask
    weights = weights2[domain_mask]
    return np.sum(values[:, point_mask] * weights[None], axis=1) / np.sum(weights)


def main() -> int:
    base.OUTPUT = base.RESULTS / OUTPUT_NAME
    base.DRIVER = Path(__file__).resolve()
    source_paths = {"prior": PRIOR, "indices": INDICES, "proxy6": PROXY6, "proxy9": PROXY9}
    before = {name: sha256(path) for name, path in source_paths.items()}

    original_dataset = base.Dataset

    class TrackingDataset:
        def __new__(cls, *args, **kwargs):
            dataset = original_dataset(*args, **kwargs)
            if Path(str(args[0])).resolve() == base.REFERENCE.resolve():
                base._hma_mask_for_weighting = np.asarray(dataset.variables["hma_mask"][:], dtype=bool)
            return dataset

    base.Dataset = TrackingDataset
    base.weighted_mean = weighted_mean
    result = base.main()

    after = {name: sha256(path) for name, path in source_paths.items()}
    supplement = {"before": before, "after": after, "source_hashes_unchanged": before == after}
    (base.OUTPUT / "background_source_manifest_supplement.json").write_text(
        json.dumps(supplement, indent=2), encoding="utf-8"
    )
    if before != after:
        raise RuntimeError("A background source changed")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
