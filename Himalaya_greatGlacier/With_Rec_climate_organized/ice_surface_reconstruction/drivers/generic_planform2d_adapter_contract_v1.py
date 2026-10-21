"""Shared immutable-input contract for generic planform-2D adapters."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


GENERIC_STATIC_CONFIG_SCHEMA = "lia-191-generic-planform2d-static-config-v1"
GENERIC_STATIC_GEOLOGY_SCHEMA = "lia-191-generic-planform2d-geology-v1"
GENERIC_STATIC_SUMMARY_SCHEMA = "lia-191-generic-planform2d-static-v1"
GENERIC_CLIMATE_CONFIG_SCHEMA = "lia-191-generic-historical-climate-export-config-v1"
GENERIC_CLIMATE_SUMMARY_SCHEMA = "lia-191-generic-historical-monthly-ti-bridge-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest().upper()


def expected_checkpoint_keys(target_years: int) -> set[str]:
    years = int(target_years)
    if years <= 0:
        raise ValueError("target_years-must-be-positive")
    return {str(year) for year in range(1, years + 1)}


def checkpoint_range_complete(keys: Iterable[str], target_years: int) -> bool:
    return {str(key) for key in keys} == expected_checkpoint_keys(target_years)


def validate_hashed_paths(
    project: Path, inputs: Mapping[str, Mapping[str, object]]
) -> dict[str, dict[str, object]]:
    validated: dict[str, dict[str, object]] = {}
    for label, item in inputs.items():
        raw_path = Path(str(item["path"]))
        path = raw_path if raw_path.is_absolute() else project / raw_path
        if not path.is_file():
            raise FileNotFoundError(f"{label}-missing:{path}")
        expected = str(item["sha256"]).upper()
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"{label}-hash-mismatch:{actual}:{expected}")
        validated[label] = {
            "path": str(path),
            "sha256": actual,
            "size_bytes": path.stat().st_size,
        }
    return validated


def selected_sample_positions(month_count: int) -> np.ndarray:
    count = int(month_count)
    if count <= 0:
        raise ValueError("month-count-must-be-positive")
    if count == 6372:
        return np.asarray([0, 1, 11, 12, 359, 360, 3180, 6359, 6360, 6371], dtype=np.int64)
    anchors = [0, 1, 11, 12, count // 2, count - 13, count - 12, count - 2, count - 1]
    return np.asarray(sorted({min(max(value, 0), count - 1) for value in anchors}), dtype=np.int64)


def static_schema_names(config: Mapping[str, object]) -> tuple[str, str, bool]:
    generic = config.get("schema") == GENERIC_STATIC_CONFIG_SCHEMA
    if generic:
        return GENERIC_STATIC_GEOLOGY_SCHEMA, GENERIC_STATIC_SUMMARY_SCHEMA, True
    return (
        "lia-191-igm-planform2d-system2005-geology-v1",
        "lia-191-igm-planform2d-system2005-static-v1",
        False,
    )


def climate_schema_names(config: Mapping[str, object]) -> tuple[str, bool]:
    schema = config.get("schema")
    if schema == GENERIC_CLIMATE_CONFIG_SCHEMA:
        return GENERIC_CLIMATE_SUMMARY_SCHEMA, True
    if schema == "lia-191-system2005-historical-climate-export-config-v1":
        return "lia-191-system2005-historical-monthly-ti-bridge-v1", False
    raise RuntimeError("historical-climate-config-schema-mismatch")
