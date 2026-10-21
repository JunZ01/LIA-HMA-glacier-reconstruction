"""Shared pure utilities for the Stage 3 initial-state search."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timing_weight(
    year: float,
    outer_start: float,
    core_start: float,
    core_end: float,
    outer_end: float,
) -> float:
    if year < outer_start or year > outer_end:
        return 0.0
    if year < core_start:
        return (year - outer_start) / (core_start - outer_start)
    if year <= core_end:
        return 1.0
    return (outer_end - year) / (outer_end - core_end)


def weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, quantiles: np.ndarray
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.full(quantiles.shape, np.nan)
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= np.sum(weights)
    return np.interp(quantiles, cumulative, values)


def select_regional_pilot(
    routes: pd.DataFrame,
    manifest: pd.DataFrame,
    per_region: int,
    area_quantiles: list[float],
    required_ids: list[int],
) -> pd.DataFrame:
    if len(area_quantiles) != per_region:
        raise ValueError("Area quantile count must equal pilot systems per region")
    core_flag = routes["farinotti_core_191"]
    if core_flag.dtype != bool:
        core_flag = core_flag.astype(str).str.lower().eq("true")
    manifest_fields = [
        "lia_unique_id",
        "lia_area_km2",
        "centroid_lon",
        "centroid_lat",
    ]
    core = routes[core_flag].merge(
        manifest[manifest_fields],
        on="lia_unique_id",
        how="left",
        validate="one_to_one",
    )
    if core["lia_area_km2"].isna().any():
        raise ValueError("At least one core system lacks LIA area")
    required = {int(value) for value in required_ids}
    if not required.issubset(set(core["lia_unique_id"].astype(int))):
        raise ValueError("A required review system is not Stage 3 core")

    selected_rows = []
    for region_code, region in core.groupby("region_code", sort=True):
        region = region.sort_values("lia_unique_id").copy()
        chosen: set[int] = set()
        quantile_targets = region["lia_area_km2"].quantile(area_quantiles)
        for slot, target in enumerate(quantile_targets, start=1):
            available = region[~region["lia_unique_id"].isin(chosen)].copy()
            available["distance"] = np.abs(
                np.log(available["lia_area_km2"]) - math.log(float(target))
            )
            row = available.sort_values(["distance", "lia_unique_id"]).iloc[0]
            chosen.add(int(row["lia_unique_id"]))
            selected_rows.append(
                {
                    "lia_unique_id": int(row["lia_unique_id"]),
                    "region_code": int(region_code),
                    "selection_slot": f"area-q{int(round(area_quantiles[slot - 1] * 100)):02d}",
                    "selection_reason": "regional-lia-area-quantile",
                }
            )

        required_here = sorted(
            required & set(region["lia_unique_id"].astype(int))
        )
        for system_id in required_here:
            if system_id in chosen:
                index = next(
                    i
                    for i, row in enumerate(selected_rows)
                    if row["lia_unique_id"] == system_id
                )
                selected_rows[index]["selection_reason"] += ";required-review"
                continue
            required_area = float(
                region.loc[
                    region["lia_unique_id"] == system_id, "lia_area_km2"
                ].iloc[0]
            )
            replaceable = [
                (i, row)
                for i, row in enumerate(selected_rows)
                if row["region_code"] == int(region_code)
                and row["lia_unique_id"] not in required
            ]
            replace_index, replaced = min(
                replaceable,
                key=lambda item: abs(
                    math.log(
                        float(
                            region.loc[
                                region["lia_unique_id"]
                                == item[1]["lia_unique_id"],
                                "lia_area_km2",
                            ].iloc[0]
                        )
                    )
                    - math.log(required_area)
                ),
            )
            chosen.remove(int(replaced["lia_unique_id"]))
            chosen.add(system_id)
            selected_rows[replace_index] = {
                "lia_unique_id": system_id,
                "region_code": int(region_code),
                "selection_slot": replaced["selection_slot"],
                "selection_reason": (
                    "required-review;replaced-nearest-area-quantile"
                ),
            }

    selected = pd.DataFrame(selected_rows).merge(
        core,
        on=["lia_unique_id", "region_code"],
        how="left",
        validate="one_to_one",
    )
    selected["required_review"] = selected["lia_unique_id"].isin(required)
    return selected.sort_values(["region_code", "lia_unique_id"]).reset_index(
        drop=True
    )


def geometry_score(
    area_m2: float,
    target_area_m2: float,
    length_m: float | None,
    target_length_m: float | None,
    hypsometry_quantiles_m: np.ndarray | None,
    target_hypsometry_quantiles_m: np.ndarray | None,
    weights: dict[str, float],
    tolerances: dict[str, float],
) -> dict:
    components: dict[str, float] = {}
    passes: dict[str, bool] = {}
    area_log_error = abs(math.log(area_m2 / target_area_m2))
    area_sigma = math.log1p(float(tolerances["area_relative"]))
    components["area"] = math.exp(-0.5 * (area_log_error / area_sigma) ** 2)
    passes["area"] = area_log_error <= area_sigma

    length_relative_error = math.nan
    if (
        length_m is not None
        and target_length_m is not None
        and length_m > 0
        and target_length_m > 0
    ):
        length_log_error = abs(math.log(length_m / target_length_m))
        length_sigma = math.log1p(float(tolerances["length_relative"]))
        length_relative_error = abs(length_m / target_length_m - 1.0)
        components["length"] = math.exp(
            -0.5 * (length_log_error / length_sigma) ** 2
        )
        passes["length"] = length_log_error <= length_sigma

    hypsometry_rmse_m = math.nan
    if (
        hypsometry_quantiles_m is not None
        and target_hypsometry_quantiles_m is not None
    ):
        simulated = np.asarray(hypsometry_quantiles_m, dtype=float)
        target = np.asarray(target_hypsometry_quantiles_m, dtype=float)
        valid = np.isfinite(simulated) & np.isfinite(target)
        if valid.any():
            hypsometry_rmse_m = float(
                np.sqrt(np.mean((simulated[valid] - target[valid]) ** 2))
            )
            sigma = float(tolerances["hypsometry_rmse_m"])
            components["hypsometry"] = math.exp(
                -0.5 * (hypsometry_rmse_m / sigma) ** 2
            )
            passes["hypsometry"] = hypsometry_rmse_m <= sigma

    active_weight = sum(float(weights[key]) for key in components)
    if active_weight <= 0:
        raise ValueError("Geometry score has no active component")
    log_score = sum(
        float(weights[key]) / active_weight * math.log(max(score, 1e-300))
        for key, score in components.items()
    )
    return {
        "geometry_score": math.exp(log_score),
        "geometry_tolerance_pass": all(passes.values()),
        "area_log_error": area_log_error,
        "length_relative_error": length_relative_error,
        "hypsometry_rmse_m": hypsometry_rmse_m,
        "active_components": ";".join(sorted(components)),
    }
