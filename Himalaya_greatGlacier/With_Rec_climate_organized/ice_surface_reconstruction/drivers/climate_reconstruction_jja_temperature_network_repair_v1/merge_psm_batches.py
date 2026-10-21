"""Merge independently saved JJA PSM batches after completeness checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


OUTPUT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm\validation_data\results\paleo_surface_validation_v0\climate_reconstruction_jja_temperature_network_repair_v1")
BATCHES = OUTPUT / "psm_batches"


def main() -> None:
    paths = sorted(BATCHES.glob("batch_*.mat"))
    if not paths:
        raise FileNotFoundError("No PSM batch files found")
    records = []
    columns = []
    arrays: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "baseline_hxb", "annual_hxb", "jja_hxb", "anchored_jja_hxb",
            "baseline_b", "annual_b", "jja_b", "annual_pcr", "jja_pcr",
            "annual_pval", "jja_pval", "annual_data_max_abs",
        )
    }
    annual_repeat_max_abs = np.nan
    for path in paths:
        batch = loadmat(path, squeeze_me=True)
        batch_records = np.atleast_1d(batch["record_ids"]).astype(int)
        batch_columns = np.atleast_1d(batch["columns"]).astype(int)
        records.extend(batch_records.tolist())
        columns.extend(batch_columns.tolist())
        for name in arrays:
            value = np.asarray(batch[name], dtype=float)
            if name.endswith("pcr") or name.endswith("pval") or name == "annual_data_max_abs":
                value = np.atleast_1d(value).reshape(-1, 1)
            elif value.ndim == 1:
                value = value.reshape(1, -1)
            arrays[name].append(value)
        if "annual_repeat_hxb" in batch:
            annual_repeat_max_abs = float(np.nanmax(np.abs(batch["annual_repeat_hxb"] - batch["annual_hxb"])))
    order = np.argsort(records)
    records_array = np.asarray(records, dtype=int)[order]
    columns_array = np.asarray(columns, dtype=int)[order]
    if len(records_array) != 50 or len(np.unique(records_array)) != 50:
        raise RuntimeError(f"Expected 50 unique records, found {len(records_array)} rows and {len(np.unique(records_array))} unique")
    merged = {
        "record_ids": records_array,
        "columns": columns_array,
        "rng_seed": 0,
        "annual_repeat_max_abs_preflight": annual_repeat_max_abs,
    }
    for name, parts in arrays.items():
        merged[name] = np.concatenate(parts, axis=0)[order]
    merged["jja_hxb_delta"] = merged["jja_hxb"] - merged["annual_hxb"]
    if not np.all(np.isfinite(merged["anchored_jja_hxb"])):
        raise RuntimeError("Merged anchored JJA operator contains nonfinite values")
    savemat(OUTPUT / "network_jja_psm_delta_anchor.mat", merged, do_compression=True)
    print(OUTPUT / "network_jja_psm_delta_anchor.mat")


if __name__ == "__main__":
    main()
