"""Build the immutable CPU/GPU bridge for lia-1d-gpu-batch-preflight-v2.

This is a static input transformation only: it never instantiates or advances
an OGGM dynamics model.  It validates the frozen 24-member selection, original
flowline and climate hashes, then writes a backend-neutral float64 NPZ bundle.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import pickle
from pathlib import Path
import tempfile

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS = PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0"
V1_TRANSACTION = RESULTS / "lia_1d_gpu_batch_preflight_transaction_v1/transaction.json"
V2_TRANSACTION = RESULTS / "lia_1d_gpu_batch_preflight_transaction_v2/transaction.json"
PARAMETERS = RESULTS / "d_layer_lhs_D_v1/parameter_design.csv"
STATIC_ROOT = RESULTS / "lia_1d_gpu_batch_preflight_transaction_v2/static_fixture_v1"
POINTS = (0, 3, 10, 48, 79, 142, 179, 183)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def read_parameter_rows() -> dict[tuple[int, int], dict]:
    rows = {}
    with PARAMETERS.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["system_id"]), int(row["point"]))
            if key[0] in (1044, 1545, 3584) and key[1] in POINTS:
                rows[key] = row
    if len(rows) != 24:
        raise RuntimeError(f"expected 24 parameter rows, found {len(rows)}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    v1 = load_json(V1_TRANSACTION)
    v2 = load_json(V2_TRANSACTION)
    if v2["transaction_id"] != "lia-1d-gpu-batch-preflight-v2":
        raise RuntimeError("wrong transaction")
    if sha256(V1_TRANSACTION) != v2["supersedes"]["sha256"]:
        raise RuntimeError("v1 transaction hash mismatch")
    if sha256(PARAMETERS) != v1["members"]["parameter_source"]["sha256"]:
        raise RuntimeError("parameter design hash mismatch")

    STATIC_ROOT.mkdir(parents=True, exist_ok=True)
    npz_path = STATIC_ROOT / "bridge_inputs_float64_v1.npz"
    manifest_path = STATIC_ROOT / "manifest.json"
    if (npz_path.exists() or manifest_path.exists()) and not args.overwrite:
        raise FileExistsError("static fixture exists; pass --overwrite only for deterministic verification")

    parameters = read_parameter_rows()
    systems = v1["members"]["systems"]
    max_nx = max(int(item["flowline"]["nx"]) for item in systems)
    batch = 24
    years = np.arange(1836, 2001, dtype=np.int64)
    shape = (batch, max_nx)
    bed_h = np.zeros(shape, dtype=np.float64)
    widths_m = np.ones(shape, dtype=np.float64)
    thick = np.zeros(shape, dtype=np.float64)
    valid_mask = np.zeros(shape, dtype=np.bool_)
    nx = np.zeros(batch, dtype=np.int64)
    dx_meter = np.zeros(batch, dtype=np.float64)
    climate_temp = np.zeros((batch, len(years), 12), dtype=np.float64)
    climate_prcp = np.zeros_like(climate_temp)
    climate_gradient = np.full_like(climate_temp, -0.0065)
    climate_ref_hgt = np.zeros(batch, dtype=np.float64)
    climate_years = np.broadcast_to(years, (batch, len(years))).copy()
    glen_a = np.zeros(batch, dtype=np.float64)
    melt_f = np.zeros(batch, dtype=np.float64)
    prcp_fac = np.zeros(batch, dtype=np.float64)
    temp_bias = np.zeros(batch, dtype=np.float64)
    temp_melt = np.zeros(batch, dtype=np.float64)
    system_ids = np.zeros(batch, dtype=np.int64)
    point_ids = np.zeros(batch, dtype=np.int64)
    lane_records = []

    lane = 0
    for system in systems:
        sid = int(system["lia_unique_id"])
        flowline_path = PROJECT_ROOT / system["flowline"]["path"]
        if sha256(flowline_path) != system["flowline"]["sha256"]:
            raise RuntimeError(f"flowline hash mismatch for {sid}")
        with gzip.open(flowline_path, "rb") as handle:
            flowlines = pickle.load(handle)
        if len(flowlines) != 1 or flowlines[0].__class__.__name__ != "RectangularBedFlowline":
            raise RuntimeError(f"unsupported flowline topology for {sid}")
        flowline = flowlines[0]
        n = int(flowline.nx)
        if n != int(system["flowline"]["nx"]):
            raise RuntimeError(f"nx mismatch for {sid}")
        for point in POINTS:
            row = parameters[(sid, point)]
            summary_path = RESULTS / f"d_layer_lhs_D_v1/system_{sid}/point_{point:04d}/summary.json"
            summary = load_json(summary_path)
            climate_path = Path(summary["climate"]["forward"]["climate_file"])
            if not climate_path.is_absolute():
                climate_path = PROJECT_ROOT / climate_path
            expected_climate_hash = system["climate_forward_sha256"]
            actual_climate_hash = sha256(climate_path)
            if actual_climate_hash != expected_climate_hash:
                raise RuntimeError(f"climate hash mismatch for {sid}:{point}")
            with xr.open_dataset(climate_path) as dataset:
                file_years = dataset.time.dt.year.values.reshape(-1, 12)[:, 0]
                if not np.array_equal(file_years, years):
                    raise RuntimeError(f"climate years mismatch for {sid}:{point}")
                climate_temp[lane] = dataset["temp"].values.astype(np.float64).reshape(-1, 12)
                climate_prcp[lane] = dataset["prcp"].values.astype(np.float64).reshape(-1, 12)
                climate_ref_hgt[lane] = float(dataset.attrs["ref_hgt"])

            bed_h[lane, :n] = np.asarray(flowline.bed_h, dtype=np.float64)
            widths_m[lane, :n] = np.asarray(flowline.widths_m, dtype=np.float64)
            thick[lane, :n] = np.asarray(flowline.thick, dtype=np.float64)
            valid_mask[lane, :n] = True
            nx[lane] = n
            dx_meter[lane] = float(flowline.dx_meter)
            glen_a[lane] = float(row["glen_a"])
            melt_f[lane] = float(row["melt_f"])
            prcp_fac[lane] = float(row["prcp_fac_initial"]) * float(row["prcp_fac_multiplier"])
            temp_bias[lane] = float(row["delta_t"])
            temp_melt[lane] = float(row["temp_melt"])
            system_ids[lane] = sid
            point_ids[lane] = point
            lane_records.append({
                "lane": lane,
                "cell": f"{sid}:{point}",
                "system_id": sid,
                "rgi_id": row["rgi_id"],
                "point": point,
                "nx": n,
                "dx_meter": float(flowline.dx_meter),
                "flowline_path": str(flowline_path),
                "flowline_sha256": sha256(flowline_path),
                "climate_path": str(climate_path),
                "climate_sha256": actual_climate_hash,
                "parameters": {
                    "delta_t": temp_bias[lane],
                    "melt_f": melt_f[lane],
                    "prcp_fac": prcp_fac[lane],
                    "glen_a": glen_a[lane],
                    "temp_melt": temp_melt[lane],
                },
            })
            lane += 1

    arrays = {
        "bed_h": bed_h,
        "widths_m": widths_m,
        "thick": thick,
        "valid_mask": valid_mask,
        "nx": nx,
        "dx_meter": dx_meter,
        "glen_a": glen_a,
        "melt_f": melt_f,
        "prcp_fac": prcp_fac,
        "temp_bias": temp_bias,
        "temp_melt": temp_melt,
        "climate_temp": climate_temp,
        "climate_prcp": climate_prcp,
        "climate_gradient": climate_gradient,
        "climate_ref_hgt": climate_ref_hgt,
        "climate_years": climate_years,
        "system_ids": system_ids,
        "point_ids": point_ids,
    }
    temporary = npz_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(npz_path)
    manifest = {
        "schema": "lia-1d-gpu-batch-bridge-v1",
        "transaction_id": v2["transaction_id"],
        "transaction_fingerprint_sha256": v2["transaction_fingerprint_sha256"],
        "dynamics_performed": False,
        "precision": "float64",
        "start_year": 1836,
        "end_year": 1846,
        "cell_count": batch,
        "max_nx": max_nx,
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "npz_path": str(npz_path.relative_to(PROJECT_ROOT)),
        "npz_sha256": sha256(npz_path),
        "parameter_design_sha256": sha256(PARAMETERS),
        "lanes": lane_records,
    }
    write_json_atomic(manifest_path, manifest)
    print(json.dumps({"status": "pass", "npz_sha256": manifest["npz_sha256"], "lanes": batch}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
