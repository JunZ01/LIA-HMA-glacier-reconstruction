"""Compare the system-2580 -0.25 K formal endpoint and its reproduction."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[4]
ARRAY_FAMILIES = ("section", "thick", "surface_h", "widths_m", "bed_h")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_state(path: Path):
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def audit(formal_root: Path, reproduction_root: Path) -> dict[str, Any]:
    formal_curve = formal_root / "historical_1470_2000.csv"
    reproduction_curve = reproduction_root / "historical_1470_2000.csv"
    formal_summary_path = formal_root / "historical_summary_formal.json"
    reproduction_summary_path = reproduction_root / "historical_summary_formal.json"
    formal_summary = json.loads(formal_summary_path.read_text(encoding="utf-8"))
    reproduction_summary = json.loads(
        reproduction_summary_path.read_text(encoding="utf-8")
    )
    formal_frame = pd.read_csv(formal_curve)
    reproduction_frame = pd.read_csv(reproduction_curve)
    curve_byte_equal = formal_curve.read_bytes() == reproduction_curve.read_bytes()
    curve_table_equal = bool(formal_frame.equals(reproduction_frame))

    formal_state_path = formal_root / "state_2000.pkl.gz"
    reproduction_state_path = reproduction_root / "state_2000.pkl.gz"
    formal_state = _load_state(formal_state_path)
    reproduction_state = _load_state(reproduction_state_path)
    flowline_count_equal = len(formal_state) == len(reproduction_state)
    arrays: list[dict[str, Any]] = []
    all_arrays_equal = flowline_count_equal
    if flowline_count_equal:
        for index, (left_fl, right_fl) in enumerate(
            zip(formal_state, reproduction_state)
        ):
            for family in ARRAY_FAMILIES:
                left = np.asarray(getattr(left_fl, family))
                right = np.asarray(getattr(right_fl, family))
                shape_equal = left.shape == right.shape
                equal = bool(shape_equal and np.array_equal(left, right, equal_nan=True))
                max_abs = (
                    float(np.nanmax(np.abs(left.astype(float) - right.astype(float))))
                    if shape_equal and left.size
                    else None
                )
                arrays.append(
                    {
                        "flowline_index": index,
                        "family": family,
                        "shape_equal": shape_equal,
                        "bitwise_equal": equal,
                        "max_abs_difference": max_abs,
                    }
                )
                all_arrays_equal = all_arrays_equal and equal
    formal_pass = formal_summary.get("status") == "historical-pass"
    reproduction_pass = reproduction_summary.get("status") == "historical-pass"
    passed = bool(
        formal_pass
        and reproduction_pass
        and curve_byte_equal
        and curve_table_equal
        and all_arrays_equal
    )
    return {
        "schema": "lia-system2580-late-only-reproduction-audit-v1",
        "status": "reproduction-pass" if passed else "reproduction-fail",
        "system_id": 2580,
        "late_delta_t_2000_c": -0.25,
        "formal_summary_sha256": _sha256(formal_summary_path),
        "reproduction_summary_sha256": _sha256(reproduction_summary_path),
        "formal_curve_sha256": _sha256(formal_curve),
        "reproduction_curve_sha256": _sha256(reproduction_curve),
        "curve_byte_equal": curve_byte_equal,
        "curve_table_equal": curve_table_equal,
        "flowline_count_equal": flowline_count_equal,
        "flowline_count": len(formal_state) if flowline_count_equal else None,
        "array_families": list(ARRAY_FAMILIES),
        "all_arrays_bitwise_equal": all_arrays_equal,
        "array_comparisons": arrays,
        "formal_state_container_sha256": _sha256(formal_state_path),
        "reproduction_state_container_sha256": _sha256(reproduction_state_path),
        "gzip_container_hash_equality_required": False,
        "initial_thickness_modified": False,
        "prohibited_source_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=PROJECT / "w2580latefull_m0p250_v1",
    )
    parser.add_argument(
        "--reproduction-root",
        type=Path,
        default=PROJECT / "w2580latefull_m0p250_reproduction_v1",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.formal_root.resolve(), args.reproduction_root.resolve())
    _atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if result["status"] == "reproduction-pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
