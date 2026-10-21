from __future__ import annotations

import json
import zlib
from pathlib import Path

import h5py
import numpy as np
import requests

from test_esgf_retrying_range_file_v1 import RetryingRangeFile, URL


OUT = Path(
    r"E:\3spyderProject\1_LIA_himalaya_oggm\validation_data\results\paleo_surface_validation_v0"
    r"\climate_reconstruction_improvement_loop_v2"
    r"\temperature_candidate_b2_esgf_raw_chunk_decode_fixture_v1"
)


def decode_shuffle(payload: bytes, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    inflated = zlib.decompress(payload)
    itemsize = dtype.itemsize
    count = int(np.prod(shape))
    if len(inflated) != count * itemsize:
        raise RuntimeError(f"inflated-size:{len(inflated)}:{count * itemsize}")
    shuffled = np.frombuffer(inflated, dtype=np.uint8).reshape(itemsize, count)
    unshuffled = shuffled.T.copy().reshape(-1)
    return np.frombuffer(unshuffled.tobytes(), dtype=dtype).reshape(shape)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    remote = RetryingRangeFile(URL)
    try:
        with h5py.File(remote, "r") as handle:
            data = handle["tas"]
            coordinate = (108, 0, 0)
            info = data.id.get_chunk_info_by_coord(coordinate)
            response = requests.get(
                URL,
                headers={"Range": f"bytes={info.byte_offset}-{info.byte_offset + info.size - 1}"},
                timeout=(30, 120),
            )
            if response.status_code != 206:
                raise RuntimeError(f"range-status:{response.status_code}")
            decoded = decode_shuffle(response.content, data.dtype, tuple(data.chunks))[0]
            reference = np.asarray(data[coordinate[0], :, :])
            difference = np.abs(decoded.astype(float) - reference.astype(float))
            result = {
                "status": "pass" if np.array_equal(decoded, reference) else "fail",
                "dtype": str(data.dtype),
                "chunk_shape": list(data.chunks),
                "compressed_bytes": int(info.size),
                "filter_mask": int(info.filter_mask),
                "compression": data.compression,
                "shuffle": bool(data.shuffle),
                "fletcher32": bool(data.fletcher32),
                "bitwise_equal": bool(np.array_equal(decoded, reference)),
                "maximum_absolute_difference": float(difference.max()),
                "sealed_climate_values_requested": 0,
                "production_modified": False,
                "enkf_run": False,
                "oggm_run": False,
            }
    finally:
        remote.close()
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
