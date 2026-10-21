from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

import build_temperature_candidate_b2_allowed_input_v8 as v8
import build_temperature_candidate_b2_allowed_input_v9 as v9


v1 = v9.v1
OUT = v1.LOOP / "temperature_balanced_pooled_candidate_b2_allowed_input_v10"


def download_range_stream(url: str, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    with requests.Session() as session:
        head = session.head(url, timeout=(30, 60))
        head.raise_for_status()
        size = int(head.headers["Content-Length"])
    if part.exists() and part.stat().st_size > size:
        part.unlink()
    last: Exception | None = None
    for attempt in range(1, 13):
        offset = part.stat().st_size if part.exists() else 0
        if offset == size:
            part.replace(target)
            return size
        try:
            with requests.Session() as session:
                response = session.get(
                    url,
                    headers={"Range": f"bytes={offset}-{size - 1}"},
                    stream=True,
                    timeout=(30, 120),
                )
                if response.status_code != 206:
                    raise IOError(f"range-status:{response.status_code}")
                mode = "ab" if offset else "wb"
                with response, part.open(mode) as handle:
                    for block in response.iter_content(8 * 1024 * 1024):
                        if block:
                            handle.write(block)
            if part.stat().st_size != size:
                raise IOError(f"range-length:{part.stat().st_size}:{size}")
            part.replace(target)
            print(f"FULL RANGE COMPLETE {target.name} {size}", flush=True)
            return size
        except Exception as error:
            last = error
            current = part.stat().st_size if part.exists() else 0
            print(
                f"RETRY FULL RANGE {attempt}/12 {type(error).__name__} "
                f"{current}/{size} {target.name}",
                flush=True,
            )
            time.sleep(min(4 * attempt, 30))
    raise last if last is not None else RuntimeError("range-stream-download-failed")


def warm_one(model: str, variable: str) -> dict[str, Any]:
    v9.download = download_range_stream
    return v9.warm_one(model, variable)


def main() -> None:
    jobs = [
        ("MPI-ESM1-2-LR", "tas"),
        ("MPI-ESM1-2-LR", "pr"),
        ("MIROC-ES2L", "tas"),
        ("MIROC-ES2L", "pr"),
    ]
    warmed: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(warm_one, *job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            warmed.append(result)
            print(f"WARM COMPLETE {result['model']} {result['variable']}", flush=True)
    v1.OUT = OUT
    v1.annual_mode = v8.cached_mode
    v1.annual_pmip = v8.cached_pmip
    v1.main()
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["experiment"] = OUT.name
    summary["v10_transport"] = (
        "resumable full-file HTTP Range streaming only for files wholly inside allowed "
        "periods; exact monthly HDF5 ranges for all boundary files"
    )
    summary["parallel_worker_count"] = 4
    summary["temporary_files_location"] = str(v9.TEMP)
    summary["temporary_files_retained"] = False
    summary["warmed_variables"] = sorted(
        warmed, key=lambda row: (row["model"], row["variable"])
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if v9.TEMP.exists() and not any(v9.TEMP.iterdir()):
        v9.TEMP.rmdir()


if __name__ == "__main__":
    main()
