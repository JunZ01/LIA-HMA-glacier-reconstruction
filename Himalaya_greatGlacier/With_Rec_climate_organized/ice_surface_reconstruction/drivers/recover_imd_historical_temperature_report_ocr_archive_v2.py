from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import requests


HERE = Path(__file__).resolve().parent
V1 = HERE / "recover_imd_historical_temperature_report_ocr_archive_v1.py"


def load_v1():
    spec = importlib.util.spec_from_file_location("imd_report_ocr_v1", V1)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {V1}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def robust_download(url: str, destination: Path) -> None:
    last_error: Exception | None = None
    headers = {"User-Agent": "Mozilla/5.0 LIA historical climate audit"}
    for attempt in range(1, 8):
        try:
            with requests.get(
                url,
                headers=headers,
                timeout=(30, 240),
                stream=True,
            ) as response:
                response.raise_for_status()
                with destination.open("wb") as stream:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            stream.write(block)
            if destination.stat().st_size == 0:
                raise RuntimeError(f"Zero-byte response from {url}")
            time.sleep(1.5)
            return
        except Exception as error:  # Network mirrors reset long batch connections.
            last_error = error
            if destination.exists():
                destination.unlink()
            time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"Download failed after retries: {url}") from last_error


def main() -> None:
    audit = load_v1()
    audit.OUT = (
        audit.LOOP / "temperature_historical_imd_report_ocr_archive_v2"
    )
    audit.DRIVER = Path(__file__).resolve()
    audit.download = robust_download
    audit.main()


if __name__ == "__main__":
    main()
