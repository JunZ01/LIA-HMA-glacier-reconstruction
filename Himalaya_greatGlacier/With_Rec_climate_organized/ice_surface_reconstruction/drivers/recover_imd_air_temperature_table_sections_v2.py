from __future__ import annotations

import importlib.util
import time
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "recover_imd_air_temperature_table_sections_v1.py"
SPEC = importlib.util.spec_from_file_location("imd_air_sections_v1", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SOURCE}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

MODULE.OUT = (
    MODULE.LOOP / "temperature_historical_imd_air_temperature_table_sections_v2"
)
MODULE.DRIVER = Path(__file__).resolve()


def robust_fetch(url: str, attempts: int = 10) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 LIA historical climate audit"},
    )
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                return response.read()
        except Exception as error:
            last_error = error
            if attempt == attempts - 1:
                break
            time.sleep(min(60, 3 * (attempt + 1)))
    raise RuntimeError(f"download failed after {attempts} attempts: {url}") from last_error


MODULE.fetch = robust_fetch

if __name__ == "__main__":
    MODULE.main()
