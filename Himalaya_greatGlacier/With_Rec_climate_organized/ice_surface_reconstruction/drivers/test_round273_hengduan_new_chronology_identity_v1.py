from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
from openpyxl import Workbook


DRIVER = Path(__file__).with_name("run_round273_hengduan_new_chronology_identity_v1.py")


def load_driver():
    spec = importlib.util.spec_from_file_location("round273_driver", DRIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError(DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    driver = load_driver()
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "poison.xlsx"
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "HLS"
        sheet.append([None, "HLSstd", "depth"])
        for year in range(1919, 2002):
            value = "LOCKED_POISON" if 1921 <= year <= 1939 else year / 1000.0
            sheet.append([year, value, 10])
        workbook.save(path)
        parsed, audit = driver.parse_modern_chronologies(path)
        assert np.isfinite(parsed["HLS"]).all()
        assert audit["HLS"]["locked_rows_skipped_before_value_access"] == 19
        assert not audit["HLS"]["locked_numeric_values_accessed"]

    candidate = np.sin(np.arange(61) / 5.0)
    exact = driver.metrics(candidate, 3.0 - 2.0 * candidate)
    independent = driver.metrics(candidate, np.cos(np.arange(61) / 3.0))
    assert exact["exact_affine_alias"] and exact["near_alias"]
    assert not independent["exact_affine_alias"] and not independent["near_alias"]
    print("ROUND273_SYNTHETIC_IDENTITY_TEST_OK")


if __name__ == "__main__":
    main()
