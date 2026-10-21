from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path


DRIVER = Path(__file__).with_name("run_iitm_homogeneous_indian_temperature_qualification_v2.py")
SPEC = importlib.util.spec_from_file_location("iitm_v2", DRIVER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def run_test() -> None:
    lines = ["107      WHIND: Western Himalaya MINIMUM TEMPERATURE (Degree Centigrade)"]
    for year in range(1901, 1951):
        if 1921 <= year <= 1939:
            payload = " ".join(["LOCKED_POISON"] * 12)
        else:
            payload = " ".join([f"{year / 100.0:.2f}"] * 12)
        lines.append(f"WHIND{year}N  {payload}")
    lines.append("107      NEIND: Northeast India MINIMUM TEMPERATURE (Degree Centigrade)")
    for year in range(1901, 1951):
        if 1921 <= year <= 1939:
            payload = " ".join(["LOCKED_POISON"] * 12)
        else:
            payload = " ".join([f"{year / 100.0:.2f}"] * 12)
        lines.append(f"NEIND{year}N  {payload}")

    converted_tokens: list[str] = []

    def guarded_converter(token: str) -> float:
        if "POISON" in token:
            raise AssertionError("Locked poison reached numeric conversion")
        converted_tokens.append(token)
        return float(token)

    with tempfile.TemporaryDirectory() as temp_dir:
        source = Path(temp_dir) / "synthetic_iitm.txt"
        source.write_text("\n".join(lines) + "\n", encoding="ascii")
        records, audit = MODULE.parse_iitm_text(source, "N", guarded_converter)

    years = sorted({row["year"] for row in records})
    expected_years = list(range(1901, 1921)) + list(range(1940, 1951))
    assert years == expected_years
    assert len(records) == 2 * len(expected_years) * 12
    assert len(converted_tokens) == 2 * len(expected_years) * 12
    assert audit["locked_rows_skipped_before_payload_access"] == 2 * 19
    assert not any(1921 <= year <= 1939 for year in audit["numeric_conversion_years"])
    print("IITM_YEAR_FIRST_SYNTHETIC_POISON_TEST_OK")


if __name__ == "__main__":
    run_test()
