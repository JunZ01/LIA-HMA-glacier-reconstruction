from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def load_v1():
    path = Path(__file__).with_name("audit_cook_chronology_source_moisture_v1.py")
    spec = importlib.util.spec_from_file_location("cook_chronology_source_moisture_v1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot-load-v1:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bounded_text_value(value: object) -> str:
    if value is None or hasattr(value, "_fieldnames"):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bytes):
        return value.decode("latin-1").strip()
    array = np.asarray(value)
    if array.size == 0 or array.size > 24:
        return ""
    if array.dtype.kind in "US":
        return " ".join(str(item).strip() for item in array.reshape(-1) if str(item).strip()).strip()
    if array.dtype == object:
        parts = [bounded_text_value(item) for item in array.reshape(-1)]
        return " ".join(part for part in parts if part).strip()
    return " ".join(str(item) for item in array.reshape(-1)).strip()


def main() -> int:
    audit = load_v1()
    audit.EXPERIMENT_ID = "climate_reconstruction_cook_chronology_source_moisture_v2"
    audit.text_value = bounded_text_value
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
