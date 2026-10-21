from __future__ import annotations

import importlib.util
import sys
import urllib.parse
from pathlib import Path


HERE = Path(__file__).resolve().parent
V1_PATH = HERE / "run_temperature_west_central_independent_evidence_acquisition_v1.py"


def load_v1():
    spec = importlib.util.spec_from_file_location("west_central_evidence_v1", V1_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {V1_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.urllib.request.urlparse = urllib.parse.urlparse
    return module


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_temperature_west_central_independent_evidence_acquisition_v2.py OUTPUT_DIR")
    load_v1().main(sys.argv[1])
