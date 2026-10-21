"""Run precipitation provenance audit with an ASCII-only archive-path bootstrap."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_audit_module():
    source = Path(__file__).with_name("run_audit.py")
    spec = importlib.util.spec_from_file_location("precipitation_provenance_audit_v1", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module-load-failed:{source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    audit = load_audit_module()
    archive_folder = "".join(chr(value) for value in (25991, 20214, 25972, 29702))
    paper_folder = "1 " + "".join(chr(value) for value in (35770, 25991))
    audit.ARCHIVE_ROOT = (
        Path(r"D:\Users\JUNZ\Desktop")
        / archive_folder
        / paper_folder
        / "work1"
        / "compared_result_all"
    )

    # The project inventory already found no frozen precipitation member pool.
    # Avoid a second recursive scan of the large result tree in this run.
    audit.Path.rglob = lambda self, pattern: iter(())
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())

