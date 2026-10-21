from __future__ import annotations

import tempfile
from pathlib import Path

from run_round278_baiyu_haizi_anchor_identity_audit_v1 import run_poison_test


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        result = run_poison_test(Path(directory) / "poison")
    assert result["status"] == "passed"
    assert result["xlsx_locked_poison_decoded"] is False
    assert result["hdf_locked_poison_decoded"] is False
    print("ROUND278_SYNTHETIC_POISON_TEST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
