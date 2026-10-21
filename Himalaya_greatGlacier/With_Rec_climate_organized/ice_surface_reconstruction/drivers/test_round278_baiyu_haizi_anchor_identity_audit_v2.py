from __future__ import annotations

import tempfile
from pathlib import Path

import run_round278_baiyu_haizi_anchor_identity_audit_v2 as audit


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        result = audit.audit.run_poison_test(Path(directory) / "poison")
    assert result["status"] == "passed"
    assert result["xlsx_locked_poison_decoded"] is False
    assert result["hdf_locked_poison_decoded"] is False
    print("ROUND278_V2_SYNTHETIC_POISON_TEST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
