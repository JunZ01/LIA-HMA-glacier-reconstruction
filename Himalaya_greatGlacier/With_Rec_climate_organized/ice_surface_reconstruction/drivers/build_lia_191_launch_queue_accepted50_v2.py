"""Run the V1 queue builder against the promoted accepted-50 registry.

The source transformation is intentionally narrow and asserted.  This keeps
the queue construction logic identical while preserving the accepted-49 V1
audit already emitted during the live route transition.
"""

from __future__ import annotations

from pathlib import Path


SOURCE = Path(__file__).with_name("build_lia_191_launch_queue_v1.py")
TEXT = SOURCE.read_text(encoding="utf-8")
REPLACEMENTS = {
    'OUT = VALIDATION / "lia_191_launch_queue_v1"':
        'OUT = VALIDATION / "lia_191_launch_queue_v2"',
    'ACCEPTED = VALIDATION / "lia_191_accepted_49_registry_v1/accepted_49_curve_metrics.csv"':
        'ACCEPTED = VALIDATION / "lia_191_accepted_50_registry_v1/accepted_50_curve_metrics.csv"',
    'if len(accepted_ids) != 49 or 2729 not in accepted_ids:\n        raise RuntimeError("accepted-49-registry-contract-failed")':
        'if len(accepted_ids) != 50 or not {2729, 5536}.issubset(accepted_ids):\n        raise RuntimeError("accepted-50-registry-contract-failed")',
    '"contract": "accepted-49 / remaining-142"':
        '"contract": "accepted-50 / remaining-141"',
}
for old, new in REPLACEMENTS.items():
    count = TEXT.count(old)
    if count != 1:
        raise RuntimeError(f"queue-source-contract-changed:{old!r}:count={count}")
    TEXT = TEXT.replace(old, new)

namespace = {"__name__": "__main__", "__file__": str(SOURCE)}
exec(compile(TEXT, str(SOURCE), "exec"), namespace)
