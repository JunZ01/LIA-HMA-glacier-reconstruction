from __future__ import annotations

import hashlib
from pathlib import Path

import run_climate_reconstruction_loop_v2_round110_unresolved_proxy_metadata_recovery_v1 as implementation


ROUND_ID = "round_110_unresolved_proxy_metadata_recovery_v3"
FROZEN_PREFIX_LINE_COUNT = 5472
FROZEN_PREFIX_SHA256 = "aa84aa86c02f6044cce20eeaac1f363d424cda07242474f0336791b6ad6bd8d2"


def main() -> int:
    project = Path(__file__).resolve().parents[4]
    facts = project / "LIA_PROJECT_FACTS.md"
    content = facts.read_bytes()
    lines = content.splitlines(keepends=True)
    if len(lines) < FROZEN_PREFIX_LINE_COUNT:
        raise RuntimeError("facts-file-shorter-than-frozen-prefix")
    prefix_hash = hashlib.sha256(b"".join(lines[:FROZEN_PREFIX_LINE_COUNT])).hexdigest()
    if prefix_hash != FROZEN_PREFIX_SHA256:
        raise RuntimeError("frozen-facts-prefix-changed-stop-and-reread")
    implementation.ROUND_ID = ROUND_ID
    implementation.FACTS_SHA256 = hashlib.sha256(content).hexdigest()
    return implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
