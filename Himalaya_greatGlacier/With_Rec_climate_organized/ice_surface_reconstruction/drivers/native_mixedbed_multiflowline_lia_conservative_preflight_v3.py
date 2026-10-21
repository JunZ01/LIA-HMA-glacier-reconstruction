"""OGGM-1.6.3 schema shim for the Section 176/177 v1 preflight.

V2 reached mapped-state construction after successful clone/F19/native-network
initialisation, then stopped because this OGGM version exposes lambdas only as
``_lambdas``.  V3 adds that read-only compatibility property and uses a fresh
short clone/output path.  No scientific rule changes.
"""

from pathlib import Path

from oggm.core.flowline import MixedBedFlowline

from . import native_mixedbed_multiflowline_lia_conservative_preflight_v1 as v1


if not hasattr(MixedBedFlowline, "lambdas"):
    MixedBedFlowline.lambdas = property(lambda self: self._lambdas)

_original_clone_and_initialize = v1._clone_and_initialize


def _short_clone_and_initialize(*, source_root, clone_root, member_ids, suffix):
    del clone_root
    return _original_clone_and_initialize(
        source_root=source_root,
        clone_root=Path.cwd() / "w1763850v3",
        member_ids=member_ids,
        suffix=suffix,
    )


def main() -> None:
    v1._clone_and_initialize = _short_clone_and_initialize
    v1.main()


if __name__ == "__main__":
    main()
