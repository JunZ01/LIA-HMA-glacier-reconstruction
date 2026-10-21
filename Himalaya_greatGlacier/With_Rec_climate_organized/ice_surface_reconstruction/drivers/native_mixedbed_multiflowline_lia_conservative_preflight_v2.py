"""Short-path execution shim for the Section 176/177 v1 science.

V1 stopped before static calculations because its isolated clone path exceeded
the Windows path limit.  V2 changes only that clone path; all scientific
functions and gates are imported unchanged from v1.
"""

from pathlib import Path

from . import native_mixedbed_multiflowline_lia_conservative_preflight_v1 as v1


_original_clone_and_initialize = v1._clone_and_initialize


def _short_clone_and_initialize(*, source_root, clone_root, member_ids, suffix):
    del clone_root
    return _original_clone_and_initialize(
        source_root=source_root,
        clone_root=Path.cwd() / "w1763850v2",
        member_ids=member_ids,
        suffix=suffix,
    )


def main() -> None:
    v1._clone_and_initialize = _short_clone_and_initialize
    v1.main()


if __name__ == "__main__":
    main()
