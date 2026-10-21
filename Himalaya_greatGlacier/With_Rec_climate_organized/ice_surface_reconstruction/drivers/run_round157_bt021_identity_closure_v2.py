"""Technical v2 launcher for the BT021 identity audit.

V1 stopped before data parsing because its table-header literal omitted the
leading ``t`` in ``trsgi``. This launcher changes only that literal, the
independent result directory, and the experiment identifier in memory.
"""

from pathlib import Path


def main() -> int:
    source_path = Path(__file__).with_name("run_round157_bt021_identity_closure_v1.py")
    source = source_path.read_text(encoding="utf-8")
    replacements = {
        '"age_CE\\trsgi\\tnumsamp"': '"age_CE\\ttrsgi\\tnumsamp"',
        "climate_reconstruction_bt021_identity_closure_20260819_v1": "climate_reconstruction_bt021_identity_closure_20260819_v2",
        '"round157_bt021_identity_closure_v1"': '"round157_bt021_identity_closure_v2"',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError(f"unexpected-v1-source-count:{old}:{source.count(old)}")
        source = source.replace(old, new, 1)
    namespace = {"__name__": "round157_bt021_identity_closure_v2_core", "__file__": str(Path(__file__).resolve())}
    exec(compile(source, str(Path(__file__).resolve()), "exec"), namespace)
    return int(namespace["main"]())


if __name__ == "__main__":
    raise SystemExit(main())
