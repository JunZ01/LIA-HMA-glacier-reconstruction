"""Run the network JJA experiment using the batched-PSM preflight scalar."""

from __future__ import annotations

from pathlib import Path


SOURCE = Path(__file__).with_name("run_experiment.py")


def main() -> None:
    code = SOURCE.read_text(encoding="utf-8")
    old = '''    annual_repeat = np.asarray(psm["annual_repeat_hxb"], dtype=float)
    annual = np.asarray(psm["annual_hxb"], dtype=float)
    preflight = {
        "selected_records": int(len(records)),
        "common_rng_annual_repeat_max_abs": float(np.nanmax(np.abs(annual_repeat - annual))),
'''
    new = '''    annual = np.asarray(psm["annual_hxb"], dtype=float)
    preflight_repeat = float(np.asarray(psm["annual_repeat_max_abs_preflight"]).squeeze())
    preflight = {
        "selected_records": int(len(records)),
        "common_rng_annual_repeat_max_abs": preflight_repeat,
'''
    if old not in code:
        raise RuntimeError("Expected unbatched preflight block was not found")
    code = code.replace(old, new)
    namespace = {"__name__": "__main__", "__file__": str(SOURCE)}
    exec(compile(code, str(SOURCE), "exec"), namespace)


if __name__ == "__main__":
    main()
