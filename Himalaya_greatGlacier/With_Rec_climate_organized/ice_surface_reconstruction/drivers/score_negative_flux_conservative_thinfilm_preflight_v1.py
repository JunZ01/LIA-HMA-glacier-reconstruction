"""Apply the preregistered final gate to the 3131 thin-film zero-MB run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--zero-root", type=Path, required=True)
    args = parser.parse_args()
    state_root = args.state_root.resolve()
    zero_root = args.zero_root.resolve()
    static = json.loads((state_root / "initial_state_summary.json").read_text(encoding="utf-8"))
    zero = json.loads((zero_root / "zero_mb_summary_formal.json").read_text(encoding="utf-8"))
    if static["status"] != "initial-state-static-pass":
        raise RuntimeError("static-preflight-not-pass")
    if int(static["system_id"]) != 3131 or int(zero["system_id"]) != 3131:
        raise RuntimeError("representative-system-drift")
    final = zero["final"]
    outline = float(static["static"]["outline_area_m2"])
    registered_length = float(static["static"]["registered_lia_length_m"])
    final_area_error = abs(float(final["area_m2"]) - outline) / outline
    final_length_error = abs(float(final["length_m"]) - registered_length) / registered_length
    flags = []
    if zero["status"] != "zero-mb-pass":
        flags.append("formal-zero-mb-fail")
    if final_area_error > 0.10:
        flags.append("five-year-model-outline-area-mismatch")
    if final_length_error > 0.10:
        flags.append("five-year-root-length-mismatch")
    result = {
        "status": "representative-preflight-pass" if not flags else "representative-preflight-fail",
        "experiment": static["experiment"],
        "system_id": 3131,
        "static_summary": str(state_root / "initial_state_summary.json"),
        "zero_mb_summary": str(zero_root / "zero_mb_summary_formal.json"),
        "formal_zero_mb_status": zero["status"],
        "five_year_model_outline_area_relative_error": final_area_error,
        "five_year_root_length_relative_error": final_length_error,
        "review_flags": flags,
        "decision": "release-1642-confirmation-only" if not flags else "close-thinfilm-mechanism-no-1642",
        "steady_50y_run": False,
        "historical_run": False,
    }
    output = state_root.parent / "representative_preflight_summary.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
