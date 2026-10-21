"""Short-path retry config for the four batch-2 system midpoint preflights."""

from __future__ import annotations

import json

from .build_lia_191_formal_delta_bisection_batch2_v1 import CONFIG_DIR, EXPERIMENT, RESULT_ROOT


def main() -> None:
    source = CONFIG_DIR / f"{EXPERIMENT}_system_preflights_v1.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    config["pilot_design"]["block"] = "B_dt_b2_pf_v2"
    config["pilot_design"]["purpose"] = (
        "Exact four-chain midpoint retry after the v1 result-write path exceeded the "
        "Windows path limit; all scientific inputs and parameter values are unchanged."
    )
    config["execution"]["output_root"] = (
        f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT}/system_preflights_v2"
    )
    config["execution"]["work_root"] = "w191_b2pf_v2"
    path = CONFIG_DIR / f"{EXPERIMENT}_system_preflights_v2.json"
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    note = {
        "schema_version": 1,
        "supersedes": str(source),
        "failure_class": "controller-result-write-path-length",
        "failed_path_length_characters": 266,
        "scientific_inputs_changed": False,
        "parameter_values_changed": False,
        "new_block": config["pilot_design"]["block"],
        "new_output_root": config["execution"]["output_root"],
        "config": str(path),
    }
    (RESULT_ROOT / "system_preflight_short_path_retry_v2.json").write_text(
        json.dumps(note, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(note, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
