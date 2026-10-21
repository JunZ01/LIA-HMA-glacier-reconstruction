"""Build the post-observation-domain-gate Layer-1 delta-T extension batch."""

from __future__ import annotations

import json

from . import build_lia_191_delta_boundary_extension_batch as _base


BLOCK = "B_delta_boundary_extension_spatial_gate_v2"
EXPERIMENT_NAME = "lia_191_delta_boundary_extension_spatial_gate_v2"
FULL_CONFIG_NAME = "lia_191_delta_boundary_extension_spatial_gate_v2.json"
PREFLIGHT_CONFIG_NAME = "lia_191_delta_boundary_extension_spatial_gate_preflight_v2.json"


def main() -> None:
    config_dir = _base.CONFIG_DIR / EXPERIMENT_NAME
    output_root = (
        _base.project_root()
        / "validation_data"
        / "results"
        / "paleo_surface_validation_v0"
        / EXPERIMENT_NAME
    )
    routing_table = (
        _base.controller_root()
        / "diagnostics"
        / "iterative_loop_spatial_downstream_v2"
        / "core191_iterative_loop_table.csv"
    )
    original_config = _base._base_config_dict

    def spatial_gate_config() -> dict:
        config = original_config()
        config["execution"]["work_root"] = "w191_dtext2"
        return config

    _base.BLOCK = BLOCK
    _base.EXPERIMENT_NAME = EXPERIMENT_NAME
    _base.OUTPUT_ROOT = output_root
    _base.ROUTING_TABLE = routing_table
    _base.CONFIG_DIR = config_dir
    _base._base_config_dict = spatial_gate_config
    _base.main()

    full_path = config_dir / "lia_191_delta_boundary_extension_v1.json"
    preflight_path = config_dir / "lia_191_delta_boundary_extension_preflight_v1.json"
    full = json.loads(full_path.read_text(encoding="utf-8"))
    full["pilot_design"]["purpose"] = (
        "One-sided local delta-T extension after the observation-domain gate, "
        "with all non-delta-T controls frozen."
    )
    full_path.rename(config_dir / FULL_CONFIG_NAME)
    final_full_path = config_dir / FULL_CONFIG_NAME
    final_full_path.write_text(json.dumps(full, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight["pilot_design"]["purpose"] = (
        "One-chain preflight for the post-observation-domain-gate Layer-1 "
        "delta-T boundary-extension batch."
    )
    preflight_path.rename(config_dir / PREFLIGHT_CONFIG_NAME)
    final_preflight_path = config_dir / PREFLIGHT_CONFIG_NAME
    final_preflight_path.write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    contract_path = output_root / "experiment_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["declared_at"] = "2026-08-14"
    contract["experiment_question"] = (
        "For the post-observation-domain-gate systems routed to layer1_delta_extension, "
        "can two one-sided local delta-T values produce a direct stable 10% total-length "
        "endpoint candidate before another physical parameter is opened?"
    )
    contract["source_routing_table"] = str(routing_table.relative_to(_base.project_root()))
    contract["configuration_files"] = {
        "full": str(final_full_path.relative_to(_base.project_root())),
        "preflight": str(final_preflight_path.relative_to(_base.project_root())),
    }
    contract["work_root"] = "w191_dtext2"
    contract_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary_path = output_root / "design_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["config_full"] = str(final_full_path)
    summary["config_preflight"] = str(final_preflight_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
